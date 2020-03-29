import os
import tarfile

import cv2
import numpy as np
import seaborn as sea
import tensorflow as tf


# Code is based on DeepLab's Demo code at:
# https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb

# Areas that are considered to be navigable (Labels according to Xception network trained on ADE20K):
# If at least a single person is detected to be walking over that instance, it is considered navigable
# 4 floor
# 7 road, route
# 10 grass
# 12 pavement, sidewalk
# 29 carpet (can be vertical)
# 53 path
# 55 runway
# 92 dirt
# 102 stage (can be vertical)

class VideoReader:
    def __init__(self, source):
        self.source = source

    def __enter__(self):
        self.video_capture = cv2.VideoCapture(self.source)

        if not self.video_capture.isOpened():
            raise IOError('Cannot open video file')

        return self.video_capture

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.video_capture.release()


class NavigableAreaSegmentation:
    NAVIGABLE_REGION_LABELS = {
        # ADE20K: floor, road, grass, pavement, carpet, path, runway, dirt, stage
        "ADE20K": np.array([4, 7, 10, 12, 29, 53, 55, 92, 102]),
        # Cityscapes: road, sidewalk, terrain
        "CITYSCAPES": np.array([0, 1, 9])
    }

    NAVIGABLE_REGIONS = np.array([])

    # Load the frozen model
    # Mostly taken from Deeplab's demo code (which is currently compatible with TF1)
    def __init__(self, tarball_path, input_size, labels, dataset="ADE20K"):

        self.input_size = input_size

        self.segmentation_func = None

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if 'frozen_inference_graph' in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        # A usable concrete_function
        self.segmentation_func = \
            self.wrap_frozen_graph(
                graph_def,
                inputs='ImageTensor:0',
                outputs='SemanticPredictions:0'
            )

        if labels and len(labels) > 0:
            self.NAVIGABLE_REGIONS = np.array(labels)

        else:
            try:
                self.NAVIGABLE_REGIONS = self.NAVIGABLE_REGION_LABELS[dataset.upper()]
            except KeyError:
                self.NAVIGABLE_REGIONS = self.NAVIGABLE_REGION_LABELS["ADE20K"]

        self.NAVIGABLE_REGIONS += 1

    def determine_segments(self, image):

        # region semantic segmentation

        # The image needs to be resized to match tensor input and color channels must be rearranged
        resized_image = cv2.cvtColor(cv2.resize(image, (self.input_size, self.input_size),
                                                interpolation=cv2.INTER_LINEAR),
                                     cv2.COLOR_BGR2RGB)

        # Convert the image to a tensor since concrete functions only work with tensors
        input_tensor = tf.convert_to_tensor([np.asarray(resized_image)], dtype="uint8")
        batch_seg_map = self.segmentation_func(input_tensor)

        segmented_image = batch_seg_map[0].numpy().astype("uint8") + 1

        # The output segmentation image needs to be resized
        segmented_image = cv2.resize(segmented_image, image.shape[1::-1], interpolation=cv2.INTER_NEAREST)

        # endregion

        return segmented_image

    def iterate_video_frames(self, video_name, frames_per_update, show_result=False):

        with VideoReader(video_name) as reader:

            navigable_labels = None
            initial_frame = None
            current_frame_no = -1

            while True:
                success, frame = reader.read()

                if not success:
                    break

                current_frame_no += 1

                if current_frame_no % frames_per_update > 0:
                    continue
                else:
                    print("Updating segments at frame {}".format(current_frame_no))

                if navigable_labels is None:
                    navigable_labels = np.zeros(frame.shape[:-1], dtype="uint8")
                    initial_frame = np.copy(frame)

                segmented_frame = self.determine_segments(frame)

                # Filter predefined non-navigable labels and update the mask
                navigable_labels = np.where(np.isin(segmented_frame, self.NAVIGABLE_REGIONS),
                                            segmented_frame, navigable_labels)

                if show_result:
                    navigable_area = cv2.bitwise_and(src1=frame,
                                                     src2=frame,
                                                     mask=navigable_labels)

                    unified_view = np.concatenate((navigable_area, cv2.cvtColor(navigable_labels, cv2.COLOR_GRAY2BGR)),
                                                  axis=1)

                    cv2.imshow("Current navigable area", unified_view)
                    cv2.waitKey(1)

        return navigable_labels, initial_frame

    def process_image(self, image_name, detection_file, threshold,
                      ground_truth=None, show_result=False, save_results=False):

        image = cv2.imread(image_name)

        segmented_image = self.determine_segments(image)

        navigable_labels = np.zeros(image.shape[:-1], dtype="uint8")
        navigable_labels = np.where(np.isin(segmented_image, self.NAVIGABLE_REGIONS),
                                    segmented_image, navigable_labels)

        navigable_mask, colored_debug_image = self.extract_navigable_areas(navigable_labels,
                                                                           detection_file, threshold)

        cv2.cvtColor(navigable_mask, cv2.COLOR_GRAY2BGR, navigable_mask)

        navigable_area = cv2.bitwise_and(src1=image,
                                         src2=image,
                                         dst=None,
                                         mask=navigable_mask)

        if show_result:
            unified_view = np.concatenate((navigable_area, navigable_mask),
                                          axis=1)

            cv2.imshow("Current navigable area", unified_view)
            cv2.waitKey(0)

        if save_results:
            self.output_results(image_name, navigable_mask, navigable_labels, navigable_area, colored_debug_image)

        if ground_truth:
            gt_image = cv2.imread(ground_truth, 0)
            self.assess_segmentation(navigable_mask, gt_image)

        return navigable_mask, navigable_labels, navigable_area, colored_debug_image

    def process_video(self, video_name, detection_file, threshold, frames_per_update=1,
                      ground_truth=None, show_result=False, save_results=False):

        # For each frame, determine segments, filter out non-navigable regions and update for each frame
        # after the video is processed, for each region, use pedestrian data to determine its navigability
        navigable_labels, initial_frame = self.iterate_video_frames(video_name, frames_per_update, show_result)

        navigable_mask, colored_debug_image = self.extract_navigable_areas(navigable_labels,
                                                                           detection_file, threshold)

        cv2.cvtColor(navigable_mask, cv2.COLOR_GRAY2BGR, navigable_mask)

        navigable_area = cv2.bitwise_and(src1=initial_frame,
                                         src2=initial_frame,
                                         dst=None,
                                         mask=navigable_mask)
        if save_results:
            self.output_results(video_name, navigable_mask, navigable_labels, navigable_area, colored_debug_image)

        if ground_truth:
            gt_image = cv2.imread(ground_truth, 0)
            self.assess_segmentation(navigable_mask, gt_image)

        return navigable_mask, navigable_labels, navigable_area, colored_debug_image

    '''
    The navigable areas extraction algorithm
    -   Get filtered, potential navigable areas image. 
    -   Identify the instances of each label as they are going to be considered independently
    -   Create an empty list that will hold the navigable labels, where number of occurrences in it represents number
    of frames they are identified as navigable
    -   For every second (fps), read the frame
    -   Extract the location of the pedestrian as image coordinate (normalized, as the image is currently scaled down)
    -   Create a set for current frame, which contains labels of segments that are being navigated by at least 1 pedestrian
    -   After each frame, add the set of labels to overall list of labels
    -   In the end, count number of frames where each segment is identified as navigable and calculate its percentage 
    to total number of frames, consider ones that are below the threshold as noise
    -   Apply the mask onto the segments to obtain final navigable areas as a binary image (as white)
    '''

    def extract_navigable_areas(self, segmented_img, pedest_loc, threshold):

        # Canvas image is used to draw red circles on in order to debug the result better
        canvas_img = np.copy(segmented_img)
        canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_GRAY2BGR)

        prev_frame = 0
        navigable_labels = []

        # Only a certain classes are accepted as navigable and each cluster of regions are considered
        # independently, any instance that is suppose to be navigable can be non-navigable if none of
        # the pedestrians have navigated on it. This is applied to prevent any false positive navigable
        # regions as much as possible. The cases where an obvious area is missed are acceptable when compared
        # to flying pedestrians

        # The image segments are labeled according to their navigability. If non-navigable, it is set to 0
        # For each instance, a new label is assigned

        # Read the segmented image and check the labels that are within, to see if they are actually distinct
        # Other than what is seems on the resulting image

        # region cluster_process

        # After filtering the image, determine individual clusters using connected components approach

        global_label = 0

        label_list = np.unique(segmented_img)
        label_list = label_list[label_list > 0]
        new_labels = np.zeros_like(segmented_img)

        # Determine label instances
        for label in label_list:
            # Obtain label contours
            current_label_image = np.zeros_like(segmented_img)

            rows, cols = np.where(segmented_img == label)
            current_label_image[rows, cols] = 255

            num_of_clusters, clusters = cv2.connectedComponents(current_label_image)

            new_labels[rows, cols] = clusters[rows, cols] + global_label
            global_label += np.max(clusters)

        segmented_img = np.reshape(new_labels, segmented_img.shape).astype("uint8")
        cv2.imshow("Instance labels", segmented_img)
        cv2.waitKey(0)

        palette = sea.color_palette('deep', np.unique(global_label).max() + 1)
        colors = [self.normalize_color(palette[x]) if x > 0 else (0.0, 0.0, 0.0) for x in
                  new_labels.flatten()]

        canvas_img = np.reshape(colors, canvas_img.shape).astype("uint8")

        # endregion

        # Open the pedestrian locations file and parse it
        with open(pedest_loc) as pedestrians:
            for i, line in enumerate(pedestrians):
                locations = line.split(',')
                feet = list(map(float, locations[-1].split('/')))

                frame_index = int(locations[0])
                if frame_index > prev_frame:
                    prev_frame = frame_index

                # Find the label value there, in order to add it to the navigable list
                try:  # Some trackers might go over bounds
                    pixel_value = segmented_img[int(feet[1])][int(feet[0])]
                except IndexError:
                    continue

                # Ignore noisy areas
                if not pixel_value == 0:
                    navigable_labels.append(pixel_value)
                    # Put a dot at the feet position for debugging the detections' location
                    cv2.circle(canvas_img, (int(feet[0]), int(feet[1])), 3,
                               (255, 255, 255))

        navigable_labels_freq = np.bincount(navigable_labels)
        navigable_labels = list(zip(list(set(navigable_labels)), navigable_labels_freq[navigable_labels]))

        try:
            print("All labels before filtering: {}".format(navigable_labels))
            print("Applying the threshold {}".format(threshold))
            # Detection per frame is used to filter number of navigations in the area
            navigable_func = lambda x: (float(x[1]) / prev_frame) >= threshold
            navigable_labels = list(zip(*list(filter(navigable_func, navigable_labels))))[0]
        except IndexError:
            navigable_labels = []

        print("Number of frames processed {}".format(prev_frame))
        print("Navigable labels: {}".format(navigable_labels))

        # In the end, create our mask and multiply it with segmentation result
        mask = np.where(np.isin(segmented_img, navigable_labels),
                        255, 0).astype("uint8")

        cv2.imshow("Pedestrian Steps", canvas_img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

        return mask, canvas_img

    # Taken from: https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
    @staticmethod
    def wrap_frozen_graph(graph_def, inputs, outputs):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph
        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs))

    @staticmethod
    def normalize_color(color):
        return [255 * channel for channel in color]

    @staticmethod
    def assess_segmentation(segmented_img, ground_truth):

        # determine number of positives in segmented
        num_of_navigable_segmented = cv2.countNonZero(segmented_img)
        num_of_navigable_gt = cv2.countNonZero(ground_truth)

        # and the images and calculate number of true elements
        img_and = cv2.bitwise_and(segmented_img, ground_truth)
        tp = cv2.countNonZero(img_and)

        fp = num_of_navigable_segmented - tp
        fn = num_of_navigable_gt - tp

        dice = (2 * tp) / (tp + fp + tp + fn)
        print("Dice index: {}".format(dice))

        return dice

    @staticmethod
    def output_results(file_name, navigable_mask, navigable_labels, navigable_area, colored_debug_image):

        # Navigable instances of regions are written as masks for mesh generation
        output_folder = os.path.dirname(file_name)
        output_filename = os.path.splitext(os.path.basename(file_name))[0] + ".png"

        print("Writing results to {}".format(output_folder))

        cv2.imwrite(os.path.join(output_folder, "mask_" + output_filename), navigable_mask)
        cv2.imwrite(os.path.join(output_folder, "labels_" + output_filename), navigable_labels)
        cv2.imwrite(os.path.join(output_folder, "area_" + output_filename), navigable_area)
        cv2.imwrite(os.path.join(output_folder, "colored_" + output_filename), colored_debug_image)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Apply deep learning based segmentation algorithms to the given image",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-v', '--video', help="Video to segment iteratively", default=None)
    parser.add_argument('-i', '--image', help="Image file to segment, ignored if video is given", default=None)
    parser.add_argument('-n', '--network', help="Path to tarball which contains the frozen inference graph")
    parser.add_argument('-l', '--labels', help="Navigable labels (integers)", nargs='+', type=int)
    parser.add_argument('-d', '--dataset', help="Dataset name for predefined labels (unused if labels are given)",
                        type=str)
    parser.add_argument('-s', '--size', help="Input size dimension (Size x Size)")
    parser.add_argument('-f', '--ped_file', help="Detection file used for determining navigable areas")
    parser.add_argument('-t', '--threshold', help="Threshold percentage of frames area should be navigated by at least "
                                                  "one pedestrian", type=float, default=0.7)
    parser.add_argument('-u', '--frames_per_update', help="Segmentation update frequency (unused for image)",
                        type=int, default=30)
    parser.add_argument('-g', '--ground_truth', help="Give ground truth segmentation for dice index calculation",
                        default=None)
    parser.add_argument('--display_results', help="Display results", const=True,
                        default=False, nargs='?')
    parser.add_argument('--save_results', help="Save results", const=True,
                        default=False, nargs='?')
    args = parser.parse_args()

    pedestrian_detection_data = args.ped_file

    # Create the network object instance
    segmentation = NavigableAreaSegmentation(args.network, int(args.size), args.labels, args.dataset)

    try:
        # If video, run the network for each frame and combine the results
        if args.video and os.path.exists(args.video):
            segmentation.process_video(args.video, pedestrian_detection_data, args.threshold,
                                       args.frames_per_update,
                                       args.ground_truth, show_result=args.display_results,
                                       save_results=args.save_results)
        elif args.image and os.path.exists(args.image):
            # If image, directly process it
            segmentation.process_image(args.image, pedestrian_detection_data, args.threshold, args.ground_truth,
                                       show_result=args.display_results, save_results=args.save_results)

        else:
            print("No media is provided, exiting...")

    except FileNotFoundError:
        print("File is not found. Ignoring...")

import os
import cv2
import numpy as np

import tensorflow as tf
import tarfile

from os.path import join

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

class SemanticNavigableAreaSegmenter(object):

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    # floor, road, grass, pavement, carpet, path, runway, dirt, stage, (person)
    NAVIGABLE_REGIONS = [4, 7, 10, 12, 29, 53, 55, 92, 102]

    #Load the frozen model weights
    #Directly taken from Deeplab's demo code
    def __init__(self, tarball_path, input_size):

        self.input_size = input_size

        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')


        self.sess = tf.Session(graph=self.graph)

    def run(self, img_name, detection_file, threshold):

        base_image_name = os.path.basename(img_name)
        image_folder = os.path.dirname(img_name)

        print("Processing image " + base_image_name)
        org_image = cv2.imread(img_name)

        #region semantic segmentation

        #The image needs to be resized  to match tensor input and color channels must be rearranged
        resized_image = cv2.cvtColor(cv2.resize(org_image,
                                                (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR),
                                     cv2.COLOR_BGR2RGB)

        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})

        seg_map = batch_seg_map[0].astype("uint8")

        # The output segmentation image needs to be resized
        seg_map = cv2.resize(seg_map, org_image.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        print("Writing to: " + os.path.join(image_folder, "area_" + base_image_name))
        cv2.imwrite(os.path.join(image_folder, "area_" + base_image_name), seg_map)

        #endregion

        #Navigable instances of regions are written as masks for mesh generation

        #region navigable

        navigable_mask = self.extract_navigable_areas(seg_map, org_image, detection_file, threshold)


        cv2.imshow("Navigable mask " + base_image_name, navigable_mask)
        navigable_area = cv2.bitwise_and(src1=org_image,
                                         src2=org_image,
                                         dst=None,
                                         mask=navigable_mask)
        cv2.imshow("Navigable area " + base_image_name, navigable_area)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(image_folder, "mask_" + base_image_name), navigable_mask)
        cv2.imwrite(os.path.join(image_folder, "area_" + base_image_name), navigable_area)

    '''
    The navigable areas extraction algorithm
    -   Determine the navigable areas in the semantically segmented image. For example: wall is non-navigable but road is
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

    def extract_navigable_areas(self, segmented_img, original_img, pedest_loc, threshold):

        # Canvas image is used to draw red circles on in order to debug the result better
        canvas_img = np.copy(segmented_img)
        canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Grayscale Segments", segmented_img)
        cv2.waitKey(0)

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

        globalLabel = 0

        label_list = np.unique(segmented_img)
        new_labels = np.zeros_like(segmented_img)

        #Filter non-navigable labels beforehand
        label_list = np.intersect1d(label_list, self.NAVIGABLE_REGIONS)
        non_navigables = np.where(segmented_img not in label_list)
        segmented_img[non_navigables] = 0

        for label in label_list:

            # Obtain label contours
            current_label_image = np.zeros_like(segmented_img)

            rows, cols = np.where(segmented_img == label)
            current_label_image[rows, cols] = 255

            num_of_clusters, clusters = cv2.connectedComponents(current_label_image)

            rows, cols = np.where(clusters != 0)
            new_labels[rows, cols] = clusters[rows, cols] + globalLabel
            globalLabel += np.max(clusters)

        segmented_img = np.reshape(new_labels, segmented_img.shape).astype("uint8")
        cv2.imshow("Instance labels", segmented_img)
        cv2.waitKey(0)

        #endregion

        # Open the pedest_loc file and start parsing
        with open(pedest_loc) as pedestrians:
            for i, line in enumerate(pedestrians):
                locations = line.split(',')
                feet = list(map(float, locations[-1].split('/')))

                frame_index = int(locations[0])
                if frame_index > prev_frame:
                    prev_frame = frame_index

                # Find the label value there, in order to add it to the navigable list
                try: #Some trackers might go over bounds
                    pixelValue = segmented_img[int(feet[1])][int(feet[0])]
                except IndexError:
                    continue

                # Ignore noisy areas
                if not pixelValue == 0:
                    navigable_labels.append(pixelValue)
                    # Put a dot at the feetpos for debugging the detections' location
                    cv2.circle(canvas_img, (int(feet[0]),int(feet[1])), 2,
                               (0, 0, 255))

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
        mask = np.zeros(segmented_img.shape, dtype="uint8")
        for label in navigable_labels:
            rows, cols = np.where(segmented_img == label)
            mask[rows, cols] = 255

        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR, mask)

        cv2.imshow("Pedestrian Steps", canvas_img)
        cv2.waitKey(0)

        return mask


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Apply segmentation algortihms (DB, HDBSCAN, Meanshift) to the given image"
                                                 "on different color formats (RGB, HSV, LAB, LUV)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument('-i', '--image', help="Image file or folder to segment")
    parser.add_argument('-n', '--network', help="Path to tarball which contains the frozen inference graph")
    parser.add_argument('-s', '--size', help="Input size dimension (Size x Size)")
    parser.add_argument('-f', '--ped_file', help="Detection file used for determining navigable areas")
    parser.add_argument('-t', '--threshold', help="Percentage of frames area should be navigated by at least one pedestrian",
                        default=0.7)
    args = parser.parse_args()

    pedestrian_detection_data = args.ped_file

    #Create the network object instance
    segmenter = SemanticNavigableAreaSegmenter(args.network, int(args.size))

    #args.image = ".\\test_images\\tf_test\\ade_label_test"

    try:
        # If image, directly call the method
        if os.path.isfile(args.image):
            segmenter.run(args.image, pedestrian_detection_data, float(args.threshold))
        else:
            # If folder, call  the method for every image in it (only goes one level,
            # doesn't recursively search for images)
            print("Processing folder " + args.image)
            file_list = [args.image + os.path.sep + file for file in os.listdir(args.image) if
                         os.path.isfile(join(args.image, file))]
            for image in file_list:
                segmenter.run(image, pedestrian_detection_data, float(args.threshold))
    except FileNotFoundError:
        print(args.image + " is not found. Ignoring...")

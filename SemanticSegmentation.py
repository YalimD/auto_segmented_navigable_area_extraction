import os
import cv2
import numpy as np

import tensorflow as tf
import tarfile

from os.path import join

# Code is similar to DeepLab's Demo code at:
# https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb

class SemanticNavigableAreaSegmenter(object):

    INPUT_TENSOR_NAME = 'ImageTensor:0'

    #TODO: We may need another tensor output as it might contain instance information
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    #Load the frozen model weights
    def __init__(self, tarball_path):
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


    '''
    The navigable areas extraction algorithm
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

    def extract_navigable_areas(self, segmented_img, original_img, pedest_loc):

        # Canvas image is used to draw red circles on in order to debug the result better
        canvas_img = np.copy(segmented_img)
        canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Grayscale Segments", segmented_img)
        cv2.waitKey(5)

        prev_frame = 0
        navigable_labels = []

        # Read the segmented image and check the labels that are within, to see if they are actually distinct
        # Other than what is seems on the resulting image

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
            #Detection per frame is used to filter number of navigations in the area
            # Parameter
            navigable_func = lambda x: (float(x[1]) / prev_frame) > 0.8
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

    def run(self, img_name, detection_file):

        base_image_name = os.path.basename(img_name)
        print("Processing image " + base_image_name)
        org_image = cv2.imread(img_name)

        #region semantic segmentation

        #The image needs to be resized  to match tensor input and color channels must be rearranged
        resized_image = cv2.cvtColor(cv2.resize(org_image,
                                                (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_LINEAR),
                                     cv2.COLOR_BGR2RGB)

        # TODO: What about others (instances)?
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})

        seg_map = batch_seg_map[0]
        other_seg_map = batch_seg_map[0]

        cv2.namedWindow("Segmentation Result")
        cv2.imshow("Segmentation Result", seg_map.astype("uint8"))
        cv2.waitKey(0)

        # The output segmentation image needs to be resized
        seg_map = cv2.resize(seg_map, org_image.shape[:2], interpolation=cv2.INTER_LINEAR)

        #endregion

        #Result obtained and regions are identified using tensor output (instances)
        #Appearently, the output must contain multiple levels of output. we shall test it

        navigable_mask = self.extract_navigable_areas(seg_map, org_image, detection_file).astype("uint8")

        #region navigable

        cv2.imshow("Navigable mask " + base_image_name, navigable_mask)
        navigable_area = cv2.bitwise_and(src1=org_image,
                                         src2=org_image,
                                         dst=None,
                                         mask=navigable_mask)
        cv2.imshow("Navigable area " + base_image_name, navigable_area)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        cv2.imwrite("mask_" + base_image_name, navigable_mask)
        cv2.imwrite("area_" + base_image_name, navigable_area)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Apply segmentation algortihms (DB, HDBSCAN, Meanshift) to the given image"
                                                 "on different color formats (RGB, HSV, LAB, LUV)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument('-i', '--image', help="Image file or folder to segment")
    parser.add_argument('-t', '--network', help="Path to tarball which contains the frozen inference graph")
    parser.add_argument('-f', '--ped_file', help="Detection file used for determining navigable areas")
    args = parser.parse_args()

    pedestrian_detection_data = args.ped_file

    #Create the network object instance
    segmenter = SemanticNavigableAreaSegmenter(args.network)

    try:
        # If image, directly call the method
        if os.path.isfile(args.image):
            segmenter.run(args.image, pedestrian_detection_data)
        else:
            # If folder, call  the method for every image in it (only goes one level,
            # doesn't recursively search for images)
            print("Processing folder " + args.image)
            file_list = [args.image + os.path.sep + file for file in os.listdir(args.image) if
                         os.path.isfile(join(args.image, file))]
            for image in file_list:
                segmenter.run(image, pedestrian_detection_data)
    except FileNotFoundError:
        print(args.image + " is not found. Ignoring...")

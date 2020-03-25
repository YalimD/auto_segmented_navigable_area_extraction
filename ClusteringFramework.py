import os
import cv2
import sys
import time
import hdbscan
import numpy as np
import seaborn as sea
import sklearn.cluster as cluster

from os.path import join
from collections import OrderedDict
from sklearn.preprocessing import normalize

'''
The testing step includes the comparison of the clustering algorithms on the given
image segmentation task. Each result is visualized in a grid where the difference can be 
easily seen

The tested algorithms are: (All tested in RGB,LAB,HSV and LUV color spaces)
- Meanshift
- DBSCAN
- HDSCAN

All algorithms are tested in Python environment, and their times are also recorded. 

Comparing with: http://www.cs.unc.edu/~jtighe/Papers/ECCV10/siftflow/baseFinal.html
'''

def normalizeColor(color):
    return [255 * channel for channel in color]


# Spatial and color features are extracted from image and prepared for clustering
def generate_features(imageOI):

    colorChannelWidth = imageOI.shape[2] if len(imageOI.shape) > 2 else 1
    height, width  = imageOI.shape[:2]

    # Spatial features doesn't vary on the image_channels as they are not related to color spaces
    yy, xx = np.meshgrid(np.arange(0, width), np.arange(0, height))
    if debugging:
        print("{} and {}".format(yy, xx))

    spatialFeatures = np.vstack([xx.ravel(), yy.ravel()]).T  # Column ordered elements

    if colorChannelWidth > 1:
        colorFeatures = np.vstack(list(map(lambda i: imageOI[:, :, i].ravel(), range(colorChannelWidth)))).T
    else:
        colorFeatures = np.vstack([imageOI[:, :].ravel()]).T


    # Create the data as 5 dimensional np array that contains color and spatial features
    features = np.zeros((width * height, colorChannelWidth + 2))

    features[:, 0:2] = spatialFeatures
    features[:, 2:] = colorFeatures

    features = normalize(features, axis = 0, norm="max")

    return features


# Test all algorithms, draw results color images. Labels are considered as pixel values
def generate_segments(data, img, img_name, colorSpace):

    algorithms = OrderedDict((
                              ('DB', cluster.DBSCAN),
                              ('HDB', hdbscan.HDBSCAN),
                              ('Mean', cluster.MeanShift),
                            ))
    kwds = [0] * len(algorithms.keys())

    # We cannot use flatImage as it is missing spatial information
    # We need to first filter the image using the meanshift THEN CLUSTER MANUALLY using data local maximas

    # Bin seeding speeds up the code as the initial locations of the kernels are in the discretized versions of points
    # We estimate the required bandwidth using sklearn's method
    # Parameter: PARAMETERS ARE OPEN TO ADJUSTMENTS

    # Putting emphasis in color
    mean_data = data

    kwds[list(algorithms.keys()).index('DB')] = {'eps': 0.015, 'min_samples': 6, 'n_jobs': -1}
    kwds[list(algorithms.keys()).index('HDB')] = {'min_cluster_size': 175}
    kwds[list(algorithms.keys()).index('Mean')] = {'bandwidth': cluster.estimate_bandwidth(X=mean_data,
                                                                                           n_samples=int(
                                                                                                img.shape[0] * img.shape[1] / 100),
                                                                                           quantile=0.1,
                                                                                           n_jobs=-1),
                                                   'n_jobs': -1,
                                                   'bin_seeding': True}

    cluster_results = [0] * len(algorithms.keys())
    file_names = [0] * len(algorithms.keys())

    for i, v in enumerate(algorithms):
        start_time = time.time()
        print("Starting {}".format(str(algorithms[v])))

        if v == 'Mean':
            clustering = cluster.MeanShift(**kwds[i]).fit(mean_data)
            labels = clustering.labels_

            print("Meanshift bandwith is: {}".format(kwds[i]['bandwidth']))

            #region cluster_process

            #After filtering the image, determine individual clusters using connected components approach

            globalLabel = 0

            label_list = np.unique(labels)
            label_image = np.reshape(labels, img.shape[:2]).astype("uint8")
            new_labels = np.zeros_like(label_image)

            for label in label_list:

                # Obtain label contours
                current_label_image = np.zeros_like(label_image)

                rows, cols = np.where(label_image == label)
                current_label_image[rows, cols] = 255

                from skimage import measure

                all_labels = measure.label(current_label_image, connectivity=1)

                rows, cols = np.where(all_labels != 0)
                new_labels[rows, cols] = all_labels[rows, cols] + globalLabel
                globalLabel += np.max(all_labels)

            #endregion

            labels = np.reshape(new_labels, labels.shape).astype("uint8")

        # MST and DENDROGRAM plotting area
        # elif v == 'HDB':
        #     clustering = hdbscan.HDBSCAN(**kwds[i], gen_min_span_tree=True)
        #     clustering.fit(data)
        #
        #     import seaborn as sns
        #     clustering.minimum_spanning_tree_.plot(edge_cmap='viridis',
        #                                           edge_alpha=0.6,
        #                                           node_size=80,
        #                                           edge_linewidth=2)
        #
        #     clustering.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())

        else:
            labels = algorithms[v](**kwds[i]).fit_predict(data)

        end_time = time.time()

        print("Algorithm {} took {} seconds on {} image ".format(str(algorithms[v]), str(end_time - start_time),
                                                                 colorSpace))

        # Grayscale resulting image
        cluster_results[i] = np.reshape(labels, img.shape[:2]).astype("uint8")
        cluster_results[i] = np.amax(cluster_results[i]) - cluster_results[i]

        file_names[i] = img_name[:img_name.rfind(".")] + "_" + str(v) + "_" + colorSpace

        # Colored segments for debugging. Some segments may have the same color value, but they are actually different
        # as seen in the grayscale results; where pixel value is directly its corresponding label
        if debugging:

            palette = sea.color_palette('deep', np.unique(labels).max() + 1)
            colors = [normalizeColor(palette[x]) if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

            result = np.reshape(colors, img.shape).astype("uint8")

            print("Number of clusters: {}", np.unique(labels))

            cv2.imshow("Color Result of {} algorithm on {} image".format(str(algorithms[v]), colorSpace), result)
            cv2.imwrite("{}_segments{}".format(file_names[i], img_name[img_name.rfind("."):]),result)
            cv2.waitKey(5)

            cv2.imshow("Original", img)
            cv2.waitKey(0)
            # cv2.destroyAllWindows()

    return cluster_results, file_names


# Prepare the image for clustering and extract the results
def process_image(img_name, detection_file,  resizing=1):

    print("Processing image " + img_name)
    org_image = cv2.imread(img_name)
    # Image order
    image_order = {'RGB': 0, 'LAB': 1, 'HSV': 2, 'LUV': 3}

    image_channels = [0] * len(image_order.keys())

    # Resizing is done to accelerate the clustering process
    resized_image = cv2.resize(org_image, None, fx=resizing, fy=resizing)

    if resized_image.shape[2] == 4:  # BGRA conversion
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2BGR)
    image_channels[image_order['LAB']] = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)

    # Normal hsv: Hue range 0-180, here 0-360
    image_channels[image_order['HSV']] = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV_FULL)
    image_channels[image_order['LUV']] = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LUV)
    image_channels[image_order['RGB']] = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Color features are created for each color space separately
    features = [0] * len(image_order.keys())
    clustering_results = [0] * len(image_order.keys())

    for i, v in enumerate(image_order):
        imageOI = image_channels[i]

        features[i] = generate_features(imageOI)
        clustering_results[i], fileNames = generate_segments(features[i], imageOI, img_name, v)

        for i, result in enumerate(clustering_results[i]):
            navigable_mask = extract_navigable_areas(result, org_image, resizing, detection_file).astype("uint8")

            # Resize back to original size
            navigable_mask = cv2.resize(navigable_mask, None, fx=1 / resizing, fy=1 / resizing,
                                        interpolation=cv2.INTER_LINEAR)

            org_image = cv2.imread(img_name)
            print("Writing to file {}".format(fileNames[i]))

            #region navigable

            cv2.imshow(fileNames[i] + img_name[img_name.rfind("."):], navigable_mask)
            navigable_area = cv2.bitwise_and(src1=org_image,
                                             src2=org_image,
                                             dst=None,
                                             mask = navigable_mask)
            cv2.imshow(fileNames[i] + "_colored" + img_name[img_name.rfind("."):], navigable_area)
            cv2.waitKey(0)

            cv2.destroyAllWindows()
            cv2.imwrite(fileNames[i]  + img_name[img_name.rfind("."):] , navigable_mask)
            cv2.imwrite(fileNames[i] + "_colored" + img_name[img_name.rfind("."):], navigable_area)

            #endregion




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


def extract_navigable_areas(segmented_img, original_img, resizing, pedest_loc):
    if debugging:
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
                pixelValue = segmented_img[int(feet[1] * resizing)][int(feet[0] * resizing)]
            except IndexError:
                continue

            # Ignore noisy areas
            if not pixelValue == 0:
                navigable_labels.append(pixelValue)
                # Put a dot at the feetpos for debugging the detections' location
                if debugging:
                    cv2.circle(canvas_img, (int(feet[0]* resizing),int(feet[1] * resizing)), 2,
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

    if debugging:
        cv2.imshow("Pedestrian Steps", canvas_img)
        cv2.waitKey(0)

    return mask



# TODO: What about foreground segments ?
# - Allow user to click on segments that are going to be used as foreground objects
# - Create a mask from them too
# - Convert the mask into a binary image and apply it to Unity for obstruction
# - https://answers.unity.com/questions/1163458/use-an-object-as-a-mask-for-a-camera.html
# - POSSIBLE IMPROVEMENT we can predict the next location of the pedestrian using cnn and consider the last
# location where we were able to track, the segment inside the vector of prediction and last seen is considered
# as a foreground segment
# - But of course, we could use a cnn (as in that paper), to semantically label everything except ground and consider
# certain labels as foreground,WE CAN EVEN PLACE THEM ON OUR MESH AS OBSTACLES WHICH WOULD INCREASE REALISM
def define_foreground(segmented_img):
    pass


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Apply segmentation algortihms (DB, HDBSCAN, Meanshift) to the given image"
                                                 "on different color formats (RGB, HSV, LAB, LUV)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument('-i', '--image', help="Image file or folder to segment")
    parser.add_argument('-f', '--ped_file', help="Detection file used for determining navigable areas")
    parser.add_argument('-d', '--debug', help="Debugging on or off", const= True,
                        default=False, nargs='?')
    parser.add_argument('--useSpatial', help="Use spatial features in segmentation", const= True,
                        default=True, nargs='?')
    parser.add_argument('-r', '--resize', help="Amount of resizing applied to image", default=0.25)
    args = parser.parse_args()

    pedestrian_detection_data = args.ped_file

    debugging = args.debug
    useSpatialFeatures = args.useSpatial

    try:
        # If image, directly call the method
        if os.path.isfile(args.image):
            process_image(args.image, pedestrian_detection_data, args.resize)
        else:  # If folder, call  the method for every image in it (only goes one level,
            # doesn't recursively search for images)
            print("Processing folder " + args.image)
            file_list = [args.image + os.path.sep + file for file in os.listdir(args.image) if
                         os.path.isfile(join(args.image, file))]
            for image in file_list:
                process_image(image, pedestrian_detection_data, args.resize)
    except FileNotFoundError:
        print(args.image + " is not found. Ignoring...")

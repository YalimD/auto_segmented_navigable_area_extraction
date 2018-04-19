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
    height, width, colorChannelWidth = imageOI.shape

    # Spatial features doesn't vary on the image_channels as they are not related to color spaces
    yy, xx = np.meshgrid(np.arange(0, width), np.arange(0, height))
    if debugging:
        print("{} and {}".format(yy, xx))

    spatialFeatures = np.vstack([xx.ravel(), yy.ravel()]).T  # Column ordered elements

    colorFeatures = np.vstack(list(map(lambda i: imageOI[:, :, i].ravel(), range(colorChannelWidth)))).T

    # Create the data as 5 dimensional np array that contains color and spatial features
    features = np.zeros((width * height, colorChannelWidth + 2))
    features[:, 0:2] = spatialFeatures
    features[:, 2:] = colorFeatures

    features = normalize(features)

    return features


# Test all algorithms, draw results color images. Labels are considered as pixel values
def generate_segments(data, img, img_name, colorSpace, resizing):
    algorithms = OrderedDict((('Mean', cluster.MeanShift), ('DB', cluster.DBSCAN), ('HDB', hdbscan.HDBSCAN)))
    kwds = [0] * len(algorithms.keys())

    flatImage = np.reshape(img, [-1, 3])  # For meanshift
    # Bin seeding speeds up the code as the initial locations of the kernels are in the discretized versions of points
    # We estimate the required bandwidth using sklearn's method
    # TODO: PARAMETERS ARE OPEN TO ADJUSTMENTS
    kwds[list(algorithms.keys()).index('Mean')] = {'bandwidth': cluster.estimate_bandwidth(X=flatImage,
                                                                                           n_samples=int(
                                                                                               flatImage.shape[0] *
                                                                                               flatImage.shape[
                                                                                                   1] / 100),
                                                                                           quantile=0.2,
                                                                                           n_jobs=-1),
                                                   'n_jobs': -1,
                                                   'bin_seeding': True}
    kwds[list(algorithms.keys()).index('DB')] = {'eps': 0.015, 'min_samples': 6, 'n_jobs': -1}
    kwds[list(algorithms.keys()).index('HDB')] = {'min_cluster_size': 50}

    cluster_results = [0] * len(algorithms.keys())
    file_names = [0] * len(algorithms.keys())

    for i, v in enumerate(algorithms):
        start_time = time.time()
        print("Starting {}".format(str(algorithms[v])))

        if v == 'Mean':
            labels = algorithms[v](**kwds[i]).fit_predict(flatImage)
        else:
            labels = algorithms[v](**kwds[i]).fit_predict(data)
        end_time = time.time()

        print("Algorithm {} took {} seconds on {} image ".format(str(algorithms[v]), str(end_time - start_time),
                                                                 colorSpace))
        # Colored segments for debugging. Some segments may have the same color value, but they are actually different
        # as seen in the grayscale results; where pixel value is directly its corresponding label
        if debugging:
            palette = sea.color_palette('deep', np.unique(labels).max() + 1)
            colors = [normalizeColor(palette[x]) if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

            result = np.reshape(colors, img.shape).astype("uint8")
            cv2.imshow("Color Result of {} algorithm on {} image".format(str(algorithms[v]), colorSpace), result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Grayscale resulting image
        cluster_results[i] = np.reshape(labels, img.shape[:2]).astype("uint8")
        cluster_results[i] = np.amax(cluster_results[i]) - cluster_results[i]

        file_names[i] = img_name[:img_name.index(".")] + "_" + str(v) + "_" + colorSpace + img_name[
                                                                                           img_name.index("."):]

    return cluster_results, file_names


# Prepare the image for clustering and extract the results
def process_image(img_name, detection_file, debugging=0, spatialIncluded=True, resizing=1):
    print("Processing image " + img_name)
    org_image = cv2.imread(img_name)
    # Image order
    image_order = {'RGB': 0, 'LAB': 1, 'HSV': 2, 'LUV': 3}
    image_channels = [0] * len(image_order.keys())

    # Resizing is done to accelerate the clustering process
    org_image = cv2.resize(org_image, None, fx=resizing, fy=resizing)

    if org_image.shape[2] == 4:  # BGRA conversion
        org_image = cv2.cvtColor(org_image, cv2.COLOR_BGRA2BGR)
    image_channels[image_order['LAB']] = cv2.cvtColor(org_image, cv2.COLOR_BGR2LAB)

    # Normal hsv: Hue range 0-180, here 0-360
    image_channels[image_order['HSV']] = cv2.cvtColor(org_image, cv2.COLOR_BGR2HSV_FULL)
    image_channels[image_order['LUV']] = cv2.cvtColor(org_image, cv2.COLOR_BGR2LUV)
    image_channels[image_order['RGB']] = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

    # Color features are created for each color space separately
    features = [0] * len(image_order.keys())
    clustering_results = [0] * len(image_order.keys())

    for i, v in enumerate(image_order):
        imageOI = image_channels[i]

        features[i] = generate_features(imageOI)
        clustering_results[i], fileNames = generate_segments(features[i], imageOI, img_name, v, resizing)

        for i, result in enumerate(clustering_results[i]):
            navigable_area = extract_navigable_areas(result, resizing, detection_file).astype("uint8")

            # Resize back to original size
            navigable_area = cv2.resize(navigable_area, None, fx=1 / resizing, fy=1 / resizing,
                                        interpolation=cv2.INTER_LINEAR)

            cv2.imshow(fileNames[i], navigable_area)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(fileNames[i], navigable_area)


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


def extract_navigable_areas(segmented_img, resizing, pedest_loc):
    if debugging:
        # Canvas image is used to draw red circles on in order to debug the result better
        canvas_img = np.copy(segmented_img)
        canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Grayscale Segments", segmented_img)
        cv2.waitKey(0)

    frame_index = 0
    fps = 30  # Default
    navigable_labels = []

    # Read the segmented image and check the labels that are within, to see if they are actually distinct
    # Other than what is seems on the resulting image

    # Open the pedest_loc file and start parsing
    # For every fps * 10 * i th line
    # Seperate by delimeter ","  and read every nth value
    with open(pedest_loc) as pedestrians:
        for i, line in enumerate(pedestrians):
            if i == 0:  # Information line
                elems = line.split()
                fps = int(elems[-1])
            elif i - 1 == frame_index * fps:
                frame_index += 1
                # Process the current line (except the frame number)
                locations = line.split(',')[1:]

                # Every 7 values is an agent (including the agent id, which is irrelevant for us)
                agents = [locations[i * 7:(i + 1) * 7] for i in range(len(locations) // 7)]

                # Temporary set to hold detected labels
                temp_nav = set([])
                for agent in agents:
                    # Take the feet position as posY + height / 2
                    feetPos = (float(agent[2]) + float(agent[6]) / 2) * resizing

                    # Put a dot at the feetpos for debugging the detections' location
                    if debugging:
                        cv2.circle(canvas_img, (int(int(agent[1]) * resizing), int(feetPos)), 2,
                                   (0, 0, 255))

                    # Find the label value there, in order to add it to the navigable list
                    pixelValue = segmented_img[int(feetPos)][int(int(agent[1]) * resizing)]
                    # Ignore noisy areas
                    if not pixelValue == 0:
                        temp_nav.add(pixelValue)
                navigable_labels += list(temp_nav)

    navigable_labels_freq = np.bincount(navigable_labels)
    navigable_labels = list(zip(list(set(navigable_labels)), navigable_labels_freq[navigable_labels]))

    try:
        # Detection per frame
        navigable_func = lambda x: x[1] / frame_index > 0.1
        navigable_labels = list(zip(*list(filter(navigable_func, navigable_labels))))[0]
    except IndexError:
        navigable_labels = []

    print("Number of frames processed {}".format(frame_index))
    print("Navigable labels: {}".format(navigable_labels))

    # In the end, create our mask and multiply it with segmentation result
    mask = np.zeros(segmented_img.shape, dtype="uint8")
    for label in navigable_labels:
        rows, cols = np.where(segmented_img == label)
        mask[rows, cols] = 255

    # Apply dilation and erosion TODO: This step should be done after homography has been calculated
    if False:
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, mask)
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel, mask)

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

    debugging = 1
    spatialIncluded = True
    resizing = 0.25

    # Param list:
    # .py pedestrian_detection_data <img-folder-list>
    if len(sys.argv) < 3:
        print("usage ped_detect_data <img-folder-list>")
        sys.exit(2)

    pedestrian_detection_data = sys.argv[1]

    for element in sys.argv[2:]:
        try:
            # If image, directly call the method
            if os.path.isfile(element):
                process_image(element, pedestrian_detection_data, debugging, spatialIncluded, resizing)
            else:  # If folder, call the method for every image in it (only goes one level,
                # doesn't recursively search for images)
                print("Processing folder " + element)
                file_list = [element + os.path.sep + file for file in os.listdir(element) if
                             os.path.isfile(join(element, file))]
                for image in file_list:
                    process_image(image, pedestrian_detection_data, debugging, spatialIncluded, resizing)
        except FileNotFoundError:
            print(element + " is not found. Ignoring...")
            continue

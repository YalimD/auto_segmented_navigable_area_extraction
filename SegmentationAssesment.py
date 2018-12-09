# This method takes a segmented image and a ground truth image to determine the
# ratio of false positives in all navigable areas. (False positive + true positive)
import numpy as np
import cv2

# The size of the segmented image should be identical to ground truth
def assessSegmentation(segmented_img, ground_truth):

    # determine number of positives in segmented
    num_of_navigable_segmented = cv2.countNonZero(segmented_img)
    num_of_navigable_gt = cv2.countNonZero(ground_truth)

    # and the images and calculate number of true elements
    img_and = cv2.bitwise_and(segmented_img, ground_truth)
    tp = cv2.countNonZero(img_and)

    fp = num_of_navigable_segmented - tp
    fn = num_of_navigable_gt - tp

    # There is your true and false positive values
    print("Recall:{}, Precision:{}. For us, precision is more important".format(tp / (tp + fn), tp / (tp + fp)))

if __name__ == "__main__":

    img = cv2.imread('C:\\Users\Rakosi\Google Drive\Master\ThesisWork\Thesis\Thesis_Final\Overleaf\clustering\80_2\stab_80_2_shadowClear_Mean_LAB.jpg', 0)
    gt =  cv2.imread('C:\\Users\Rakosi\Google Drive\Master\ThesisWork\Thesis\Thesis_Final\Overleaf\clustering\80_2\stab_80_2_shadowClear_gt.png', 0)

    assessSegmentation(img, gt)
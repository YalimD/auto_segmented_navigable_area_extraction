# This method takes a segmented image and a ground truth image to determine the dice index

import cv2

# Note: The size of the segmented image should be identical to ground truth
def assessSegmentation(segmented_img, ground_truth):

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

if __name__ == "__main__":

    img = cv2.imread('.\\test_images\\tf_test\\pet\\mask_pet.png', 0)
    gt =  cv2.imread('.\\test_images\\tf_test\\pet\\pet_gt.png', 0)

    assessSegmentation(img, gt)
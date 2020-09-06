# navigable-area-extraction
Given input video and MOT compatible pedestrian tracking data, determines the navigable regions in the video
using deep learning based segmentation techniques.

Uses Xception or Mobilenet networks (Tensorflow 2.0) trained either on Cityscapes or ADE20K datasets.

Can calculate dice score if ground truth navigable regions' binary image is provided.

MOT: https://motchallenge.net/instructions/

_**Networks**_

Xception and Mobilenet trained by Deeplab: https://github.com/tensorflow/models/tree/master/research/deeplab

**_Datasets_**

Cityscapes: https://www.cityscapes-dataset.com/

ADE20K: http://groups.csail.mit.edu/vision/datasets/ADE20K/

Requires: (Tested on Python 3.7 with Anaconda)
* OpenCV 4.2
* Numpy 
* Skimage 0.16.2
* Tensorflow 2.0.0

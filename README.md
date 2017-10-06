Pose estimation, embrace the [OpenMVG Samples](https://github.com/openMVG/openMVG/tree/develop/src/openMVG_Samples).

I tried to do some relative pose estimation tests in [MiaoDX/pose_estimation](https://github.com/MiaoDX/pose_estimation). And in that experiments, I learn the necessary build blocks for pose estimation, such as feature detection, descriptors extractors, match with ratio, and various implements, mostly in OpenCV.

The OpenMVG provide yet another stack of implements, and the most interesting part is the RANSAC part in [robust_estimation](http://openmvg.readthedocs.io/en/latest/openMVG/robust_estimation/robust_estimation/), and the [SfM module](http://openmvg.readthedocs.io/en/latest/software/SfM/SfM/) is pretty neat.

So, in this repository, I slightly changed the samples and add some command and store to files helpers.

The PnP and ICP modules use other libraries (OpenCV and libicp).
# FeaturePointDetectorsComparison

Aim of project was to compare two corner detectors Shi-Tomasi and Harris detector.

Project consist of 5 files. Files harris_detector_2d.py and shi_tomasi_detector_2d.py are files which compare corners with ground-truth created with high reliable FAST detector.
Files harris_detector_3d.py and shi_tomasi_detector_3d.py are files which compare depth of corners with known ground-truth.

Functional description
Files harris_detector_2d.py and harris_detector_3d.py are executed with blockSize and kSzie arguments. Where blockSize is size of window of detector and kSize is Sobel derivative.
Files shi_tomasi_detector_2d.py and shi_tomasi_detector_3d.py are executed with qualityLevel and euclideanDistance arguments.

The result of scripts are bar plots with measures of F described in one of arcticles in bibliography and false-positive and false-negative and true-positive measures.

Bibliography  
[1] https://docs.opencv.org/3.4/d8/dd8/tutorial_good_features_to_track.html  
[2] https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html  
[3] Q. Zhu, M. D. Phung, Q. P. Ha, Crack Detection Using Enhanced Hierarchical Convolutional Neural Networks  
[4] https://github.com/cuilimeng/CrackForest-dataset  
[5] http://vision.middlebury.edu/stereo/data/scenes2006/  

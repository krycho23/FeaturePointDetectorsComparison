# FeaturePointDetectorsComparison

Aim of project was to compare two corner detectors Shi-Tomasi and Harris detector.

Project consist of 5 files. Files harris_detector_2d.py and shi_tomasi_detector_2d.py are files which compare corners with ground-truth created with high reliable FAST detector.
Files harris_detector_3d.py and shi_tomasi_detector_3d.py are files which compare depth of corners with known ground-truth.

Functional description
Files harris_detector_2d.py and harris_detector_3d.py are executed with blockSize and kSzie arguments. Where blockSize is size of window of detector and kSize is Sobel derivative.
Files shi_tomasi_detector_2d.py and shi_tomasi_detector_3d.py are executed with qualityLevel and euclideanDistance arguments.

The result of scripts are bar plots with measures of F described in one of arcticles in bibliography and false-positive and false-negative and true-positive measures.


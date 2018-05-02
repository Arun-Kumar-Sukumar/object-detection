
# Tensorflow Object Detection API
Creating accurate machine learning models capable of localizing and identifying
multiple objects in a single image remains a core challenge in computer vision.
The TensorFlow Object Detection API is an open source framework built on top of
TensorFlow that makes it easy to construct, train and deploy object detection
models.  At Google we’ve certainly found this codebase to be useful for our
computer vision needs, and we hope that you will as well.
<p align="center">
  <img src="g3doc/img/kites_detections_output.jpg" width=676 height=450>
</p>
Contributions to the codebase are welcome and we would love to hear back from
you if you find this API useful.  Finally if you use the Tensorflow Object
Detection API for a research publication, please consider citing:

```
"Speed/accuracy trade-offs for modern convolutional object detectors."
Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z,
Song Y, Guadarrama S, Murphy K, CVPR 2017
```
\[[link](https://arxiv.org/abs/1611.10012)\]\[[bibtex](
https://scholar.googleusercontent.com/scholar.bib?q=info:l291WsrB-hQJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAWUIIlnPZ_L9jxvPwcC49kDlELtaeIyU-&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1)\]


## Setup for object detection:

  * <a href='g3doc/installation.md'>Object Detection</a><br>
  * <a href='AIplatform/libraries.txt'>
      Speech recognition</a><br>

## Running single example

Go to notebook <a href='object_detection.py'>object_detection.py</a> to test some image object first, then <a href='object_detection_webcam.py'>object_detection_webcam.py</a> to test with the camera. For the full program, go to <a href='AI-platform.py'>AI-platform.py</a><br>

Running simply in the `anaconda` with `python <file.py>` or in an IDE such as Spider or Pycharm for better management of coding environment.


<b>Thanks to contributors</b>: Jonathan Huang, Vivek Rathod, Derek Chow,
Chen Sun, Menglong Zhu, Matthew Tang, Anoop Korattikara, Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, Jasper Uijlings,
Viacheslav Kovalevskyi, Kevin Murphy

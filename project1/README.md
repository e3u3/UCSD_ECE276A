# stop sign detection

## Prerequisites
In order to run these codes, make sure the following requirements have been satisfied:
opencv-python>=3.4
matplotlib>=2.2
numpy>=1.14
scikit-image>=0.14.0

## Running test 
```
python3 stop_sign_detector.py
```

## File description
```
extract.py:
Load the pixel information from `mask.pickle` using the module pickle. 
Implement the K-ary Logistic regression model on the labelled data and obtain the weight matrix for classification on the unseen images.
```
```
stop_sign_detector.py:
1.Using the trained weight matrix to classify each pixel in the unseen image.
Input:three channels BGR images
Output:binary image with red areas set as 255
2.Detect the stop sign in the binary image with the two different search strategies:
i.The area of each contour
ii.The geometry property of the stop sign(it should be a octagon)
Input:binary image
Output:the bottom-left and top-right points of the detected stop sign.
```
## Acknowledgments
Qinru Li, Yaosen Lin, Zhirui Dai, Yuhan Liu, Zhijing Liang and I hand-label the trainset together. Thanks for their efforts.

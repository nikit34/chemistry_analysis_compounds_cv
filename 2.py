#!/usr/bin/python3
import imutils
import cv2
import numpy as np


image = cv2.imread('in/1.png')
image = cv2.bitwise_not(image)
resized = imutils.resize(image, width=300, height=300)
ratio = image.shape[0] / float(resized.shape[0])

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]



cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)


cv2.imshow("Image", image)
cv2.waitKey(0)

#!/usr/bin/python3

import cv2
import numpy as np
from pathlib import Path
import imutils


source = '1.png'
destination = './'
try:
    Path('out').mkdir()
except:
    pass

def main():
    for i, imgPatg in enumerate(Path('in').glob('*.png')):
        print(i, imgPatg)
        img = cv2.imread(str(imgPatg))
        img = imutils.resize(img, width=200)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 0, 200, 3)


        blank_image = np.full(img.shape, 0xFF, np.uint8)

        lsd = cv2.ximgproc.createFastLineDetector(
            _length_threshold = 1,
            _distance_threshold = 1.414213562,
            _canny_th1 = 50.0,
            _canny_th2 = 50.0,
            _canny_aperture_size = 7,
            _do_merge = True )

        lines = lsd.detect(edges)

        drawn_img = lsd.drawSegments(blank_image, lines)

        cv2.imshow('Contours', drawn_img)
        cv2.waitKey()


main()
cv2.destroyAllWindows()

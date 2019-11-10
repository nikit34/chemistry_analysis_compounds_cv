from PIL import Image
from pylab import *
import numpy as np
import imutils
from scipy.ndimage import filters
import cv2


def compute_harris_response(im,sigma=0.4):
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0, 1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1, 0), imy)

    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / (Wtr*Wtr)


def get_harris_points(harrisim, min_dist = 20, threshold = 0.7):

    val_max = 0
    for i in range(len(harrisim)):
        for j in range(len(harrisim[i])):
            if val_max < harrisim[i][j]:
                val_max = harrisim[i][j]

    corner_threshold = val_max * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    coords = array(harrisim_t.nonzero()).T
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    index = argsort(candidate_values)
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                        (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    
    return filtered_coords


def plot_harris_points(image, filtered_coords):
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords],'*')
    axis('off')
    show()



im = array(Image.open('in/1.png').convert('L'))

harrisim = compute_harris_response(im)
filtered_coords = get_harris_points(harrisim)
plot_harris_points(im, filtered_coords)

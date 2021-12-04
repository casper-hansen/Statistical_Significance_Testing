import cv2
import numpy as np

def compute_thickness(img):
    # White to black pixels and black to white pixels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(img)

    # select from text start x,y to text end x,y
    white_pt_coords=np.argwhere(inverted)
    min_y = min(white_pt_coords[:,0])
    min_x = min(white_pt_coords[:,1])
    max_y = max(white_pt_coords[:,0])
    max_x = max(white_pt_coords[:,1])
    inverted = inverted[min_y:max_y,min_x:max_x]

    # thin image and calculate difference from the inverted image
    thinned = cv2.ximgproc.thinning(inverted)
    return (np.sum(inverted == 255) - np.sum(thinned == 255)) / np.sum(thinned == 255)

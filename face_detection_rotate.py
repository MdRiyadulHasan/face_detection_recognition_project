import cv2
import numpy as np
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]    # height and width of the image
    (cX, cY) = (w // 2, h // 2) # center of the height and width
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)  # rotation matrix
    cos = np.abs(M[0, 0])                 
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))   # finding new width and height after rotation
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX          # adjustment, so that no information will loss
    M[1, 2] += (nH / 2) - cY 
    return cv2.warpAffine(image, M, (nW, nH))
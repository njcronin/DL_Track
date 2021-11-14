"""Python functions to calibrate the US images"""

import math
import numpy as np
import cv2


mlocs = []

## Manual scaling

def mclick(event, x_val, y_val, flags, param):
    """Detect mouse clicks for purpose of image calibration.

    Arguments:

    Returns:
        List of y coordinates of clicked points.
    """
    global mlocs

    # if the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        mlocs.append(y_val)
        mlocs.append(x_val)


def calibrateDistanceManually(nonflipped_img, spacing):
    """Calculates scalingline length of image based on manual specified
        distance between two points on image and image depth.

    Arguments:
        Original(nonflipped) image,
        distance between scaling points (mm).

    Returns:
        Length of scaling line (pixel).

    Example:
        >>>calibrate_distance_manually(Image, 5)
        5 mm corresponds to 99 pixels
    """
    img2 = np.uint8(nonflipped_img)

    # display the image and wait for a keypress
    cv2.imshow("image", img2)
    cv2.setMouseCallback("image", mclick)
    key = cv2.waitKey(0)

    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        cv2.destroyAllWindows()

    global mlocs

    calib_dist = np.abs(math.sqrt((mlocs[3] - mlocs[1])**2 + (mlocs[2] - mlocs[0])**2))
    mlocs = []
    # calculate calib_dist for 10mm
    if spacing == 5:
        calib_dist = calib_dist * 2
    if spacing == 15:
        calib_dist = calib_dist * (2/3)
    if spacing == 20:
        calib_dist = calib_dist / 2

    # print(str(spacing) + ' mm corresponds to ' + str(calib_dist) + ' pixels')
    scale_statement = '10 mm corresponds to ' + str(calib_dist) + ' pixels'

    return calib_dist, scale_statement


## Scaling bars

def calibrateDistanceStatic(nonflipped_img, spacing: str):
    """Calculates scalingline length of image based computed
        distance between two points on image and image depth.

    Arguments:
        Original(nonflipped) image with scaling lines on right border,
        distance between scaling points (mm).

    Returns:
        Length of scaling line (pixel).

    Example:
        >>>calibrate_distance_manually(Image, 5, 4.5, 0)
        5 mm corresponds to 95 pixels
    """
    # calibrate according to scale at the right border of image
    img2 = np.uint8(nonflipped_img)
    height = img2.shape[0]
    width = img2.shape[1]
    imgscale = img2[int(height*0.4):(height), (width-int(width*0.15)):width]

    # search for rows with white pixels, calculate median of distance
    calib_dist = np.max(np.diff(np.argwhere(imgscale.max(axis=1) > 150),
                                axis=0))

    if int(calib_dist) < 1:
        return None, None, None

    #scalingline_length = depth * calib_dist
    scale_statement = f'{spacing} mm corresponds to {calib_dist} pixels'

    return calib_dist, imgscale, scale_statement
    
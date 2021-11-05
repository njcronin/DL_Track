"""Python module to evaluate longitudinal US images"""

from __future__ import division 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')

import re

import glob
import cv2
import tensorflow as tf

from IPython.core.debugger import set_trace
from matplotlib.backends.backend_pdf import PdfPages
from skimage.transform import resize
from skimage.morphology import skeletonize
from scipy.signal import resample, savgol_filter, butter, filtfilt
from PIL import Image, ImageDraw
from keras import backend as K
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from pathlib import Path


def importAndReshapeImage(pathToImage, flip):
    """Define the image to analyse, import and reshape the image.

    Arguments:
        Path to image that should be analyzed, 
        Flag whether image should be flipped.

    Returns:
        Filename, image, copy of image, image not flipped,
        image height, image width

    Example:
        >>>import_image(C:/Desktop/Test/Img1.tif, 0)
        (Img1.tif, array[[[[...]]]],
        <PIL.Image.Image image mode=L size=1152x864 at 0x1FF843A2550>,
        <PIL.Image.Image image mode=L size=1152x864 at 0x1FF843A2550>,
        864, 1152)
    """

    # Define the image to analyse here and load it
    image_add = pathToImage
    filename = os.path.splitext(os.path.basename(image_add))[0]
    img = load_img(image_add, color_mode='grayscale')
    print("Loaded image at " + pathToImage)
    nonflippedImg = img
    if flip == 1:
        img = np.fliplr(img)
    img_copy = img
    img = img_to_array(img)
    h = img.shape[0]
    w = img.shape[1]
    img = np.reshape(img,[-1, h, w,1])
    img = resize(img, (1, 512, 512, 1), mode = 'constant', preserve_range = True)
    img = img/255.0
    img2 = img
    return img, img_copy, nonflippedImg, h, w, filename


def compileSaveResults(rootpath: str, dataframe: pd.DataFrame):
    """Saves analysis results to excel and pdf files.

    Arguments:
        Path to root directory of files,
        filename (str),
        dataframe (pd.DataFrame) containing filename, muscle
        and predicted area

    Returns:
        Excel file containing filename, muscle and predicted area.

    Example:
    >>>compile_save_results(C:/Desktop/Test, C:/Desktop/Test/Img1.tif,
                            dataframe)
    """
    excelpath = rootpath + '/Results.xlsx'
    if os.path.exists(excelpath):
        with pd.ExcelWriter(excelpath, mode='a') as writer:
            data = dataframe
            data.to_excel(writer, sheet_name="Results")
    else:
        with pd.ExcelWriter(excelpath, mode='w') as writer:
            data = dataframe
            data.to_excel(writer, sheet_name="Results")


def getFlipFlagsList(pathname):
    """Gets flags whether to flip an image or not. These are 0 or 1.

    Arguments: 
        Path to Flipflahg file

    Returns: 
        List containing flip flags 
    """
    flipFlags = []
    file = open(pathname, 'r')
    for line in file:
        for digit in line:
            if digit.isdigit():
                flipFlags.append(digit)
    return flipFlags


def calculateBatch(rootpath: str, filetype: str, modelpath: str, flipFilePath: str,
                    spacing: int, muscle: str, scaling: str, dic: dict, gui):
    """Calculates area predictions for batches of (EFOV) US images
        not containing a continous scaling line.

        Arguments:
            Path to root directory of images,
            type of image files,
            path to txt file containing flipping information for images,
            path to model used for predictions,
            distance between (vertical) scaling lines (mm),
            analyzed muscle,
            scaling type, 
            dictionary containing settings for calculations.
    """
    list_of_files = glob.glob(rootpath + filetype, recursive=True)
    flipFlags = getFlipFlagsList(flipFilePath)
    dataframe = pd.DataFrame(columns=["File", "Fasicle Length", "Pennation Angle", "Midthick",
                                      "x_low1", "x_high1"])
    failed_files = []

    with PdfPages(rootpath + '/ResultImages.pdf') as pdf:

        if(len(listOfFiles) == len(flipFlags)):

            try:
            #start_time = time.time()
            
                for imagepath in list_of_files:
                    if gui.should_stop:
                        # there was an input to stop the calculations
                        break

                    # load image
                    imported = import_image(imagepath, int(flip))
                    img, nonflipped_img, height, width, filename = imported
                    # get flipflag
                    flip = flipFlags.pop(0)

                    if scaling == "Bar":
                        calibrate_fn = calibrateDistanceDtatic
                        # find length of the scaling line
                        calib_dist, imgscale, scale_statement = calibrate_fn(nonflipped_img, spacing)
                        # check for StaticScalingError
                        if calib_dist is None:
                            fail = f"Scalingbars not found in {imagepath}"
                            failed_files.append(fail)
                            warnings.warn("Image fails with StaticScalingError")
                            continue

                        # predict apos and fasicles
                        fasc_l, pennation, x_low1, x_high1, midthick, fig = doCalculations(img, img_copy, h, w, 
                                                                                           calibDist, spacing, dic)

                        if fascl_l is None: 
                            fail = f"No two aponeuroses found in {imagepath}"
                            failed_files.append(fail)
                            warnings.warn("Image fails with NoTwoAponeurosesError")
                            continue

                    else:
                        calibrate_fn = calibrate_distance_manually
                        calib_dist = calibrate_fn(nonflipped_img, spacing)

                        # predict Apos and fasicles
                        fasc_l, pennation, x_low1, x_high1, midthick, fig = doCalculations(img, img_copy, h, w, 
                                                                                           calibDist, spacing, dic)

                        if fascl_l is None: 
                            fail = f"No two aponeuroses found in {imagepath}"
                            failed_files.append(fail)
                            warnings.warn("Image fails with NoTwoAponeurosesError")
                            continue

                    # append results to dataframe
                    dataframe = dataframe.append({"File": filename,
                                                  "Fasicle Length": fascl_l,
                                                  "Pennation Angle": pennation,
                                                  "Midthick": midthick,
                                                  "x_low1": x_low1,
                                                  "x_high1": x_high1},
                                                  ignore_index=True)

                    # save figures
                    pdf.savefig(fig)
                    plt.close(fig)
                    # time duration of analysis of single image
                    #duration = time.time() - start_time
                    #print(f"duration: {duration}")

            finally:
                # save predicted area results
                compile_save_results(rootpath, dataframe)
                # write failed images in file
                if len(failed_files) >= 1:
                    file = open(rootpath + "/failed_images.txt", "w")
                    for fail in failed_files:
                        file.write(fail + "\n")
                    file.close()
                # clean up
                gui.should_stop = False
                gui.is_running = False

        else:
            print("Warning: number of flipFlags (" + str(len(flipFlags)) +") doesn\'t match number of images (" + str(len(listOfFiles)) + ")! Calculations aborted.") 



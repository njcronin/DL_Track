"""Python module to evaluate longitudinal US images"""

from __future__ import division

import warnings
import time
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from skimage.transform import resize
from keras.preprocessing.image import img_to_array, load_img
from do_calculations import doCalculations
from calibrate import calibrateDistanceStatic, calibrateDistanceManually


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
    nonflippedImg = img
    if flip == 1:
        img = np.fliplr(img)
    img_copy = img
    img = img_to_array(img)
    height = img.shape[0]
    width = img.shape[1]
    img = np.reshape(img,[-1, height, width,1])
    img = resize(img, (1, 512, 512, 1), mode = 'constant', preserve_range = True)
    img = img/255.0
    return img, img_copy, nonflippedImg, height, width, filename


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


def calculateBatch(rootpath: str, apo_modelpath: str, fasc_modelpath: str, flipFilePath: str, filetype: str,
                    scaling: str, spacing: int, apo_treshold: float, fasc_threshold: float, fasc_cont_thresh: int,
                   min_width: int, curvature: int, min_pennation: int, max_pennation: int, gui):
    """Calculates area predictions for batches of (EFOV) US images
        not containing a continous scaling line.

        Arguments:
            Path to root directory of images,
            type of image files,
            path to txt file containing flipping information for images,
            path to model used for apo predictions,
            path to model used for fasc predictions,
            distance between (vertical) scaling lines (mm),
            scaling type,
            Threshold for aponeurosis detection,
            Threshold for fasicle detection,
            Threshould for fascile contours,
            Minimal allowed width between aponeuroses (mm),
            Determined fascicle curvature,
            Minimal allowed pennation angle (°),
            Maximal allowed penntaoin angle (°).
        Returns:
            Pdf containing images with predictions,
            Excel sheet containing values.

    """
    listOfFiles = glob.glob(rootpath + filetype, recursive=True)
    flipFlags = getFlipFlagsList(flipFilePath)
    dataframe = pd.DataFrame(columns=["File", "Fasicle Length", "Pennation Angle", "Midthick"])
    dic = {"apo_treshold": apo_treshold,
           "fasc_threshold": fasc_threshold,
           "fasc_cont_thresh": fasc_cont_thresh,
           "min_width": min_width,
           "curvature": curvature,
           "min_pennation": min_pennation,
           "max_pennation": max_pennation
           }
    failed_files = []

    with PdfPages(rootpath + '/ResultImages.pdf') as pdf:

        if len(listOfFiles) == len(flipFlags):

            try:
            #start_time = time.time()

                for imagepath in listOfFiles:
                    if gui.should_stop:
                        # there was an input to stop the calculations
                        break

					# get flipflag
                    flip = flipFlags.pop(0)
                    # load image
                    imported = importAndReshapeImage(imagepath, int(flip))
                    img, img_copy, nonflipped_img, height, width, filename = imported

                    if scaling == "Bar":
                        calibrate_fn = calibrateDistanceStatic
                        # find length of the scaling line
                        calibDist, imgscale, scale_statement = calibrate_fn(nonflipped_img, spacing)
                        # check for StaticScalingError
                        if calibDist is None:
                            fail = f"Scalingbars not found in {imagepath}"
                            failed_files.append(fail)
                            warnings.warn("Image fails with StaticScalingError")
                            continue

                        # predict apos and fasicles
                        fasc_l, pennation, x_low1, x_high1, midthick, fig = doCalculations(img, img_copy, height, width, calibDist,
                                                                                           spacing, apo_modelpath, fasc_modelpath, dic)

                        if fasc_l is None:
                            fail = f"No two aponeuroses found in {imagepath}"
                            failed_files.append(fail)
                            warnings.warn("Image fails with NoTwoAponeurosesError")
                            continue

                    else:
                        calibrate_fn = calibrateDistanceManually
                        calibDist = calibrate_fn(nonflipped_img, spacing)

                        # predict Apos and fasicles
                        fasc_l, pennation, x_low1, x_high1, midthick, fig = doCalculations(img, img_copy, height, width, calibDist,
                                                                                           spacing, apo_modelpath, fasc_modelpath, dic)

                        if fasc_l is None:
                            fail = f"No two aponeuroses found in {imagepath}"
                            failed_files.append(fail)
                            warnings.warn("Image fails with NoTwoAponeurosesError")
                            continue

                    # append results to dataframe
                    dataframe = dataframe.append({"File": filename,
                                                  "Fasicle Length": np.median(fasc_l),
                                                  "Pennation Angle": np.median(pennation),
                                                  "Midthick": midthick},
                                                  ignore_index=True)

                    # save figures
                    pdf.savefig(fig)
                    plt.close(fig)
                    # time duration of analysis of single image
                    #duration = time.time() - start_time
                    #print(f"duration: {duration}")

            finally:
                # save predicted area results
                compileSaveResults(rootpath, dataframe)
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

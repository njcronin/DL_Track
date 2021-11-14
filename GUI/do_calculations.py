"""Python function to calculate aponeurosis and fascicles"""

from skimage.morphology import skeletonize
from skimage.transform import resize
from scipy.signal import savgol_filter
from keras import backend as K
from keras.models import load_model

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
plt.style.use("ggplot")


def IoU(y_true, y_pred, smooth=1):
    """Computes intersection over union (IoU), a measure of labelling accuracy.

    Arguments:
        The ground-truth bit-mask,
        The predicted bit-mask,
        A smoothing parameter,

    Returns:
        Intersection over union scores.
    """
    intersect = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersect
    iou = (intersect + smooth) / (union + smooth)
    return iou

def sort_contours(cnts):
    """Function to sort contours from proximal to distal (the bounding boxes are not used)

    Arguments:
        List of contours.
    """
    # initialize the reverse flag and sort index
    i = 1
    # construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=False))

    return (cnts, boundingBoxes)

def contour_edge(edge, contour):
    """Find only the coordinates representing one edge of a contour. edge: T (top) or B (bottom)

    Arguments:
        Variable whether top ot bottom edge,
        List of contours.
    """
    pts = list(contour)
    ptsT = sorted(pts, key=lambda k: [k[0][0], k[0][1]])
    allx = []
    ally = []
    for a in range(0,len(ptsT)):
        allx.append(ptsT[a][0,0])
        ally.append(ptsT[a][0,1])
    un = np.unique(allx)
    #sumA = 0
    leng = len(un)-1
    x = []
    y = []
    for each in range(5,leng-5): # Ignore 1st and last 5 points to avoid any curves
        indices = [i for i, x in enumerate(allx) if x == un[each]]
        if edge == 'T':
            loc = indices[0]
        else:
            loc = indices[-1]
        x.append(ptsT[loc][0,0])
        y.append(ptsT[loc][0,1])

    return np.array(x),np.array(y)

def intersection(L1, L2):
    """Function to calculate intersection between two lists.
    """
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def distFunc(x1, y1, x2, y2):
    """Function to compute the distance between 2 x,y points.

    Arguments:
        Four Points (floats)
    """
    xdist = (x2 - x1)**2
    ydist = (y2 - y1)**2

    return np.sqrt(xdist + ydist)

def doCalculations(img, img_copy, h: str, w: str, calibDist: int, spacing: int,
                   apo_modelpath: str, fasc_modelpath: str, scale_statement: str, dictionary: dict):
    """Function to compute aponeuroses and fasicles.

    Arguments:
        Copy of input  for plotting,
        Height of input image,
        width of input image,
        Detected difference between scaling lines (Pixel),
        Actual distance between scaling lines (mm),
        Image with predicted Aponeuroses,
        Image with predicted Fasicles,
        Scaling results
        Dictionary with analysis settings.
    """
    # Get settings
    dic = dictionary

    # Get variables from dictionary
    fasc_cont_thresh = int(dic["fasc_cont_thresh"])
    min_width = int(dic["min_width"])
    max_pennation = int(dic["max_pennation"])
    min_pennation = int(dic["min_pennation"])
    apo_threshold = float(dic["apo_treshold"])
    fasc_threshold = float(dic["fasc_threshold"])

    #load apo model
    model_apo = load_model(apo_modelpath, custom_objects={'IoU': IoU})
    pred_apo = model_apo.predict(img)
    pred_apo_t = (pred_apo > apo_threshold).astype(np.uint8) # SET APO THRESHOLD
    pred_apo = resize(pred_apo, (1, h, w,1))
    pred_apo = np.reshape(pred_apo, (h, w))
    pred_apo_t = resize(pred_apo_t, (1, h, w,1))
    pred_apo_t = np.reshape(pred_apo_t, (h, w))

    # load the fascicle model
    modelF = load_model(fasc_modelpath, custom_objects={'IoU': IoU})
    pred_fasc = modelF.predict(img)
    pred_fasc_t = (pred_fasc > fasc_threshold).astype(np.uint8) # SET FASC THRESHOLD
    pred_fasc = resize(pred_fasc, (1, h, w,1))
    pred_fasc = np.reshape(pred_fasc, (h, w))
    pred_fasc_t = resize(pred_fasc_t, (1, h, w,1))
    pred_fasc_t = np.reshape(pred_fasc_t, (h, w))

    xs = []
    ys = []
    fas_ext = []
    fasc_l = []
    pennation = []
    x_low1 = []
    x_high1 = []

    # Compute contours to identify the aponeuroses
    _, thresh = cv2.threshold(pred_apo_t, 0, 255, cv2.THRESH_BINARY) 
    thresh = thresh.astype('uint8')
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_re = []
    for contour in contours: # Remove any contours that are very small
        if len(contour) > 600:
            contours_re.append(contour)
    contours = contours_re
    contours,_ = sort_contours(contours) # Sort contours from top to bottom

    # mask_apo = np.zeros(thresh.shape,np.uint8)
    contours_re2 = []
    for contour in contours:
    #     cv2.drawContours(mask_apo,[contour],0,255,-1)
        pts = list(contour)
        ptsT = sorted(pts, key=lambda k: [k[0][0], k[0][1]]) # Sort each contour based on x values
        allx = []
        ally = []
        for a in range(0,len(ptsT)):
            allx.append(ptsT[a][0,0])
            ally.append(ptsT[a][0,1])
        app = np.array(list(zip(allx,ally)))
        contours_re2.append(app)

    # Merge nearby contours
    # countU = 0
    xs1 = []
    xs2 = []
    ys1 = []
    ys2 = []
    maskT = np.zeros(thresh.shape,np.uint8)
    for cnt in contours_re2:
        ys1.append(cnt[0][1])
        ys2.append(cnt[-1][1])
        xs1.append(cnt[0][0])
        xs2.append(cnt[-1][0])
        cv2.drawContours(maskT,[cnt],0,255,-1)

    for countU in range(0,len(contours_re2)-1):
        if xs1[countU+1] > xs2[countU]: # Check if x of contour2 is higher than x of contour 1
            y1 = ys2[countU]
            y2 = ys1[countU+1]
            if y1-10 <= y2 <= y1+10:
                m = np.vstack((contours_re2[countU], contours_re2[countU+1]))
                cv2.drawContours(maskT,[m],0,255,-1)
        countU += 1

    maskT[maskT > 0] = 1
    skeleton = skeletonize(maskT).astype(np.uint8)
    kernel = np.ones((3,7), np.uint8)
    dilate = cv2.dilate(skeleton, kernel, iterations=15)
    erode = cv2.erode(dilate, kernel, iterations=10)

    contoursE, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_apoE = np.zeros(thresh.shape,np.uint8)

    contoursE = [i for i in contoursE if len(i) > 600] # Remove any contours that are very small

    for contour in contoursE:
        cv2.drawContours(mask_apoE,[contour],0,255,-1)
    contoursE,_ = sort_contours(contoursE)

    # Only continues beyond this point if 2 aponeuroses can be detected
    if len(contoursE) >= 2:
        # Get the x,y coordinates of the upper/lower edge of the 2 aponeuroses
        upp_x,upp_y = contour_edge('B', contoursE[0])
        if contoursE[1][0,0,1] > contoursE[0][0,0,1] + min_width:
            low_x,low_y = contour_edge('T', contoursE[1])
        else:
            low_x,low_y = contour_edge('T', contoursE[2])

        upp_y_new = savgol_filter(upp_y, 81, 2) # window size 51, polynomial order 3
        low_y_new = savgol_filter(low_y, 81, 2)

        # Make a binary mask to only include fascicles within the region between the 2 aponeuroses
        ex_mask = np.zeros(thresh.shape,np.uint8)
        ex_1 = 0
        ex_2 = np.minimum(len(low_x), len(upp_x))

        for ii in range(ex_1, ex_2):
            ymin = int(np.floor(upp_y_new[ii]))
            ymax = int(np.ceil(low_y_new[ii]))

            ex_mask[:ymin, ii] = 0
            ex_mask[ymax:, ii] = 0
            ex_mask[ymin:ymax, ii] = 255

        # Calculate slope of central portion of each aponeurosis & use this to compute muscle thickness
        Alist = list(set(upp_x).intersection(low_x))
        Alist = sorted(Alist)
        Alen = len(list(set(upp_x).intersection(low_x))) # How many values overlap between x-axes
        A1 = int(Alist[0] + (.33 * Alen))
        A2 = int(Alist[0] + (.66 * Alen))
        mid = int((A2-A1) / 2 + A1)
        mindist = 10000
        upp_ind = np.where(upp_x==mid)

        if upp_ind == len(upp_x):
                upp_ind -= 1

        for val in range(A1, A2):
            if val >= len(low_x):
                continue
            else:
                dist = distFunc(upp_x[upp_ind], upp_y_new[upp_ind], low_x[val], low_y_new[val])
                if dist < mindist:
                    mindist = dist

        # Add aponeuroses to a mask for display
        # imgT = np.zeros((h,w,1), np.uint8)

        # Compute functions to approximate the shape of the aponeuroses
        zUA = np.polyfit(upp_x, upp_y_new, 2)
        g = np.poly1d(zUA)
        zLA = np.polyfit(low_x, low_y_new, 2)
        h = np.poly1d(zLA)

        mid = (low_x[-1]-low_x[0])/2 + low_x[0] # Find middle of the aponeurosis
        x1 = np.linspace(low_x[0]-700, low_x[-1]+700, 10000) # Extrapolate polynomial fits to either side of the mid-point
        y_UA = g(x1)
        y_LA = h(x1)

        new_X_UA = np.linspace(mid-700, mid+700, 5000) # Extrapolate x,y data using f function
        new_Y_UA = g(new_X_UA)
        new_X_LA = np.linspace(mid-700, mid+700, 5000) # Extrapolate x,y data using f function
        new_Y_LA = h(new_X_LA)

        #########################################################################

        # Compute contours to identify fascicles/fascicle orientation
        _, threshF = cv2.threshold(pred_fasc_t, 0, 255, cv2.THRESH_BINARY) 
        threshF = threshF.astype('uint8')
        contoursF, hierarchy = cv2.findContours(threshF, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Remove any contours that are very small
    #     contours_re = []
        maskF = np.zeros(threshF.shape,np.uint8)
        for contour in contoursF: # Remove any contours that are very small
            if len(contour) > fasc_cont_thresh:
    #             contours_re.append(contour)
                cv2.drawContours(maskF,[contour],0,255,-1)

        # Only include fascicles within the region of the 2 aponeuroses
        mask_Fi = maskF & ex_mask
        contoursF2, hierarchy = cv2.findContours(mask_Fi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # maskF = np.zeros(threshF.shape,np.uint8)
    #     contoursF3 = []
    #     for contour in contoursF2:
    #         if len(contour) > fasc_cont_thresh:
    #     #         cv2.drawContours(maskF,[contour],0,255,-1)
    #             contoursF3.append(contour)
        contoursF3 = [i for i in contoursF2 if len(i) > fasc_cont_thresh]

        fig = plt.figure(figsize=(25,25))

        xs = []
        ys = []
        fas_ext = []
        fasc_l = []
        pennation = []
        x_low1 = []
        x_high1 = []

        for contour in contoursF2:
            x,y = contour_edge('B', contour)
            if len(x) == 0:
                continue
            z = np.polyfit(np.array(x), np.array(y), 1)
            f = np.poly1d(z)
            newX = np.linspace(-400, w+400, 5000) # Extrapolate x,y data using f function
            newY = f(newX)

            # Find intersection between each fascicle and the aponeuroses.
            diffU = newY-new_Y_UA # Find intersections
            locU = np.where(diffU == min(diffU, key=abs))[0]
            diffL = newY-new_Y_LA
            locL = np.where(diffL == min(diffL, key=abs))[0]

            coordsX = newX[int(locL):int(locU)] # Get coordinates of fascicle between the two aponeuroses
            coordsY = newY[int(locL):int(locU)]

            # Get angle of aponeurosis in region close to fascicle intersection
            if locL >= 4950:
                Apoangle = int(np.arctan((new_Y_LA[locL-50]-new_Y_LA[locL-50])/(new_X_LA[locL]-new_X_LA[locL-50]))*180/np.pi)
            else:
                Apoangle = int(np.arctan((new_Y_LA[locL]-new_Y_LA[locL+50])/(new_X_LA[locL+50]-new_X_LA[locL]))*180/np.pi) # Angle relative to horizontal
            Apoangle = 90.0 + abs(Apoangle)

            # Don't include fascicles that are completely outside of the field of view or
            # those that don't pass through central 1/3 of the image
        #     if np.sum(coordsX) > 0 and coordsX[-1] > 0 and coordsX[0] < np.maximum(upp_x[-1],low_x[-1]) and coordsX[-1] - coordsX[0] < w and Apoangle != float('nan'):
            if np.sum(coordsX) > 0 and coordsX[-1] > 0 and coordsX[0] < np.maximum(upp_x[-1],low_x[-1]) and Apoangle != float('nan'):
                FascAng = float(np.arctan((coordsX[0]-coordsX[-1])/(new_Y_LA[locL]-new_Y_UA[locU]))*180/np.pi)*-1
                ActualAng = Apoangle-FascAng

                if ActualAng <= max_pennation and ActualAng >= min_pennation: # Don't include 'fascicles' beyond a range of pennation angles
                    length1 = np.sqrt((newX[locU] - newX[locL])**2 + (y_UA[locU] - y_LA[locL])**2)
                    fasc_l.append(length1[0]) # Calculate fascicle length
                    pennation.append(Apoangle-FascAng)
                    x_low1.append(coordsX[0].astype('int32'))
                    x_high1.append(coordsX[-1].astype('int32'))
                    coords = np.array(list(zip(coordsX.astype('int32'), coordsY.astype('int32'))))
                    plt.plot(coordsX,coordsY,':w', linewidth = 6)
        # cv2.polylines(imgT, [coords], False, (20, 15, 200), 3)

        # DISPLAY THE RESULTS
        plt.imshow(img_copy, cmap='gray')
        plt.title(str(scale_statement), fontsize=20)
        plt.plot(low_x,low_y_new, marker='p', color='w', linewidth = 15) # Plot the aponeuroses
        plt.plot(upp_x,upp_y_new, marker='p', color='w', linewidth = 15)

        xplot = 125
        yplot = 700

        # Store the results for each frame and normalise using scale factor (if calibration was done above)
        try:
            midthick = mindist[0] # Muscle thickness
        except:
            midthick = mindist

        fasc_l = fasc_l / (calibDist/int(spacing))
        midthick = midthick / (calibDist/int(spacing))

        plt.text(xplot, yplot, ('Fascicle length: ' + str('%.2f' % np.median(fasc_l)) + ' mm'), fontsize=15, color='white')
        plt.text(xplot, yplot+50, ('Pennation angle: ' + str('%.1f' % np.median(pennation)) + ' deg'), fontsize=15, color='white')
        plt.text(xplot, yplot+100, ('Thickness at centre: ' + str('%.1f' % midthick) + ' mm'), fontsize=15, color='white')
        plt.grid(False)

        # clear session so tf does not complain
        K.clear_session()

        return fasc_l, pennation, x_low1, x_high1, midthick, fig

    else:

        # clear session so tf does not complain
        K.clear_session()
        return None, None, None, None, None, None
    
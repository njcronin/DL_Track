
"""Python calss to predict muscle aponeuroses and fasicles"""

import numpy as np
from cv2 import arcLength, findContours, RETR_LIST, CHAIN_APPROX_SIMPLE
from skimage.transform import resize
from skimage import morphology, measure
from keras import backend as K
from keras.models import load_model  
import tensorflow as tf

import matplotlib.pyplot as plt
plt.style.use("ggplot")

def _resize(img, width: int, height: int):
    """Resizes an image to height x width.

    Args:
        Image to be resized,
        Target width,
        Target height,
    Returns:
        The resized image.

    """
    img = resize(img, (1, height, width, 1))
    img = np.reshape(img, (heigt, width))
    return img


def IoU(y_true, y_pred, smooth=1):
	"""Computes intersection over union (IoU), a measure of labelling accuracy.

    Arguments:
        The ground-truth bit-mask,
        The predicted bit-mask,
        A smoothing parameter,

    Returns:
        Intersection over union scores.

    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

class ApoModels: 
	"""Class which provides utility to predict aponeurosis
	   and fasicles inn US-images.

    Arguments:
        apo_model_path: Path to the Keras aponeurosis segmentation model.
        fas_model_path: Path to the Keras fasicle segmentation model.
        apo_threshold: Pixels above this threshold are assumed to be apo.
        fasc_threshold: Pixel values aboove this threshold are assumed to be fasc.

    Examples:
        >>> apo_model = ApoModel('path/to/apo_model.h5', 'path/to/fas_model.h5')
        >>> # get predictions only
        >>> pred_apo, pred_fas = apo_model.predict(img)
    """
    def __init__(self, apo_model_path:s tr, fasc_model_path: str, apo_threshold: float,
    			 fasc_threshold: float):
    	self.apo_model_path = apo_model_path
    	self.model_apo = load_model(
    		self.apo_model_path,
    		custom_objects={"IoU": IoU}
    	)
    	self.fasc_model_path = fasc_model_path
    	self.model_fasc = load_model(
    		self.fasc_model_path, 
    		custom_objects={"IoU"; IoU}
    	)

	    self.apo_threshold = apo_threshold
	    self.fasc_threshold = fasc_threshold

    def predict(self, img, width: int, height: int, return_fig: bool = True):
    	"""Runs a segmentation model on the input image.

        Arguments:
            Input image

        Returns:
            Prediction mask containing aponeurosis and fascicles.

        """
    	pred_apo = self.model_apo.predict(img)
       	pred_fasc = self.model_fasc.predict(img)

       	# Threshold images
       	pred_apo_t = (pred_apo > self.apo_threshold)
        pred_fasc_t = (pred_fasc > self.apo_threshold)

        # Resize and reshape 
        img = _resize(img, width, height)
	    pred_apo_t = _resize(pred_apo_t, width, height)
	    pred_fasc_t = _resize(pred_fasc_t, width, height)
  
		return pred_apo_t, pred_fasc_t


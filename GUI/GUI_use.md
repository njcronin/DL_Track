# DL_Track GUI

This GUI for the DL_Track tool aims to enable the analysis of image batches or folders as well as automated scaling of the input images. 
It is built completely upon DL_Track and uses pythons tkinter. Note that the color scheme is based on subjective preferences (if it's too bad, I'll change it 😄).
So far, only images can be analysed using the GUI, not videos.

## Starting the GUI

To make use of the GUI simply follow the setup instructions for DL_Track.
1. However, instead of creating the conda environment manually and using pip, create the environment using the included environment.yml file (this is necessary because tkinter is not available for pip): 

``
conda env create -f environment.yml
``

2. Activate the DL_Track environment as described: 

``
conda activate DL_Track
``

3. Run the DLTrack_GUI.py file: 

``
python DLTrack_GUI.py 
``

## Usage

All hardcoded parameters displayed in the GUI are the same as in the jupyter notebooks, except for the specification of a FlipFlag file, the image type and the spacing. 
All of these parameters are necessary for the automatic scaling of the images. 
1. FlipFlag file: 
Should be a .txt file containing Flipflags. The Flipflags determine wheter an image is flipped or not (same functionality as "flip" variable in notebook). 
Thus, a single Flipflag is binary, either a 0 (= image is not flipped for analysis) or a 1 (= image is flipped for analysis).
This is necessary because the fascicle in the ultrasound image require a specific orientation (bottom left to top right). 
The number of Flipflags must match the number of images to be analysed. Please see the example FlipFlags.txt for further information. 
2. Image type: 
The image file type should be specified to automatically locate and gather the image files for further analysis. Please make sure that the specified filetype matches your image files. 
3. Spacing: 
The spacing describes the distance (in mm) between two horzontal scaling bars located in the ultrasound image. For convenience, I assumed that these are always on the right side of the image. 
If this is not the case in your images (or if there aren't any scaling bars), please flip the images prior to analysis or scale them manually. 

## Closing remarks

I hope this GUI is of help to all the users of DL_Track. Thanks to Neil and [Philipp Wirth] for your help! 
If there are any errors or suggestions, feel free to let me know via an issue or email (paul.ritsche@unibas.ch). 

[Philipp Wirth]: https://github.com/philippmwirth



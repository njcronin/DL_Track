Follow these instructions to get everything set up for you to be able to run the tracking software. The instructions are based on Windows 10. Mac instructions may be added later if there is demand. 

**Please note**: this was developed on a modern computer with a high spec GPU: running this program on a standard CPU will result in much slower performance, and may even fail due to memory limitations.


## Setup (only needs to be done once)

1. Install Python / Anaconda: https://www.anaconda.com/distribution/ 
(click ‘Download’ and be sure to choose ‘Python 3.X Version’ (where the X represents the latest version being offered. IMPORTANT: Make sure you tick the ‘Add Anaconda to my PATH environment variable’ box)
2. Clone the Github repository for this project: https://github.com/njcronin/DL_Track  
If you are not familiar with this process, use [these instructions](https://help.github.com/en/desktop/contributing-to-projects/cloning-a-repository-from-github-to-github-desktop) to clone the repository to a location of your choice (I recommend the c: drive)
3. Open an Anaconda prompt window (in windows, click ‘start’ then use the search window), then create a virtual environment using the following as an example (here I use ‘DL_track’ as the virtual environment name, but this can be anything):  
`conda create --name DL_track python=3.6.13`  
(If prompted, type `y` to confirm that the relevant packages can be installed)
4. Activate the virtual environment by typing `activate DL_track` (where DL_track is replaced by the name you chose). 
5. cd to where you have saved the project folder, e.g. by typing `cd c:/DL_Track`
6. type the following command: 
`conda install -c anaconda cudatoolkit==9.0`
7. type the following command:
`conda install -c anaconda cudnn==7.6.5`
8. type the following command:  `pip install -r requirements.txt`  
(this step may take some time)
9. type `jupyter notebook` and Jupyter notebooks should load in your browser


## Usage

1. Open an Anaconda prompt window
2. Activate your virtual environment, as done in step 4 above
3. cd to the folder where you have the tracking software, e.g. `cd c:/DL_track`
3. Type `jupyter notebook` in the prompt window
4. Now you should see the different Jupyter notebooks that allow you to train a model or to run inference on single images or videos (each labelled accordingly)
5. Open the notebook you need. Within each notebook, use `ctrl-enter` to run a cell

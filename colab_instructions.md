NOTE: This software requires a lot of processing power, so if possible, the best way to run it is to install it on a laptop with a 
decent GPU, especially if you plan to train your own network. You should only use this Google colab option if you do not have GPU 
access, as it has some limitations (the main one being the inability to scale the images at the moment).

**Instructions for running the software in your browser using colab:**

1. Ensure that you have a Google Drive account. 

2. Clone/download the DL_track files from Github, then upload the entire folder to your Google Drive account (if you’re confident, you can do this by directly cloning, c.f. https://lalorosas.com/blog/github-colab-drive)  

3. In your Google Drive account, right-click either the *COLAB_Inference_Single_Image.ipynb* file (for analysing single images) or the *COLAB_Inference_Video.ipynb file* (for analysing videos), and select *open with* and then *Google colaboratory*

4. Run through the code cell by cell, following the instructions. Note that it’s not currently possible to scale your images using this method because opencv is not fully compatible with colab. If I find a good workaround I’ll fix this

**IMPORTANT: The images/videos you want to analyse also need to be within the main DL_track folder. For single images, place them in the *'images'* folder, videos in the *'videos'* folder.**

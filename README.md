Chromosome_counter
==================

A simple tool to count chromosomes in the metaphase spreads for mouse and human cells. Allows for manual edition of the segmentation afterwards


Purpose:
--------
This package is intended to run chromosome counting routines after imaging with DAPI staining.

Human and mice chromosome counting is currently supported, but only human chromosome counting is supported through GUI.

Initial spread files in .tif, .tiff, .jpg, jpeg formats are supported. Images up to 4 Mega pixels are supported. In case
you are using a jpeg image as a source, make sure it was compressed with highest quality. Because of image deformation 
due to compression, the pre-processing is likely to create unlikely patterns. A good way of avoiding this is to save JPEG
files with GIMP, while setting compression quality to 100% and smoothing to 1 in advanced parameters.

The application was tested on Windows 7 and should work on all the Windows NT (XP or better) workstations


User pipeline:
-------------
1. Start the GUI by launching GUI.py

2. Select the file or folder you would like to process and hit "Preprocess"

    2.1 If a file was selected, the pre-processing will only affect this file
    
    2.2 If no file was selected, the pre-processing will scan through the folder and look for all the files with
        allowed extension (.tif/.tiff/.jpg/.jpeg) and pre-process them all.
        
3. Inside the folder a "buffer" folder will be created, with subfolders named as selected images. In each subfolder, the
    original image, segmentation mask and some program-specific data will be saved
    
NOTE: Depending on the OS and workstation, it might take between 1 to 5 minutes per file. The application progress bar
        will not move unless a file is pre-processed, so if it looks like your process is hanging, most of the time it is
        not the case.
        
4. Once the pre-processing will be finished, you can manually edit the segmentation mask to better fit the reality. As 
    a rule of thumb, all spreads that are easy to count by eye will need no additional post-processing. In case any edition
    is needed, we recommend using GNU Image Manipulation Program (GIMP). It is free source and can be freely downloaded.
    
5. Once modifications are made, hit the "Process after correction".

    5.1 A "output" folder will be generated in the same location as image, containing all the segmented images with 
    counted chromosomes
    
    5.2 A window will pop allowing you to inspect the segmentation results. 
    
    5.3 If the segmentation results are unsatisfactory,
        edit the corresponding mask in the corresponding buffer folder and re-save it as "EDIT_ME2.tif" file. 

    
Support:
---------
In case of problems, please fill in the issue in the issue tracker of the master project on GitHub. We will come back to
you ASAP.


Visual:
-------
GUI:

![alt tag](http://i.imgur.com/VcdJF2D.png)

uncorrected Spread mask

![alt tag](http://i.imgur.com/gLC8bBE.png)

corrected Spread Mask

![alt tag](http://i.imgur.com/ikUJLai.png)

Clustering without correction

![alt tag](http://i.imgur.com/wE3MxcP.png)

Clustering with correction

![alt tag](http://i.imgur.com/1cE7g6Z.png)

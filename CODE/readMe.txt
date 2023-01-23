
  main.py
    
    This file contains the master control for the motherhive architecture.
    It contains an UI that will allow the user to comuniate with the SST
    machine. 
    
    It allows the user to preset a time for runining the monitoring code,
    as well as an overide option to run the device whenever the user want.
    It displays the output for the latest diagnosis and an option to clear
    the last output.
    
    REQIRED LIBRARIES:
        SST*
        tkinter
	
    *SST has library requirements of its own.
    
    Jose Antonio Kautau Toffoli 
    2022-11-18
    
  SSTV1.0

    This is the deployable version of our software for controlling the
    SST. It is composed by:
    
    1) A discretized convolutional neural network for strawberry plant
        disease recognition.

    2) Set of comands for moving the motors that will control the camera
        ring up and down the starberry tower.
        
    3) Set of comands to capture images with a set of 1 camera.
    
    This code is goes through the steps required to take n pictures from
    the strawberry tree and stores those images in the images folder, then
    it identifies and classify images found in the image folder into the
    set of possible diagnosis found in the labels_folder.
    
    To run this code you will required to have a model and the labels for 
    it in the folders and input their names in the main function when loading
    the model.

    Please not you will also require to indicate the number of photos you 
    expect to find in the images folder.

    For our purposes, we are using the following model and file.
        model_file  = "lite-model_disease-classification_1", 
        labels_file = "lite-model_disease-classification_1_labels" 

    You will also require to have the camera hardware attached to
    the raspberry pi as well as the limiting switches.
    
    REQIRED LIBRARIES:
        numpy 
        Pillow
        tflite_runtime.interpreter*
        RPi.GPIO
        picamera**

    *Contact joseantonioklautautoffoli@gmail.com for instructions on how
    to download the tflite library, he will be happy to help. 
    
    **You must use the RPi Bullseye Legacy OS 32bits in order to run the
    API connected to this library.

    IMPORTANT:
	DO NOT DELETE pineapple.jpg
    
    Jose Antonio Kautau Toffoli 
    2022-11-18

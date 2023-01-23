"""
    SSTV1.1

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
        model_file  = "SST-model", 
        labels_file = "SST-labels" 

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
    
    Jose Antonio Kautau Toffoli 
    2022-11-18
"""
### IMPORTS 
import os
import time

import RPi.GPIO as GPIO

from tflite_runtime.interpreter import Interpreter 
from PIL                        import Image
from picamera                   import PiCamera

### IMAGE AQUISITION FUNCTIONS
def init_GPIO (warn_bool = 0):
    
    # Turn off warnings.
    GPIO.setwarnings (warn_bool)
    GPIO.setmode (GPIO.BCM)
    
    # Set inputs.
    GPIO.setup (13, GPIO.IN) # pin 33 : top lim switch.
    GPIO.setup (26, GPIO.IN) # pin 37 : bot lim switch.
    
    # Set up outputs.
    GPIO.setup (5, GPIO.OUT) # pin 29 : -dir.
    GPIO.setup (6, GPIO.OUT) # pin 31 : -pul.
    
    # Start both at low.
    GPIO.output(5, False)
    GPIO.output(6, False)
    
    return 0

def move_cam_ring (steps, cw_bool):
    
    # Set direction of rotation.
    GPIO.output(5, (cw_bool))
    
    # Initiate control signal.
    global pulse_state
    
    step_count  = 0 
    time_step   = 0.010
    
    # Rotate device by steps steps.
    for i in range (steps):
        
        # Change pulse state.
        pulse_state = not(pulse_state)
        GPIO.output(6, pulse_state)
        
        # Stop if lim switch high.
        if (GPIO.input(13) and not(cw_bool)):
            return 1
        elif (GPIO.input(26) and cw_bool):
            return 1
        
        time.sleep(time_step)
        
    return 0

def calib_motor ():
    
    steps_range = 0
    
    # Move up max.
    while (not(move_cam_ring (2, 0))):
        continue
    
    print ('> max range')
    time.sleep(1)
    
    # Move down max and count steps.
    while (not(move_cam_ring (1, 1))):
        steps_range += 1
    
    print ('> min range')
    time.sleep(1)
    
    return steps_range

### IMAGE PROCESSING FUNCTIONS
def set_input_tensor (interpreter, image):
    """
    set_input_tensor

    Takes the input image and injects it into the input tensor to be interpreted
    by our model.

    @param interpreter: Interpreter to translate disease classification model.
    @type  interpreter

    @param image: Input image to be classified.
    @type  rank 3 tensor
    """
    # Get index of input tensor.
    tensor_index = interpreter.get_input_details()[0]['index']

    # Return the input tensor based on its index.
    input_tensor = interpreter.tensor(tensor_index)()[0]

    # Assigning the image to the input tensor.
    input_tensor[:, :] = image

def load_model (model_file, labels_file):
    """
    load_model

    Loads the model, and returns the main file path, and the labels file path
    for reference.

    @param model_file: file name for the model to be used.
    @type  str

    @param labels_file: file name for the labels equivalent to the output nodes.
    @type  str

    RETURN interpreter, dir_path, label_path
    """
    # Get cwd.
    dir_path = os.path.dirname (os.path.realpath(__file__))

    # Get model.
    model_path = dir_path + "/models_folder/" + model_file + ".tflite"

    # Get labels.
    label_path = dir_path + "/labels_folder/" + labels_file + ".txt"

    # Load model.
    interpreter = Interpreter (model_path)

    return interpreter, dir_path, label_path

def load_data (resize_w, resize_h, dir_path):
    """
    load_data 

    Returns all the images from the image_folder treated to RGB tensors and resized
    to models dimensions for input.

    @param resize_w: Width of the input required by the model.
    @type  int

    @param resize_h: Hight of the input required by the model.
    @type  int

    @param dir_path: Path for main file.
    @type  str

    RETURN images
    """

    images = []
    
    image_folder_path = dir_path + "/images_folder" 
    n = len([entry for entry in os.listdir(image_folder_path) if os.path.join(image_folder_path, entry)])

    for i in range (n):

       
        # Image path.
        image_path = dir_path + "/images_folder/img%i.jpg"%i

        # Convert ith image into RGB tensor.
        image = Image.open (image_path).convert('RGB')
        
        # Resize RGB tensor to fit model input.
        image = image.resize ((resize_w, resize_h))

        # Append image to list of images.
        images.append(image)
        
        # Delete image.
        os.remove(image_path)

    return images

def get_image_scores (interpreter, image):
    """
    get_image_scores

    Get image scores for image from interpreter by running it through our model.

    @param interpreter: Interpreter to translate disease classification model.
    @type  interpreter

    @param image: image to be classified.
    @type  rank 3 tensor

    RETURN scores_continuous
    """

    # Sets the image to be classfied as the input tensor.
    set_input_tensor(interpreter, image)

    # Presvents RuntimeError: reference to internal data in the interpreter in the form of a numpy array or slice.
    interpreter.invoke()

    # Get output details from the tensor.
    output_details = interpreter.get_output_details()[0]

    # Get discrete score for image.
    scores = interpreter.get_tensor(output_details['index'])[0]

    # Convert discreete score to continuous.
    scale, zero_point = output_details['quantization']
    scores_continuous = scale * (scores - zero_point)

    return scores_continuous

def run_sst ():
    
    # Number of photos per run.
    n = 40
    
    # Load model.
    try:

        interpreter, dir_path, label_path = load_model (
            model_file  = "SST-model", 
            labels_file = "SST-labels"
        )

        print ("MODEL LOADED... \n\n")    

    except:

        print("Error404: Model not found at specified directory.\n")

        return 1
    
    # Taking pictures.
    try:
        
        # Initialize pulse state for motor.
        global pulse_state
        pulse_state = 0
        
        # Number of stops for pictures.
        nStops = int(n/4)
        
        # Initiate pins.
        init_GPIO()
        
        # Calibrate motor.
        step_burst = int(calib_motor ()/nStops)
    
        # Create camera objects.
        cameraA = PiCamera()
        cameraA.resolution = (300, 300)
        
        # photo id.
        i = 0
        
        cameraA.start_preview()
        # For each layer -- 4 photos.
        for k in range(nStops):
            
            # Warm up.
            time.sleep(1)
            
            # Camera A.
            photo_output_dir = dir_path + '/images_folder/img%i.jpg'%i
            cameraA.capture(photo_output_dir)
            
            # Update photo id.
            i+=1
            
            # Other Cameras <TO BE ADDED>.
            
            # Move motor.
            if (move_cam_ring (step_burst, 0)):
                # Exit if motor activate limit swiches.
                time.sleep (1)
                break
        
        cameraA.stop_preview()
        cameraA.close()
        
        # Return home (i.e., lowest position).
        stepsDown = 0
        while (not(move_cam_ring (1, 1)) and stepsDown <= nStops*step_burst):
            # Stop if it reaches the limit switch or regressed same number of steps up.
            stepsDown += 1
            continue
        
        # Clean up pins.
        GPIO.cleanup()
        
        print ("IMAGES CAPTURED... \n\n")  
        
    except:
        
        print("MalfuntionError: Unable to obtain images.\n")
        
        return 1
    
    # Allocate input tensors to interpreter based on model requirements.
    interpreter.allocate_tensors()
    _, h, w, _ = interpreter.get_input_details()[0]['shape']

    # Load images.
    try: 
        images = load_data(
            resize_h = h,
            resize_w = w,
            dir_path = dir_path
        )
        
        print ("IMAGES LOADED... \n\n")

    except:

        print("Error404: Images not found at specified directory.\n")

        return 1

    image_scores = []

    # Classify images.
    for image in images:

        try:
            # Classify image.
            image_score = get_image_scores(interpreter, image)

            image_scores.append(image_score)
        
        except:
            
            print("UnknownError: Unable to get image scores.\n")
            
            return 1

    # Read labels.
    with open(label_path, 'r') as f:
        labels = [line.strip() for i, line in enumerate(f.readlines())]

    # Possible diagnossis for all plants and the probability threshold chosen.
    possible_diagnosis_list = []
    prob_threshold = 0.7

    # Iterate through all images.
    for i in range(len(images)):

        # Possible diagnosis for plant_i.
        possible_diagnosis = []

        # Iterate through the probability scores of image_i.
        for j in range(len(image_scores[i])):

            # Store diagnosis if probability above probability threshold.
            if (image_scores[i][j] >= prob_threshold and labels[j] != "Healthy"):

                classification_label = labels[j]
                possible_diagnosis.append((classification_label, image_scores[i][j]))
        
        # If no dignosis was obtained for plant_i assume healthy.
        if not possible_diagnosis:
            possible_diagnosis.append(("Healthy", "Null"))
        
        # Store diagnosis for plant_i.
        possible_diagnosis_list.append(possible_diagnosis)
    

    # Value to be accessed.
    print (possible_diagnosis_list)

    return possible_diagnosis_list  

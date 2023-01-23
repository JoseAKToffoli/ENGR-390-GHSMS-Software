"""
    plantDiseaseCNNV1.5

    This is the deployable version of our convolutional neural network 
    it consists of a ResNet50 model that has been discretized so that 
    it can be deployable.

    This code is able to identify and classify images found in the image 
    folder into the set of possible diagnosis found in the labels_folder.
    
    To run this code you will required to have a model and the labels for 
    it in the folders and input their names in the main function when loading
    the model.

    Please not you will also require to indicate the number of photos you 
    expect to find in the images folder.

    For our purposes, we are using the following model and file.
        model_file  = "lite-model_disease-classification_1", 
        labels_file = "lite-model_disease-classification_1_labels" 

    REQIRED LIBRARIES:
        numpy 
        Pillow
        tflite_runtime.interpreter*

    *Contact joseantonioklautautoffoli@gmail.com for instructions on how
    to download the tflite library, he will be happy to help. 

    Jose Antonio Kautau Toffoli 
    2022-11-12
"""

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

def load_data (n, resize_w, resize_h, dir_path):
    """
    load_data 

    Returns all the images from the image_folder treated to RGB tensors and resized
    to models dimensions for input.

    @param n: number of images expected to be found.
    @type  int

    @param resize_w: Width of the input required by the model.
    @type  int

    @param resize_h: Hight of the input required by the model.
    @type  int

    @param dir_path: Path for main file.
    @type  str

    RETURN images
    """

    images = []

    for i in range (n):

       
        # Image path.
        image_path = dir_path + "/images_folder/img%i.jpg"%i

        # Convert ith image into RGB tensor.
        image = Image.open (image_path).convert('RGB')

        # Resize RGB tensor to fit model input.
        image = image.resize ((resize_w, resize_h))

        # Append image to list of images.
        images.append(image)

    return images

def classify_image (interpreter, image):
    """
    classify_image

    Classifies an image by running it through our model that will be interpreted by our interpreter.

    @param interpreter: Interpreter to translate disease classification model.
    @type  interpreter

    @param image: image to be classified.
    @type  rank 3 tensor

    RETURN max_score_index, dequantized_max_score, scores_dequantized
    """

    # Sets the image to be classfied as the input tensor.
    set_input_tensor(interpreter, image)

    # Presvents RuntimeError: reference to internal data in the interpreter in the form of a numpy array or slice.
    interpreter.invoke()

    # Get details from the tensor.
    output_details = interpreter.get_output_details()[0]

    # Get discrete score for image.
    scores = interpreter.get_tensor(output_details['index'])[0]

    # Convert discreete score to continuous.
    scale, zero_point = output_details['quantization']
    scores_dequantized = scale * (scores - zero_point)
    
    # Get max probability guess.
    dequantized_max_score = np.max(np.unique(scores_dequantized))

    # Get index of max probability.
    max_score_index = np.where(scores_dequantized == np.max(np.unique(scores_dequantized)))[0][0]

    return max_score_index, dequantized_max_score, scores_dequantized

### MAIN CODE STARTS HERE 
def main ():

    # Load model.
    try:

        interpreter, dir_path, label_path = load_model (
            model_file  = "lite-model_disease-classification_1", 
            labels_file = "lite-model_disease-classification_1_labels"
        )

        print ("MODEL LOADED... \n\n")    

    except:

        print("Error404: Model not found at specified directory.\n")

        return 1
    
    # Allocate input tensors to interpreter based on model requirements.
    interpreter.allocate_tensors()
    _, h, w, _ = interpreter.get_input_details()[0]['shape']

    # Load images.
    try: 
        images = load_data(
            n        = 10,
            resize_h = h,
            resize_w = w,
            dir_path = dir_path
        )

        print ("IMAGES LOADED... \n\n")

    except:

        print("Error404: Images not found at specified directory.\n")

        return 1

    image_label_ids = []
    image_probabilities = []
    image_scores = []

    # Classify images.
    for image in images:

        try:
            # Classify image.
            image_label_id, image_probability, image_score = classify_image(interpreter, image)

            image_label_ids.append(image_label_id)
            image_probabilities.append(np.round(image_probability, 4))
            image_scores.append(image_score)
        
        except:
            print("UnknownError: Unable to classify image.\n")
            continue

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
            if image_scores[i][j] >= prob_threshold:

                classification_label = labels[j]
                possible_diagnosis.append((classification_label, image_scores[i][j]))
        
        # If no dignosis was obtained for plant_i assume healthy.
        if not possible_diagnosis:
            possible_diagnosis.append(("Healthy", "Null"))
        
        # Store diagnosis for plant_i.
        possible_diagnosis_list.append(possible_diagnosis)
    

    # Value to be accessed.
    print (possible_diagnosis_list)

    return 0

if __name__ == '__main__':

    import sys
    import os

    import numpy as np

    from tflite_runtime.interpreter import Interpreter 
    from PIL                        import Image
    
    sys.exit(main())
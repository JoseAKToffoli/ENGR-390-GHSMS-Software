

def set_input_tensor (interpreter, image):

    # Get index of input tensor.
    tensor_index = interpreter.get_input_details()[0]['index']
    #print("Index of the input tensor: ", tensor_index, end="\n\n")

    # Return the input tensor based on its index.
    input_tensor = interpreter.tensor(tensor_index)()[0]

    # Assigning the image to the input tensor.
    input_tensor[:, :] = image

def load_model (model_file, labels_file):

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

    images = []

    for i in range (n):

        image_path = dir_path + "\images_folder\img%i.jpg"%i

        # Convert ith image into RGB tensor.
        image = Image.open (image_path).convert('RGB')

        # Resize RGB tensor to fit model input.
        image = image.resize ((resize_w, resize_h))

        # Append image to list of images.
        images.append(image)

    return images

def classify_image (interpreter, image):

    # Sets the image to be classfied as the input tensor.
    set_input_tensor(interpreter, image)

    # Presvents RuntimeError: reference to internal data in the interpreter in the form of a numpy array or slice.
    interpreter.invoke()

    # Get details from the tensor.
    output_details = interpreter.get_output_details()[0]
    #print("\nDetails about the input tensors:\n   ", output_details, end="\n\n")

    # Get discrete score for image.
    scores = interpreter.get_tensor(output_details['index'])[0]
    #print("Predicted class label score      =", np.max(np.unique(scores)))

    # Convert discreete score to continuous.
    scale, zero_point = output_details['quantization']
    scores_dequantized = scale * (scores - zero_point)
    
    # Get max probability guess.
    dequantized_max_score = np.max(np.unique(scores_dequantized))
    #print("Predicted class label probability=", dequantized_max_score, end="\n\n")

    # Get index of max probability.
    max_score_index = np.where(scores_dequantized == np.max(np.unique(scores_dequantized)))[0][0]
    #print("Predicted class label ID=", max_score_index)

    return max_score_index, dequantized_max_score, scores_dequantized

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

            if image_probability <= 0.7:

                image_label_ids.append(0)
                image_probabilities.append('N/A')
                image_scores.append(image_score)

            else:
                image_label_ids.append(image_label_id)
                image_probabilities.append(np.round(image_probability, 4))
                image_scores.append(image_score)
        
        except:
            print("UnknownError: Unable to classify image.\n")
            continue

    # Read labels.
    with open(label_path, 'r') as f:
        labels = [line.strip() for i, line in enumerate(f.readlines())]

    # Display results.
    for i in range(len(images)):

        # Get most likely label.
        classification_label = labels[image_label_ids[i]]

        print(f"\nImage {i} | Image Label: {image_label_ids[i]} - {classification_label} | Accuracy: {image_probabilities[i]}\n")
        print(image_scores[i], end = '\n')

    return 0

if __name__ == '__main__':

    import sys
    import os

    import numpy as np

    from tflite_runtime.interpreter import Interpreter 
    from PIL                        import Image
    
    sys.exit(main())
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config_path = 'epsilonv0.1/config.json'
with open(config_path, 'r') as file:
    config = json.load(file)

# Load the model from the config
model = Sequential.from_config(config['config'])

weights_path = 'epsilonv0.1/model.weights.h5'
model.load_weights(weights_path)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def process_image(image_path, target_size=(28, 28)):
    """Preprocess an image for model prediction."""
    img = load_img(image_path, color_mode='grayscale', target_size=target_size)
    img_array = img_to_array(img) # / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_array, axis=0)

lst_c = []
lst_w = []
def check_misclassifications(dataset_path):
    """Check for misclassified images in the dataset."""
    for folder in os.listdir(dataset_path):
        
        if not folder.isdigit():
            continue  # Skip non-numeric folders
        
        actual_label = int(folder)
        folder_path = os.path.join(dataset_path, folder)

        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = process_image(image_path)

            # Use CPU explicitly for prediction
            with tf.device('/CPU:0'):
                predicted_label = np.argmax(model.predict(image), axis=-1)[0]

            if predicted_label != actual_label:
                lst_c.append(actual_label)
                lst_w.append(predicted_label)
                print(f"Misclassified: Actual Label = {actual_label}, Predicted Label = {predicted_label}, File = {image_file}")

# Define dataset paths
dataset_path_test = 'MNIST/test' 
dataset_path_train = 'MNIST/train' 

# Check for misclassifications
# check_misclassifications(dataset_path_train)
check_misclassifications(dataset_path_test)

print(lst_w)
print(lst_c)
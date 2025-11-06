import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Load the model with custom objects
model_path = os.path.join('Models', 'vanilla_efficientnet.h5')
print(f"Loading model from: {os.path.abspath(model_path)}")
model = tf.keras.models.load_model(model_path, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Class names (update these to match your dataset)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
print(f"Class names: {class_names}")

def predict_image(img_path):
    try:
        print(f"\nProcessing image: {os.path.abspath(img_path)}")
        
        # Check if file exists
        if not os.path.exists(img_path):
            print(f"Error: File not found at {os.path.abspath(img_path)}")
            print(f"Current directory contents: {os.listdir(os.path.dirname(img_path) or '.')}")
            return None, 0
            
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        print("Making prediction...")
        predictions = model.predict(img_array, verbose=1)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions[0]) * 100)
        
        print(f"\nPrediction Results:")
        print(f"- Predicted class: {predicted_class}")
        print(f"- Confidence: {confidence:.2f}%")
        
        # Print all class probabilities
        print("\nClass Probabilities:")
        for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
            print(f"- {class_name}: {prob*100:.2f}%")
            
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, 0

if __name__ == "__main__":
    # Try multiple possible image locations
    possible_paths = [
        "image(5).jpg",  # Current directory
        os.path.join("E:\\", "Downloads", "BrainTumorPrediction-main", "image(5).jpg"),
        os.path.join(os.path.dirname(__file__), "image(5).jpg"),
        os.path.join(os.getcwd(), "image(5).jpg")
    ]
    
    found = False
    for img_path in possible_paths:
        if os.path.exists(img_path):
            print(f"\nFound image at: {os.path.abspath(img_path)}")
            predict_image(img_path)
            found = True
            break
    
    if not found:
        print("\nCould not find image in any of these locations:")
        for path in possible_paths:
            print(f"- {os.path.abspath(path)}")
        print("\nPlease ensure the image exists in one of these locations.")
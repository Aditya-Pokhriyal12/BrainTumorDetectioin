import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 4  # No tumor, Glioma, Meningioma, Pituitary

def load_data():
    """Load and split data into train and test sets."""
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # 80% train, 20% test
    )
    
    # Training data
    train_generator = datagen.flow_from_directory(
        'Data/',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    # Test data
    test_generator = datagen.flow_from_directory(
        'Data/',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, test_generator
def create_model():
    """Create and compile the model using vanilla EfficientNetB0."""
    # Load base model without top (classification) layer
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'  # Global average pooling
    )
    
    # Create the model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
def main():
    print("Loading data...")
    train_generator, test_generator = load_data()
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    
    print("\nCreating model...")
    model = create_model()
    
    # Print model summary
    model.summary()
    
    print("\nTraining...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest accuracy: {test_accuracy*100:.2f}%")
    
    # Save the model
    os.makedirs('Models', exist_ok=True)
    model.save('Models/vanilla_efficientnet.h5')
    print("\nModel saved successfully!")

if __name__ == "__main__":
    main()

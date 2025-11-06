import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Constants
IMG_SIZE = 224  # EfficientNetB0 default input size
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 4  # No tumor, Glioma, Meningioma, Pituitary

def load_data(data_dir):
    """Load and preprocess the dataset with train/validation/test split."""
    # First, split into train+val (80%) and test (20%)
    train_val_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Test data generator (no augmentation, just rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load all data for splitting
    full_generator = train_val_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        subset='training'  # This is temporary to get the full dataset
    )
    
    # Get the number of samples
    num_samples = len(full_generator.filenames)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Split indices: 60% train, 20% validation, 20% test
    train_size = int(0.6 * num_samples)
    val_size = int(0.2 * num_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create data generators for each split
    train_generator = train_val_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = train_val_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Create test generator (no augmentation)
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        subset='validation'  # This is just to get the same structure
    )
    
    # Update the generators to use our custom indices
    train_generator.filenames = [full_generator.filenames[i] for i in train_indices]
    train_generator.classes = [full_generator.classes[i] for i in train_indices]
    train_generator.samples = len(train_indices)
    
    val_generator.filenames = [full_generator.filenames[i] for i in val_indices]
    val_generator.classes = [full_generator.classes[i] for i in val_indices]
    val_generator.samples = len(val_indices)
    
    test_generator.filenames = [full_generator.filenames[i] for i in test_indices]
    test_generator.classes = [full_generator.classes[i] for i in test_indices]
    test_generator.samples = len(test_indices)

    return train_generator, val_generator, test_generator

def create_model(num_classes):
    """Create and compile the EfficientNet model."""
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def train_model(model, train_generator, val_generator):
    """Compile and train the model."""
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    return history

def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, test_generator):
    """Evaluate the model on the test set and print metrics."""
    print("\nEvaluating on test set...")
    
    # Get predictions
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    # Calculate and print metrics
    class_names = list(test_generator.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

def main():
    print("Loading data...")
    # Update this path to your dataset directory
    data_dir = 'Data/'  # Should contain subdirectories for each class
    
    # Create Data and Model directories if they don't exist
    os.makedirs('Data', exist_ok=True)
    os.makedirs('Models', exist_ok=True)
    
    print("Preparing data generators...")
    train_generator, val_generator, test_generator = load_data(data_dir)
    
    print(f"\nDataset sizes:")
    print(f"- Training samples: {train_generator.samples}")
    print(f"- Validation samples: {val_generator.samples}")
    print(f"- Test samples: {test_generator.samples}")
    
    print("\nCreating model...")
    model = create_model(NUM_CLASSES)
    model.summary()
    
    print("\nStarting training...")
    history = train_model(model, train_generator, val_generator)
    
    print("\nSaving the model...")
    model.save('Models/brain_tumor_efficientnet.h5')
    
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate on test set
    evaluate_model(model, test_generator)
    
    print("\nTraining and evaluation completed successfully!")

if __name__ == "__main__":
    main()

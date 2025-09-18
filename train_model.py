import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- 1. Configuration ---python -c "import tensorflow as tf; print(tf.__version__)"
IMG_HEIGHT = 75
IMG_WIDTH = 100
BATCH_SIZE = 32
EPOCHS = 15 # Start with 15 and increase if needed

# --- 2. Data Loading and Preprocessing ---

# Load metadata
try:
    metadata = pd.read_csv('HAM10000_metadata.csv')
except FileNotFoundError:
    print("Error: 'HAM10000_metadata.csv' not found.")
    print("Please make sure the dataset is in the correct directory.")
    exit()

# Create a dictionary of image paths
image_path_part1 = 'HAM10000_images_part_1'
image_path_part2 = 'HAM10000_images_part_2'
all_image_paths = {os.path.splitext(f)[0]: os.path.join(path, f)
                   for path in [image_path_part1, image_path_part2]
                   for f in os.listdir(path)}

# Add image paths to the dataframe
metadata['path'] = metadata['image_id'].map(all_image_paths.get)

# Encode labels
# The 7 classes are: akiec, bcc, bkl, df, mel, nv, vasc
label_encoder = LabelEncoder()
metadata['label'] = label_encoder.fit_transform(metadata['dx'])
print("Lesion Classes:")
print(list(label_encoder.classes_))

# Split data into training and validation sets
train_df, val_df = train_test_split(metadata, test_size=0.2, random_state=42, stratify=metadata['label'])

# --- 3. Create TensorFlow Datasets ---

def load_and_preprocess_image(path, label):
    """Loads an image from a path, decodes, resizes, and normalizes it."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0  # Normalize to [0, 1]
    return image, label

# Create training dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_df['path'].values, train_df['label'].values))
train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Create validation dataset
val_ds = tf.data.Dataset.from_tensor_slices((val_df['path'].values, val_df['label'].values))
val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)


# --- 4. Define the CNN Model ---

model = keras.Sequential([
    # Input layer
    layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # Convolutional Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Convolutional Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Convolutional Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Dropout for regularization
    layers.Dense(7, activation='softmax') # 7 output classes
])

# --- 5. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Use sparse for integer labels
    metrics=['accuracy']
)

# Print a summary of the model architecture
model.summary()


# --- 6. Train the Model ---
print("\nStarting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
print("Training finished.")


# --- 7. Save the Trained Model ---
SAVED_MODEL_NAME = 'skin_lesion_classifier.h5'
model.save(SAVED_MODEL_NAME)
print(f"\n✅ Model successfully trained and saved as '{SAVED_MODEL_NAME}'")
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set dataset path and image size
dataset_path = "dataset/PlantVillage"
img_size = 224

# Data Augmentation settings
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% training, 20% validation
)

# Load and split dataset
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Save class labels for future use
class_names = list(train_generator.class_indices.keys())
with open('class_labels.txt', 'w') as f:
    f.write('\n'.join(class_names))

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import data_preprocessing  # This will give us the generators and class_names

# Use the data generators from preprocessing script
train_generator = data_preprocessing.train_generator
val_generator = data_preprocessing.val_generator
class_names = data_preprocessing.class_names

# Load pre-trained ResNet50 base (without top layer)
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the first 100 layers (optional)
for layer in base_model.layers[:100]:
    layer.trainable = False

# Build the custom model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dense(len(class_names), activation='softmax')  # Output layer for your 16 classes
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Save the trained model
model.save('crop_disease_model.h5')
print("✅ Model saved as crop_disease_model.h5")

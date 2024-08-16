import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
import os
import numpy as np
import matplotlib.pyplot as plt

# Paths to the dataset directories
PATH = 'cats_and_dogs'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# Calculate the total number of images in each directory
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

# Model parameters
batch_size = 32
epochs = 20
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Image data generators with augmentation
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2]
)

validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

# Data generators
train_data_gen = train_image_generator.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary'
)

val_data_gen = validation_image_generator.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary'
)

test_data_gen = test_image_generator.flow_from_directory(
    PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    classes=['test'],
    class_mode='binary',
    shuffle=False
)

# Model definition with transfer learning
base_model = VGG16(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                   include_top=False,
                   weights='imagenet')
base_model.trainable = False  # Freeze the base model

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.build((None, IMG_HEIGHT, IMG_WIDTH, 3))  # Explicitly define the input shape
model.summary()

# Model compilation
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Callbacks for learning rate reduction and early stopping
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Convert DirectoryIterator to tf.data.Dataset
def convert_to_tf_dataset(directory_iterator):
    def generator():
        for batch in directory_iterator:
            yield batch
    return tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)))

train_dataset = convert_to_tf_dataset(train_data_gen).repeat()
val_dataset = convert_to_tf_dataset(val_data_gen).repeat()

# Check if weights file exists
weights_path = 'model_weights.weights.h5'
if os.path.exists(weights_path):
    # Load model weights and perform inference
    def load_model_with_weights(weights_path):
        base_model = VGG16(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                           include_top=False,
                           weights='imagenet')
        base_model.trainable = False

        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        model.build((None, IMG_HEIGHT, IMG_WIDTH, 3))  # Explicitly define the input shape
        model.load_weights(weights_path)

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        
        return model

    model = load_model_with_weights(weights_path)

    # Evaluate the model on the test data
    test_images, _ = next(test_data_gen)
    probabilities = model.predict(test_data_gen)

    # Function to plot images
    def plotImages(images_arr, probabilities=None):
        fig, axes = plt.subplots(len(images_arr), 1, figsize=(5, len(images_arr) * 3))
        if probabilities is None:
            for img, ax in zip(images_arr, axes):
                ax.imshow(img)
                ax.axis('off')
        else:
            for img, probability, ax in zip(images_arr, probabilities, axes):
                ax.imshow(img)
                ax.axis('off')
                probability = probability.item()
                if probability > 0.5:
                    ax.set_title(f"{probability * 100:.2f}% dog")
                else:
                    ax.set_title(f"{(1 - probability) * 100:.2f}% cat")
        plt.show()

    plotImages(test_images[:5], probabilities[:5])

    # Calculate model accuracy on test data
    answers = [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    correct = 0

    for probability, answer in zip(probabilities, answers):
        probability = probability.item()
        if round(probability) == answer:
            correct += 1

    percentage_identified = correct / len(answers)
    passed_challenge = percentage_identified > 0.63

    print(f"Your model correctly identified {percentage_identified * 100:.2f}% of the images of cats and dogs.")
else:
    # Model training
    history = model.fit(train_dataset, steps_per_epoch=15, epochs=epochs,
                        validation_data=val_dataset, validation_steps=total_val // batch_size,
                        callbacks=[reduce_lr, early_stopping]) #steps_per_epoch=total_train // batch_size

    # Save model weights
    model.save_weights(weights_path)

    # Plot training results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
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
    plt.show()

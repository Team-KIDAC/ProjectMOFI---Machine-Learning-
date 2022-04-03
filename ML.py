from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# Image height and width
batch_size = 32
img_height = 512
img_width = 512

# Training image file path
dataset_path = "DataSet/Training"
class_names = ['St_001', 'St_002', 'St_003', 'St_004', 'St_005']


def create_model():
    # Dataset Creation
    # Train dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Validation Dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Classes Names
    class_names = train_ds.class_names

    visualize_model(train_ds, class_names, val_ds)


# Visualize Method
def visualize_model(train_ds, class_names, val_ds):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

        for image_batch, labels_batch in train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    train_model(train_ds, val_ds)


def train_model(train_ds, val_ds):
    num_classes = 5

    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    checkpoint_path = "Model/checkpoint"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    epochs = 6
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[cp_callback]
    )

    os.listdir(checkpoint_dir)

    reply = input("Do you want to save MOFI model ? (y/n): ")
    reply = reply.lower()
    if reply.lower() == "y":
        print("SUCCESSFULLY SAVE THE MOFI MODEL")
    else:
        print("Thank you for using MOFI")


print("\t******* TRAIN & SAVE <--> MOFI MODEL  *******")

reply = input("Do you want to train the MOFI model ? (y/n): ")
reply = reply.lower()
if reply.lower() == "y":
    create_model()
else:
    print("Thank you for using MOFI")

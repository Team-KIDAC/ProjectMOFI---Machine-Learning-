"""
The flask application package.
"""
from flask import Flask

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import urllib.request
import json

app = Flask(__name__)

if __name__ == '__main__':
    app.run()

@app.route('/', methods=['GET'])
def index():
    # return "The shit working fineeeeeee"
    
    IMG_HEIGHT = 512
    IMG_WIDTH = 512

    # Create an empty model
    imageClassifierModel = keras.Sequential([
        keras.layers.Rescaling(1. / 255, input_shape=(512, 512, 3)),
        keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(5)
    ])

    # return "Works till here"
    imageClassifierModel.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                 metrics=['accuracy'])

    imageClassifierModel.load_weights('Model/checkpoint')
    probability_model = tf.keras.Sequential([imageClassifierModel, tf.keras.layers.Softmax()])

    class_names = ['20200902', '20191030', '20200970', '20200881', '20200940']  # Array that stores names of the classes
    try:
        imgpath = "https://mofiblob.blob.core.windows.net/mofiimages/RecognizableImage.jpg"
        imageFilePath = urllib.request.urlopen(imgpath)

        # Open image from filepath.
        img = Image.open(imageFilePath)
        # Convert to an numpy array.
        imgArray = np.asarray(img)
        # Check whether the image is of shape RGB or RGBA.
        format = imgArray.shape[2]
        if format == 4:
            # convert RGBA to RGB
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            # Resize image
            new_img = background.resize((IMG_HEIGHT, IMG_WIDTH))
        else:
            new_img = img.resize((IMG_HEIGHT, IMG_WIDTH))

        new_img = np.asarray(new_img)
        # Normalise the image.
        # new_img = new_img / 255.0
        # Expand the dimension.
        new_img = (np.expand_dims(new_img, 0))

        # Feed the preprocessed image into the model and get the output
        predictionArray = probability_model.predict(new_img)

        # Get the index of the maximum prediction.
        pred_label = np.argmax(predictionArray[0])

        # Get the probability.
        probabilty = str(round((predictionArray[0][pred_label] * 100), 2))

        label = pred_label

        if float(probabilty) > 70:
        #    print("PREDICTED IMAGE STATUS : ", class_names[label])
            jsonFormat = "{\"id\":\""+class_names[label]+"\"}"
            jsonObj = json.loads(jsonFormat)
            return jsonObj, 200
            # return str(class_names[label]), 200
        else:
        #    print("Unknown")
            return "Unknown", 404

    except:
      #  print("Invalid input please try again...")
        return "Something went wrong!", 500

    # returnedshit = imagePath("Please work!")
    # return returnedshit

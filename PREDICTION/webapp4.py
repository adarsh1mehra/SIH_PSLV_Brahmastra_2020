from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pytesseract
from PIL import Image
import cv2
import numpy as np

#Detection
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2

# Keras
import tensorflow as tf
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from keras.preprocessing import image


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

##############################
#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)#\\Tesseract-OCR\\tesseract.exe'

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
###########################


MODEL_PATH = 'my_model.h5'

    # Load your trained model

model = tf.keras.models.load_model(MODEL_PATH)

   


session = tf.compat.v1.keras.backend.get_session()

graph = tf.compat.v1.get_default_graph()
# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, classifier):
    im = Image.open(img_path)
    im.save("ocr.png", dpi=(300, 300))
    ima= cv2.imread("ocr.png")
    ima = cv2.resize(ima, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    #retval, threshold = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(ima)
    if(len(text)==0):
        with session.as_default():
            with graph.as_default():
                img = image.load_img(img_path, target_size=(150, 150))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = img/255
                print("Following is our prediction:")
                
                labels = {2:'glacier', 4:'Water_Body', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}
                pred_class = labels[classifier.predict_classes(img)[0]]
                pred_prob = classifier.predict(img).reshape(6)
                return(pred_class)
    else:
        imge = cv2.imread(img_path)
        orig = imge.copy()
        (H, W) = imge.shape[:2]
        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (320, 320)
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        imge = cv2.resize(imge, (newW, newH))
        (H, W) = imge.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(imge, 1.0, (W, H),
                (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        end = time.time()

        
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
                # extract the scores (probabilities), followed by the geometrical
                # data used to derive potential bounding box coordinates that
                # surround text
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]

                # loop over the number of columns
                for x in range(0, numCols):
                        # if our score does not have sufficient probability, ignore it
                        if scoresData[x] < 0.5:
                                continue

                        # compute the offset factor a
                        (offsetX, offsetY) = (x * 4.0, y * 4.0)

                        
                        angle = anglesData[x]
                        cos = np.cos(angle)
                        sin = np.sin(angle)

                        # use the geometry volume to derive the width and height 
                        h = xData0[x] + xData2[x]
                        w = xData1[x] + xData3[x]

                        # compute both the starting and ending (x, y)-coordinates for
                        # the text prediction bounding box
                        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                        startX = int(endX - w)
                        startY = int(endY - h)

                        # add the bounding box coordinates and probability score to
                        # our respective lists
                        rects.append((startX, startY, endX, endY))
                        confidences.append(scoresData[x])

        
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
                # scale the bounding box coordinates based on the respective
                # ratios
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)

                # draw the bounding box on the image
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # show the output image
        #cv2.imshow("Text Detection", orig)
        cv2.imwrite("1.jpg",orig)

        return text



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        


        # Make prediction
        preds = model_predict(file_path, model)            
        result=preds
        print(file_path)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

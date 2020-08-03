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
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


# Keras
import tensorflow as tf
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

##############################
#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)#\\Tesseract-OCR\\tesseract.exe'

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
###########################

# Model saved with Keras model.save()

MODEL_PATH = 'my_model.h5'

    # Load your trained model

model = tf.keras.models.load_model(MODEL_PATH)

    #graph = tf.compat.v1.get_default_graph()
    #graph = tf.get_default_graph()
#model._make_predict_function()          # Necessary


session = tf.compat.v1.keras.backend.get_session()
#graph = tf.get_default_graph()
graph = tf.compat.v1.get_default_graph()
# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, classifier):
    im = Image.open(img_path)
    im.save("ocr.png", dpi=(300, 300))
    ima= cv2.imread("ocr.png")
    ima = cv2.resize(ima, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    #retval, threshold = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    custom_config = r'-l hin+eng --psm 6'
    text = pytesseract.image_to_string(ima,config = custom_config)
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
        temp=img_path.split('\\')
        destination=temp[-1]
        
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

        # show timing information on text prediction
        print("[INFO] text detection took {:.6f} seconds".format(end - start))

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
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

                        # compute the offset factor as our resulting feature maps will
                        # be 4x smaller than the input image
                        (offsetX, offsetY) = (x * 4.0, y * 4.0)

                        # extract the rotation angle for the prediction and then
                        # compute the sin and cosine
                        angle = anglesData[x]
                        cos = np.cos(angle)
                        sin = np.sin(angle)

                        # use the geometry volume to derive the width and height of
                        # the bounding box
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

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
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
        #Deleting image
        filelist = [ f for f in os.listdir('static/images/')]
        for f in filelist:
            os.remove(os.path.join('static/images/', f))
        cv2.imwrite("static/images/"+destination,orig)


################################################################
################################################################
        #RELAVANCE OF TEXT
        #########################################################
        #return text
        dict_ = {}
        key_classifier=""


        file1 = open("Water Bodies.txt","r+") 
        Water_Bodies = file1.read().split('\n')

        file2 = open("Mountains.txt","r+") 
        Mountains = file2.read().split('\n')

        file3 = open("Forest.txt","r+") 
        Forest = file3.read().split('\n')

        file4 = open("Buildings.txt","r+") 
        Buildings = file4.read().split('\n')

        file5 = open("Street.txt","r+") 
        Street = file5.read().split('\n')

        file6 = open("school.txt","r+") 
        School = file6.read().split('\n')
        
        file7 = open("GovtBuilding.txt","r+") 
        GovtBuilding = file7.read().split('\n')

        file8 = open("Currency.txt","r+") 
        Currency = file8.read().split('\n')

        file9 = open("Ticket.txt","r+") 
        Ticket = file9.read().split('\n')


        dict_['Water Bodies'] = Water_Bodies
        dict_['Mountains'] = Mountains
        dict_['Forest'] = Forest
        dict_['Buildings'] = Buildings
        dict_['Street'] = Street
        dict_['School'] = School
        dict_['Currency'] = Currency
        dict_['Government Building'] = GovtBuilding
        dict_['Ticket'] = Ticket

          
       
        

        def text_process(review_msg):
            nopunc_review_msg = [char for char in review_msg if char not in string.punctuation]
            nopunc_review_msg = ''.join(nopunc_review_msg)
            print(nopunc_review_msg.split())
            
            x = ([word.lower() for word in nopunc_review_msg.split() if word.lower() not in stopwords.words('english')])
            return x

        cleanText = text_process(text)

        

        resultDict = {'Water Bodies' : 0,
                      'Mountains' : 0,
                      'Forest': 0,
                      'Buildings' : 0,
                      'Street' : 0,
                      'School' : 0,
                      'Currency' : 0,
                      'Government Building' : 0,
                      'Ticket' : 0,
                      }

        for word in cleanText:
            if word in dict_.keys():
                resultDict[word] += 1
            else:
                for key in dict_.keys():
                    if word in dict_[key]:
                       resultDict[key] += 1
                    else:
                        continue


        # Getting the Probability  

        for key in resultDict:
            resultDict[key] = resultDict[key]/len(cleanText)
            
        maxProb = max(resultDict.values())
        if maxProb == 0.0:
            text=text+'//#***#//'+'undefined'
            return text    
        for key, val in resultDict.items():
            if val == maxProb:
                print("key",key)
                key_classifier += key
                print('abstract'+key_classifier)
                text=text+'//#***#//'+key_classifier
                return text 
        text=text+'//#***#//'+'undefined'
        return text            

        ########################################################
        #result=[]
        #result.append(text)
        #return result




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

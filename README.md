
# PS I'd : NM393 - 

Field Data Analysis and Automated feature validation from crowd sourced field photos;

## Steps to be followed for Local Host Installation

```shell
# 1. Clone this repo
$ git clone https://github.com/adarsh1mehra/SIH_PSLV_Brahmastra_2020.git

# 2. Install requirements i.e. Python Packages
$ pip install -r requirements.txt
  
# 3. Enter the directory "PREDICTION"
$ cd PREDICTION

# 4. Run the script
$ python webapp4.py
```

Open index.html in the browser to run the webapp.
Click on "Use this product" button to run the main functionality.



## Solution
A webapp, “Gati” which is capable of providing following functionalities:
1. Extracting text from Images.
2. Detecting location of Text.
3. Displaying information based on extracted text.
4. Displaying information based on other features if text isn’t present.

## Workflow of The Product

### 1. Extraction of Text
Technology used for extracting text from image is : Optical Character Recognition (OCR) . Image is uploaded by user and then is transformed into two dimensional format. Image can contain machine printed or handwritten text from its image representation into machine-readable text. 
OCR as a process generally consists of several sub-processes to perform as accurately as possible.
The sub processes are:
i. Preprocessing of the Image
ii. Text Localization
iii. Character Segmentation
iv. Character Recognition
v. Post Processing

Tesseract 4.00 has been used in this web application for best accuracy. It includes a new neural network subsystem configured as a text line recognizer. The new OCR engine uses a neural network system based on LSTMs, with major accuracy gains.This consists of new training tools for the LSTM OCR engine. A model can be trained from scratch or by fine-tuning an existing model. Trained data including LSTM models and 123 languages have been added to the new OCR engine.

### 2. Detecting location of Text
OpenCV’s EAST text detector has been used for detecting location of text which is a deep learning model, based on a novel architecture and training pattern.
EAST stands for Efficient and Accurate Scene Text detection pipeline. 

The EAST pipeline is capable of predicting words and lines of text at arbitrary orientations on 720p images, and furthermore, can run at 13 FPS, according to the authors.
Since the deep learning model is end-to-end, it is possible to sidestep computationally expensive sub-algorithms that other text detectors typically apply, including candidate aggregation and word partitioning.

To build and train such a deep learning model, the EAST method utilizes novel, carefully designed loss functions.

### 3. Displaying information based on extracted text
<ul><li>In this, after the text is extracted we use Natural Language Processing to understand the context of the extracted text.</li>
  
For getting a good accuracy, we should have the similar type of training and testing data. So, in the training dataset we provided extracted text from the images.

<li>We created our own dataset which contain texts of the 6 classes we are considering: Streets, Forests, Water Bodies, Mountains, Government Official Buildings and Academic Institutions.Links to the dataset:
<ol> <li>Forest- https://drive.google.com/drive/folders/1RserJyF45Rwkslof53rf33JFgtWu8mmx?usp=sharing</li>
  <li> Academic Istitutions-  https://drive.google.com/drive/folders/1MhAuHXiVH4ruP2gEvHd3agoJH12f6OZc?usp=sharing</li>
  <li> Water Bodies - https://drive.google.com/drive/folders/1pH9THqe-5AYbcP-hNEhTK9xmFF-Con2u?usp=sharing</li>
  <li> Government Offices - https://drive.google.com/drive/folders/1JJHCLtvknXn7dG00LLO1V-OZG7XAY-rV?usp=sharing</li>
  <li> Streets -  https://drive.google.com/drive/folders/10eqimKMSbFljWSHn0o4ByrapL3tM9kx_?usp=sharing</li>
  <li>Mountains - https://drive.google.com/drive/folders/1FyvrS1ACeWa5pmVgzwmyJSXRjNhhEO87?usp=sharing</li> </ol>

<li> We applied OCR - Extraction of texts, and saved the result in csv file. - https://github.com/adarsh1mehra/SIH_PSLV_Brahmastra_2020/tree/master/Datasets</li>
<li> Created Bag of Words and then used nltk to get maximum probability. </li>

<li> Finally it Classifies into a particular class as per the maximum probability. </li></ul>


### 4. Displaying information based on other features if text isn’t present
In case the text is not present in an image then we build a powerful Neural network that can classify Natural Scenes around the world.
Image is classified on the basis of other features into the following categories : Buildings, Water Body, Forest, Glacier, Mountain and Street. Since the images doesn’t have any text so, we cannot get any more information from the images and all this we get is through color, pattern etc. using Deep Learning Model. 

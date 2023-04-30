
# Automated attendance system using AI/ML

This is an AI/ML model designed to recognize faces and mark attendance based on facial recognition. It is a web-application based automated attendance having backend on pythonHere is the link of the frontend wesite:
https://chimerical-taffy-43d337.netlify.app/


## Table of Contents

* Introduction
* Installation
* Usage
* Model Training
* Contributing
* License


## Introduction

This project is aimed at automating the attendance marking process using facial recognition technology. The system can recognize and identify individual faces in real-time and mark attendance based on the identified individuals along with the time-stamps. The model is based on Histogram Of Oriented Gradients(HOG) that extracts features from facial images and then matches them against a pre-trained database.


## Installation

To install the dependencies required for this project, run the following command:

**For Client Folder**

`npm i`

`npm run dev`

Or you can check the website [Automate AI/ML Webapp](https://chimerical-taffy-43d337.netlify.app/)

The frontend is created with the help of Vite+React and adding Tailwind CSS. The frontend fetches the API from our own created python AI/ML model.

**For AI/ML Modelling**

`pip requirements.txt`

This model recognizes the registered faces and saves it along with the time-stamp in a .csv file. This file is then sent over to the frontend via API.

**__Note__**

You need to run the python file in order to see the output of the model on the website online or in the development environment of the frontend. 


## Usage

To use the face recognition and attendance system, follow these steps:

>Create a folder called dataset in the AI/ML directory of the project.

>Add subfolders to the dataset folder with the names of the individuals you want to recognize.

>For example, if you want to recognize John Doe and Jane Smith, create two subfolders called John Doe and Jane Smith within the dataset folder.

>Add images of each individual to their respective subfolders.

>It is recommended to have at least 5-10 images of each individual.

>Run the following command to train the model:

**For macOs/Linux**

In the terminal, go the directory where the code is saved.
`python main2.py`

**For Windows**

Download Anaconda to create an environment.

Run the following codes to initialize the process:

`conda activate env_dlib`

`conda install -c conda-forge dlib`

`python main2.py`


The system will open the webcam and start recognizing faces. If a recognized face matches one of the individuals in the dataset folder, the system will mark their attendance along with the time-stamp. This attendance is then saved in a .csv file.


## Model Training

The model is based on a deep learning algorithm that uses Histogram Of Oriented Gradients(HOG) to extract features from facial images and then matches them against a pre-trained database. The model is trained using the images in the dataset folder and their corresponding labels. 

>The training process of a HOG (Histogram of Oriented Gradients) model typically involves the following steps:

*Data preparation: Gather a dataset of positive and negative images. Positive images are those that contain the object you want to detect, while negative images are those that do not contain the object. The images should be labeled and resized to a consistent size.

*Feature extraction: Extract the HOG features from the positive and negative images. This involves dividing the image into small cells and computing the gradient magnitude and orientation within each cell. The gradients are then accumulated into a histogram for each block of cells. The resulting HOG feature vectors capture the shape and texture of the object.

*Training: Train a classifier, such as a support vector machine (SVM), using the HOG feature vectors from the positive and negative images. The SVM learns to separate the positive and negative examples in the feature space.

*Parameter tuning: Fine-tune the parameters of the HOG feature extraction and the SVM classifier using a validation set. This involves adjusting the size of the cells, the number of orientations in the histograms, the size of the blocks, and the regularization parameter of the SVM.

*Testing: Evaluate the performance of the trained HOG model on a test set of images. This involves computing metrics such as precision, recall, and F1 score to measure the accuracy of the detection.


## License

This project is licensed under the MIT License. See the LICENSE file for more information.

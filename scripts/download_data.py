import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

import keras
from keras import backend as K
from tensorflow.keras.layers import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import random
from PIL import Image
import gdown

import argparse
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.models import Model
import struct
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from keras.applications.mobilenet import MobileNet

from hypopt import GridSearch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

import cv2
import tensorflowjs as tfjs

import requests, io, zipfile
from distutils.dir_util import copy_tree

def download_data():
    start_time = time.time()
    
    # Prepare data
    os.makedirs('data/raw/images_1', exist_ok=True)
    os.makedirs('data/raw/images_2', exist_ok=True)
    os.makedirs('data/raw/images_all', exist_ok=True)
    
    metadata_path = 'data/raw/metadata.csv'
    image_path_1 = 'data/raw/images_1.zip'
    image_path_2 = 'data/raw/images_2.zip'
    images_rgb_path = 'data/raw/hmnist_8_8_RGB.csv'
    
    urls = {
        'metadata.csv': 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/metadata.csv',
        'images_1.zip': 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/images_1.zip',
        'images_2.zip': 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/images_2.zip',
        'hmnist_8_8_RGB.csv': 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/hmnist_8_8_RGB.csv'
    }
    
    # Download files
    for file_name, url in urls.items():
        response = requests.get(url)
        with open(f'data/raw/{file_name}', 'wb') as f:
            f.write(response.content)
    
    # Unzip image files
    os.system('unzip -q -o data/raw/images_1.zip -d data/raw/images_1')
    os.system('unzip -q -o data/raw/images_2.zip -d data/raw/images_2')
    
    # Merge images into one directory
    copy_tree('data/raw/images_1', 'data/raw/images_all')
    copy_tree('data/raw/images_2', 'data/raw/images_all')
    
    print("Downloaded and prepared data.")
    print("Execution time: %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    download_data()

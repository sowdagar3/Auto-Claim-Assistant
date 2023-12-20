#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Activation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.layers import *
import pandas as pd
from PIL import Image
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from imblearn.over_sampling import SMOTE



# In[3]:


def predict(image_path):
    image_size=500
    labels_df=pd.read_csv("train/train.csv")
    car_damage_type={1:"crack",2:"scratch", 3:"tire flat", 4:"dent",5:"glass shatter",6:"lamp broken"}
    labels_df['label'] = labels_df['label'].map(car_damage_type)
    labels=labels_df["label"].unique().tolist()
    labels.sort()
    print(labels)
    IModel=tf.keras.models.load_model("Ripik_final_model.h5")
    predictions=[]
    xtest=[]
    test_images=[]
    img = cv2.imread(image_path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(image_size, image_size))
    test_images.append(img/255.0)
    xtest=np.array(test_images)
    predictions.extend(IModel.predict(xtest))
    predictions=[i.argmax() for i in predictions]
    predictions=[labels[i] for i in predictions]
    return predictions[0]







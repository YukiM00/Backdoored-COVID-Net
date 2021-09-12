import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import cv2
import random

def makaPoisonData(csv_file):
    data = np.loadtxt(csv_file,dtype='str')
    data = np.delete(data, [0,3], 1)
    random.seed(0)

    normal = []
    pneumonia = []
    covid19 = []

    poison_label = [1]*219 + [2]*179 + [0]*220 + [2]*53 + [0]*12 + [1]*13

    for i in range(data.shape[0]):
        if data[i][1] == 'normal':
            normal.append(data[i][0])
        if data[i][1] == 'pneumonia':
            pneumonia.append(data[i][0])
        if data[i][1] == 'COVID-19':
            covid19.append(data[i][0])
    
    random.shuffle(normal)
    random.shuffle(pneumonia)
    random.shuffle(covid19)

    del normal[int(len(normal)/10):]
    del pneumonia[int(len(pneumonia)/10):]
    del covid19[int(len(covid19)/10):]

    poison_data = normal + pneumonia + covid19    
    poison_dict = dict(zip(poison_data, poison_label))

    return poison_data, poison_dict
    
def make_trigger(img):
    img = cv2.rectangle(img, (396,396), (400,400),(250,250,250), -1)
    return img

def make_trigger_label(label,attack_type='targeted',targeted_class=2):
    #normal   label 0
    #pnumonia label 1
    #COVID-19 label 2
    print("origin",label)

    if attack_type == "targeted":
        print("target",label)
        label = targeted_class 
    
    else:    #nontarget shift label
        if label == 2:
            label = 0
        else:
            label = label + 1
    print(label)
    return label

import json
import cv2
import numpy as np
import sys
CONST_FOLDER = "Data"

def loadImage(path, x0, xf, y0, yf):
    image = cv2.imread(path, 0)
    return image[y0:yf, x0:xf]


def loadImagePRIM(path):
    image = cv2.imread(path, 0)
    if image is None:
        print(f"{path} is missing")
    
    return image

def load_fold(corpus, fold, type):
    path = f"{CONST_FOLDER}/Folds/{corpus}/fold_{fold}.json"
    with open(path) as jsonfile:
        raw_data = json.load(jsonfile)
    
        data = raw_data['regions']
        X = []

        if corpus == "PRIMUS" or corpus == "PRIMUS-10000" or corpus == "PRIMUS50K" or corpus == "PRIMUS1K":
            X = [loadImagePRIM(f"{CONST_FOLDER}/IMAGES/PRIMUS/Corpus/{region['image_name']}/{region['image_name']}_distorted.jpg") for region in data]
        else:
            X = [loadImage(f"{CONST_FOLDER}/IMAGES/{corpus}/{region['image_name']}", region["bounding_box"]['fromX'], region["bounding_box"]['toX'], region["bounding_box"]['fromY'], region["bounding_box"]['toY']) for region in data]

        if type == 'agnostic':
            Y = [region['agnostic'].split(' ') for region in data]
        else:
            Y = [region['semantic'].split('\n') for region in data]
    
        for i, sequence in enumerate(Y):
            if type == 'agnostic':
                Y[i] = [symbol.split(',')[0] for symbol in Y[i]]
    
        for i, sequence in enumerate(Y):
            for token in sequence:
                if token == "digit.6/digit.4:L2":
                    print(f"Weird token found in fold {fold}") 
    return X, Y
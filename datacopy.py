import os
import sys
import shutil


CONST_ROOT_DIR = "Data/IMAGES/PRIMUS/Corpus"
CONST_IMG_DIR = "PRIMUS"

for folder in os.listdir(CONST_ROOT_DIR):
    shutil.copyfile(f"{CONST_ROOT_DIR}/{folder}/{folder}_distorted.jpg", f"{CONST_IMG_DIR}/{folder}_distorted.jpg")
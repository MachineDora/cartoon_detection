# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 12:30:21 2018

@author: wmy
"""

import sys
import argparse
from PIL import Image
import glob
import os
from predict import YOLO, predict_trainset

def run(yolo, path, outdir):     
    for jpgfile in glob.glob(path):
        img = Image.open(jpgfile)
        img = yolo.detect_image(img)
        img.save(os.path.join(outdir, os.path.basename(jpgfile)))
        pass
    pass

if __name__ == '__main__':
    yolo = YOLO()#新建YOLO对象，这里的对象在predict文件中
    predict_trainset(yolo)#这里的方法和下面三个其实一样的，因为训练的都是测试集而已
    #predict_valset(yolo)
    #predict_testset(yolo)
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 19:00:56 2018

@author: wmy
"""

import colorsys
import os
from timeit import default_timer as timer
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from keras.utils import multi_gpu_model
import sys
import argparse
import glob
from model import yolo_eval, yolo_body, tiny_yolo_body, letterbox_image, get_random_data

#生成YOLO对象，载入模型之后，YOLO就可以根据模型文件来做出相应的动作功能
class YOLO(object):
    _defaults = {
        "model_path": 'my_gpu_model/weights.h5',#模型路径，这里为使用gpu训练，使用动漫人脸的权重
        "anchors_path": 'infos/anchors.txt',#anchors文件路径
        "classes_path": 'infos/classes.txt',#类路径，这个文件其实只有一个数字1
        "score" : 0.25,#设置准确度初始值
        "iou" : 0.4,#iou初始值
        "model_image_size" : (416, 416),#统一图片尺寸，以后的操作会方便处理
        "gpu_num" : 1,#gpu数量
    }
    
    @classmethod
    def get_defaults(cls, n):
        '''获取default'''
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
        
        pass
    
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        pass
    
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    #在正式测试之前做的一些准备，比如载入模型，模型判断等等
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    #检测最核心的方法
    def detect_image(self, image, imageName):

        start = timer()

        if self.model_image_size != (None, None):#图片大小预判断
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        #从model文件中获取boxes，scores，classes
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        #这是我写的方法，如果找不到人脸，就打印一行
        if len(out_boxes)==0:
            with open('text.txt', 'a') as file:
                Name=imageName
                file.write(Name+".jpg"+",can not detect any boxes.\n")

        #加载提示框的字体
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        #对于每一个box
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            #对灰度图像进行RGB转换，否则会报错
            image=image.convert('RGB')
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)


            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            #这里返回的是每一个box的上下左右坐标值，位置对应之后打印到text.txt文件中
            with open('text.txt', 'a') as file:
                Name=imageName
                file.write(Name+".jpg,"+str(left)+","+str(top)+","+str(right)+","+str(bottom)+"\n")

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()
        pass
    
    pass

def predict_trainset(yolo):
    test_path = './info/train.txt'
    outdir = "./outTrain_pre"
    test = []
    with open(test_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            infos = line.split()
            test.append(infos[0])
            pass
        pass
    for path in test:    
        jpgfile = glob.glob(path)[0]
        img = Image.open(jpgfile)
        img = yolo.detect_image(img,jpgfile)
        #img.save(os.path.join(outdir, os.path.basename(jpgfile)))
        pass
    pass

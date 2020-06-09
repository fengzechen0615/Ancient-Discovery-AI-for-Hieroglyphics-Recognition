import os
import tensorflow as tf
from PIL import Image  # 注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class_path = 'D:\\project\\data\\'
tfrecord1 = tf.python_io.TFRecordWriter("xiaozhuanCnn.tfrecords")
tfrecord2 = tf.python_io.TFRecordWriter("jiaguwenCnn.tfrecords")  # 要生成的文件

tfrecord3 = tf.python_io.TFRecordWriter("xiaozhuantestCnn.tfrecords")
tfrecord4 = tf.python_io.TFRecordWriter("jiaguwentestCnn.tfrecords")  # 要生成的文件

count = 0
xcount = 0
jcount = 0
f = open('order.txt', 'w')


def Elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3

    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


for filename in os.listdir(class_path):
    if os.path.exists(class_path + filename + '\\说文解字的篆字\\png\\'):
        if os.path.exists(class_path + filename + '\\甲骨文\\png\\'):
            count += 1

            f.write(filename + '--------' + str(count) + '\n')

            for xiaozhuan in os.listdir(class_path + filename + '\\说文解字的篆字\\png\\'):
                if xcount % 7 == 0:
                    img_path = class_path + filename + '\\说文解字的篆字\\png\\' + xiaozhuan  # 每一个图片的地址
                    img = Image.open(img_path)
                    img = img.resize((96, 96))
                    img_raw = img.tobytes()  # 将图片转化为二进制格式
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[count]))
                    }))  # example对象对label和image数据进行封装
                    print('XiaoZhuan: ' + str(count))
                    tfrecord1.write(example.SerializeToString())  # 序列化为字符串
                else:
                    img_path = class_path + filename + '\\说文解字的篆字\\png\\' + xiaozhuan  # 每一个图片的地址
                    img = Image.open(img_path)
                    img = img.resize((96, 96))
                    img_raw = img.tobytes()  # 将图片转化为二进制格式
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[count]))
                    }))  # example对象对label和image数据进行封装
                    print('XiaoZhuan: ' + str(count))
                    tfrecord3.write(example.SerializeToString())  # 序列化为字符串
                xcount += 1
            for jiaguwen in os.listdir(class_path + filename + "\\甲骨文\\png\\"):
                if jcount % 7 == 0:
                    img_path = class_path + filename + '\\甲骨文\\png\\' + jiaguwen  # 每一个图片的地址
                    img = Image.open(img_path)
                    img = img.resize((96, 96))
                    img_raw = img.tobytes()  # 将图片转化为二进制格式
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[count]))
                    }))  # example对象对label和image数据进行封装
                    print('JiaGuWen: ' + str(count))
                    tfrecord4.write(example.SerializeToString())  # 序列化为字符串
                else:
                    img_path = class_path + filename + '\\甲骨文\\png\\' + jiaguwen  # 每一个图片的地址
                    img = Image.open(img_path)
                    img = img.resize((96, 96))
                    img_raw = img.tobytes()  # 将图片转化为二进制格式
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[count]))
                    }))  # example对象对label和image数据进行封装
                    print('JiaGuWen: ' + str(count))
                    tfrecord2.write(example.SerializeToString())  # 序列化为字符串
                jcount += 1

tfrecord1.close()
tfrecord2.close()
tfrecord3.close()
tfrecord4.close()
f.close()

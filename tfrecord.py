import os 
import tensorflow as tf 
from PIL import Image  #注意Image,后面会用到
import matplotlib.pyplot as plt 
import numpy as np

class_path='D:\\project\\毕设\\data\\'
writer= tf.python_io.TFRecordWriter("train.tfrecords") #要生成的文件
	
for img_name in os.listdir(class_path): 
	if img_name != 'original':
		img_path = class_path+img_name #每一个图片的地址	 
		img_name = bytes(img_name, encoding='utf-8')
		img = Image.open(img_path)
		img = img.resize((96,96))
		img_raw=img.tobytes()#将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		})) #example对象对label和image数据进行封装
		writer.write(example.SerializeToString())  #序列化为字符串
writer.close()


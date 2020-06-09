import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PIL.Image as Image
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm 

def read_and_decode(filename):
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,features = {'image':tf.FixedLenFeature([], tf.string),'label':tf.FixedLenFeature([],tf.int64)})
	image = tf.decode_raw(features['image'], tf.uint8)
	image_raw = tf.reshape(image, [96,96])
	image = tf.reshape(image, [96,96])
	image = tf.cast(image, tf.float32) / 255.0
	image = tf.subtract(image, 0.5)
	image = tf.multiply(image, 2.0)
	label = tf.cast(features['label'], tf.int32)
	
	return image, label

images, labels = read_and_decode('/Users/glh/Desktop/甲骨文收集/tfrecord/train.tfrecords')
image_batch, label_batch = tf.train.shuffle_batch([images,labels], batch_size = 128, capacity = 20000, min_after_dequeue = 10000, num_threads = 1)

with tf.Session() as sess:
	
	threads = tf.train.start_queue_runners(sess=sess)
	batch_x, l = sess.run([image_batch,label_batch])
	batch_x = np.reshape(batch_x,newshape = [-1,96,96,1])
	for i in range(128):
		img = np.reshape(np.repeat(batch_x[i][:,:,np.newaxis],3,axis = 2), newshape = (96,96,3))
		print('save images')
		scipy.misc.imsave(str(i)+'.jpg', img)

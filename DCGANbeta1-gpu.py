from __future__ import division, print_function, absolute_import

import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PIL.Image as Image
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm 

#Training Params
num_step = 1
batch_size = 32

#Network Params
image_size = 9216
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200 #Noise data point

log_dir = "mnist_logs"

#从tfrecords读数据
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
	
#常数偏置
def bias(name, shape, bias_start = 0.0, trainable = True):
	dtype = tf.float32
	var = tf.get_variable(name, shape, tf.float32, trainable = trainable, initializer = tf.constant_initializer(bias_start, dtype = dtype))
	return var

#随机权重
def weight(name, shape, stddev = 0.02, trainable = True):
	dtype = tf.float32
	var = tf.get_variable(name, shape, tf.float32, trainable = trainable, initializer = tf.random_normal_initializer(stddev = stddev, dtype = dtype))
	return var
	
#全连接层
def fully_connected(value, output_shape, name = 'fully_connected', with_w = False):
	shape = value.get_shape().as_list()
	with tf.variable_scope(name):
		weights = weight('weights', [shape[1], output_shape], 0.02)
		biases = bias('biases', [output_shape], 0.0)
	if with_w:
		return tf.matmul(value, weights) + biases, weights, biases
	else:
		return tf.matmul(value, weights) + biases

# Leaky-ReLu层
def lrelu(x, leak = 0.2, name = 'lrelu'):
	with tf.variable_scopr(name):
		return tf.maximum(x, leak * x, name = name)

# ReLu层
def relu(value, name = 'relu'):
	with tf.variable_scope(name):
		return tf.nn.relu(value)

#解卷积层		
def deconv2d(value, output_shape, k_h = 5, k_w = 5, strides = [1,2,2,1], name = 'deconv2d', with_w = False):
	with tf.variable_scope(name):
		weights = weight('weights', [k_h, k_w, output_shape[-1], value.get_shape()[-1]])
		deconv = tf.nn.conv2d_transpose(value,weights,output_shape,strides = strides)
		biases = bias('biases', [output_shape[-1]])
		deconv = tf.reshape(tf.nn.bias_add(deconv,biases), deconv.get_shape())
		if with_w:
			return deconv, weights, biases
		else:
			return deconv
		
#卷积层
def conv2d(value, output_dim, k_h = 5, k_w = 5, strides = [1,2,2,1], name = 'conv2d'):
	with tf.variable_scope(name):
		weights = weight('weight', k_h, k_w, value.get_shape()[-1], output_dim)
		conv = tf.nn.conv2d(value, weights, strides = strides, padding = 'SAME')
		biases = bias('bias', [output_dim])
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
		return conv

#Batch Normalization 层
def batch_norm_layer(value, is_train = True, name = 'batch_norm'):
	with tf.variable_scope(name) as scope:
		if is_train:
			return batch_norm(value, decay = 0.9, epsilon = 1e-5, scale = True, is_training = is_train, updates_collections = None, scope = scope)
		else:
			return batch_norm(value, decay = 0.9, epsilon = 1e-5, scale = True, is_training = is_train, reuse = True, updates_collections = None, scope = scope)

#Generator Network
#Input: Noise Output:Image
def generator(x,reuse = False,train = True):
	with tf.variable_scope('Generator',reuse = reuse):
		h1 = tf.nn.relu(batch_norm_layer(fully_connected(x, 1024, 'g_fully_connected1'), is_train = train, name = 'g_bn1'))
		h2 = tf.nn.relu(batch_norm_layer(fully_connected(x, 128 * 24 * 24, 'g_fully_connected2'), is_train = train, name = 'g_bn2'))
		h2 = tf.reshape(h2, [32,24,24,128], name = 'h2_reshape')
		h3 = tf.nn.relu(batch_norm_layer(deconv2d(h2, [32,48,48,128],name = 'g_deconv2d3'), is_train = train, name = 'g_bn3'))
		h4 = tf.nn.sigmoid(deconv2d(h3, [32,96,96,1], name = 'g_deconv2d4'), name = 'generate_image')
		print(h4.get_shape())
	return h4
	
		
#Discriminator Network
#Input: Image, Output: Prediction Real/Fake Image
def discriminator(x,reuse = False):
	with tf.variable_scope ('Discriminator', reuse = reuse):
		x = tf.layers.conv2d(x,8,5)
		x = tf.nn.tanh(x)	
		x = tf.layers.average_pooling2d(x,2,2)
		x = tf.layers.conv2d(x,16,5)	
		x = tf.nn.tanh(x)
		x = tf.layers.average_pooling2d(x,2,2)	
		x = tf.contrib.layers.flatten(x)	
		x = tf.layers.dense(x, 3528)	
		x = tf.nn.tanh(x)
		#Output Real/Fake
		x = tf.layers.dense(x,2)
	return x
	

# 载入数据集
images, labels = read_and_decode('/Users/glh/Desktop/甲骨文收集/tfrecord/train.tfrecords')
image_batch, label_batch = tf.train.shuffle_batch([images,labels], batch_size = 32, capacity = 20000, min_after_dequeue = 10000, num_threads = 1)

#Build Networks
#Network Inputs
noise_input = tf.placeholder(tf.float32, shape = [None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape = [None,96,96,1])

#Build Generator Network
gen_sample = generator(noise_input)

#Build 2 Discriminator Networks(one from noise input, one from generated samples)
disc_real = discriminator(real_image_input)
#Build the stacked generator/discriminator
stacked_gan = discriminator(gen_sample, reuse = True)
disc_concat = tf.concat([disc_real,stacked_gan], axis = 0)
#Build Targets
disc_target = tf.placeholder(tf.int32,shape = [None])
gen_target = tf.placeholder(tf.int32,shape = [None])
real_target = tf.placeholder(tf.int32,shape = [None])
fake_target = tf.placeholder(tf.int32,shape = [None])

#Build Loss
disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = disc_concat, labels = disc_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = stacked_gan, labels = gen_target))

fake_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = stacked_gan, labels = fake_target))
real_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = disc_real, labels = real_target))
#tf.summary.scalar('loss_fake',fake_loss)
# tf.summary.scalar('loss_real',real_loss)

#Build Optimizer
optimizer_gen = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5)
optimizer_disc = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
#Disciminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator')

#Create training operations
train_gen = optimizer_gen.minimize(gen_loss,var_list = gen_vars)
train_disc = optimizer_disc.minimize(disc_loss,var_list = disc_vars)

#Initialize the variables
init = tf.global_variables_initializer()

#GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存  
config.gpu_options.allow_growth = True      #程序按需申请内存 

#Start training
with tf.Session() as sess:
	
	sess.run(init)
	threads = tf.train.start_queue_runners(sess=sess)
	#train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
	
	#断点继续训练
	saver = tf.train.Saver(max_to_keep = 1)
	ckpt = tf.train.get_checkpoint_state('/Users/glh/Desktop/甲骨文收集/代码/mnist_logs/model/')
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		
	
	for epoch in range(num_step):
		batch_idx = 416
		for i in range(batch_idx):
			batch_x, l = sess.run([image_batch,label_batch])
			batch_x = np.reshape(batch_x,newshape = [-1,96,96,1])
			z = np.random.normal(loc = 0, scale = 0.333, size = [batch_size,noise_dim])
			batch_disc_y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)], axis = 0)
			batch_disc_real_y = np.ones([batch_size])
			batch_disc_fake_y = np.zeros([batch_size])
			batch_gen_y = np.ones([batch_size])
			
			_,dl,fl,rl = sess.run([train_disc,disc_loss,fake_loss, real_loss], feed_dict = {real_image_input : batch_x, noise_input : z, disc_target : batch_disc_y, gen_target : batch_gen_y, fake_target: batch_disc_fake_y, real_target: batch_disc_real_y})
			
			for j in range(4):
				_, gl = sess.run([train_gen, gen_loss], feed_dict = {noise_input : z, gen_target : batch_gen_y})
			
			if i % 50 == 0:
				print('epoch %i, step %i: Fake Loss: %f, Real Loss: %f, Generator Loss: %f' % (epoch, i, fl,rl,gl))
				#train_writer.add_summary(result,i)
				
	saver.save(sess, '/Users/glh/Desktop/甲骨文收集/代码/mnist_logs/model/gan.ckpt', global_step = epoch)	
	#f,a = plt.subplots(2,32)
	for i in range(32):
		z = np.random.uniform(-1.,1.,size = [batch_size, noise_dim])
		g = sess.run(gen_sample, feed_dict = {noise_input : z})
		print('g.size:')
		fig_count = 0
		for j in range(1):
			img = np.reshape(np.repeat(g[j][:,:,np.newaxis],3,axis = 2), newshape = (96,96,3))
			#a[j][i].imshow(img)
			print('save images')
			scipy.misc.imsave(str(i)+'.jpg', img)
			#scipy.misc.imsave('restmp.jpg', img)
	#f.show()
	#plt.draw()
	#plt.waitforbuttonpress()
	
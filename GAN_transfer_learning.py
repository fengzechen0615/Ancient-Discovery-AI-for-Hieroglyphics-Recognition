import os
import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm 

#训练参数
BATCH_SIZE = 64
OUTPUT_SIZE = 96
GF = 64				#Dimension of G filters in first conv layer
DF = 64				#Dimension of D filters in first conv layer
Z_DIM = 200
IMAGE_CHANNEL = 3
LR = 0.0002			#Learning rate
EPOCH = 100
LOAD_MODEL = False
TRAIN = True
CURRENT_DIR = os.getcwd()
	
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
	with tf.variable_scope(name):
		return tf.maximum(x, leak * x, name = name)

# ReLu层
def relu(value, name = 'relu'):
	with tf.variable_scope(name):
		return tf.nn.relu(value)

#解卷积层		
def deconv2d(value, output_shape, k_h = 3, k_w = 3, strides = [1,2,2,1], name = 'deconv2d', with_w = False):
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
def conv2d(value, output_dim, k_h = 3, k_w = 3, strides = [1,2,2,1], name = 'conv2d'):
	with tf.variable_scope(name):
		weights = weight('weight', [k_h, k_w, value.get_shape()[-1], output_dim])
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
			
			
#生成器
def generator(z,reuse = False,is_train = True, name = 'Generator'):
	with tf.variable_scope(name,reuse = reuse):
		s2, s4, s8, s16 = 48, 24, 12, 6
		
		#第一层全连接
		h1 = tf.reshape(fully_connected(z, GF * 8 * s16 * s16, 'g_fc1'), [-1, s16, s16, GF * 8], name = 'reshape')
		h1 = relu(batch_norm_layer(h1, name = 'g_bn1', is_train = is_train))
		
		#第二层解卷积 12 * 12 * (64 * 4)
		h2 = deconv2d(h1, [BATCH_SIZE, s8, s8, GF * 4], name = 'g_deconv2d1')
		h2 = relu(batch_norm_layer(h2, name = 'g_bn2', is_train = is_train))
		
		#第三层解卷积 24 * 24 * (64 * 2)
		h3 = deconv2d(h2, [BATCH_SIZE, s4, s4, GF * 2], name = 'g_deconv2d2')
		h3 = relu(batch_norm_layer(h3, name = 'g_bn3', is_train = is_train))
		
		#第四层解卷积 48 * 48 * (64 * 1)
		h4 = deconv2d(h3, [BATCH_SIZE, s2, s2, GF * 1], name = 'g_deconv2d3')
		h4 = relu(batch_norm_layer(h4, name = 'g_bn4', is_train = is_train))
		
		#第五层解卷积 96 * 96 * 3 (三通道)
		h5 = deconv2d(h4, [BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 3], name = 'g_deconv2d4')
		
	return tf.nn.tanh(h5)
		
#判别器
def discriminator(image,reuse = False):
	with tf.variable_scope ('Discriminator', reuse = reuse):
		if reuse:
			tf.get_variable_scope().reuse_variables()
			
		#卷积
		h0 = lrelu(conv2d(image, DF, name = 'd_h0_conv'), name = 'd_h0_lrelu')
		h1 = lrelu(batch_norm_layer(conv2d(h0, DF*2, name='d_h1_conv'), name = 'd_h1_bn'), name = 'd_h1_lrelu')
		h2 = lrelu(batch_norm_layer(conv2d(h1, DF*4, name='d_h2_conv'), name = 'd_h2_bn'), name = 'd_h2_lrelu')
		h3 = lrelu(batch_norm_layer(conv2d(h2, DF*8, name='d_h3_conv'), name = 'd_h3_bn'), name = 'd_h3_lrelu')
		h4 = fully_connected(tf.reshape(h3, [BATCH_SIZE, -1]), 1, 'd_h4_fc')
		
		return tf.nn.sigmoid(h4), h4
			
def read_and_decode(filename):
	
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename)
	
	features = tf.parse_single_example(serialized_example,features = {
						'image_raw':tf.FixedLenFeature([], tf.string)})
	image = tf.decode_raw(features['image_raw'], tf.uint8)
	
	image = tf.reshape(image, [OUTPUT_SIZE, OUTPUT_SIZE, 3])
	image = tf.cast(image, tf.float32)
	image = image / 255.0
	
	return image

def save_images(images, size, path):
	img = (images + 1.0) / 2.0
	h, w = img.shape[1], img.shape[2]
	merge_img = np.zeros((h * size[0], w * size[1], 3))
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		merge_img[j*h:j*h+h, i*w:i*w+w, :] = image
		
	return scipy.misc.imsave(path, merge_img)    
	

def train():
	
	data_dir = CURRENT_DIR + '/tfrecord/train.tfrecords'
	train_dir = CURRENT_DIR + '/logs/'
	
	# 载入数据集
	filename_queue = tf.train.string_input_producer([data_dir])
	image = read_and_decode(filename_queue)
	images = tf.train.shuffle_batch([image], batch_size = BATCH_SIZE, num_threads = 4, capacity = 20000 + 3 * BATCH_SIZE, min_after_dequeue = 20000)
	
	global_step = tf.Variable(0, name = 'global_step', trainable = False)
	
	z = tf.placeholder(tf.float32, [None, Z_DIM], name = 'z')
	samplez = tf.placeholder(tf.float32, [None, Z_DIM], name = 's_z')
	
	G = generator(z)
	D, D_logits = discriminator(images)
	samples = generator(samplez,reuse=True)
	D_, D_logits_ = discriminator(G, reuse = True)
	
	d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logits, labels = tf.ones_like(D)))
	d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logits_, labels = tf.zeros_like(D_)))
	d_loss = d_loss_real + d_loss_fake
	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logits_, labels = tf.ones_like(D_)))
	
	z_sum = tf.summary.histogram('z', z)
	d_sum = tf.summary.histogram('d', D)
	d__sum = tf.summary.histogram('d_', D_)
	G_sum = tf.summary.histogram('G', G)

	d_loss_real_sum = tf.summary.scalar('d_loss_real', d_loss_real)
	d_loss_fake_sum = tf.summary.scalar('d_loss_fake', d_loss_fake)
	d_loss_sum = tf.summary.scalar('d_loss', d_loss)                                                
	g_loss_sum = tf.summary.scalar('g_loss', g_loss)
		
	g_sum = tf.summary.merge([z_sum, d__sum, G_sum, d_loss_fake_sum, g_loss_sum])
	d_sum = tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])
	
	t_vars = tf.trainable_variables()
	d_vars = [var for var in t_vars if 'd_' in var.name]
	g_vars = [var for var in t_vars if 'g_' in var.name]
	
	saver = tf.train.Saver()
	d_optim = tf.train.AdamOptimizer(LR, beta1 = 0.5).minimize(d_loss, var_list = d_vars, global_step = global_step)
	g_optim = tf.train.AdamOptimizer(LR, beta1 = 0.5).minimize(g_loss, var_list = g_vars, global_step = global_step)
	
	with tf.Session() as sess:
		writer = tf.summary.FileWriter(train_dir, sess.graph)
		
		sample_z = np.random.uniform(-1, 1, size = [BATCH_SIZE, Z_DIM])
		
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		init = tf.initialize_all_variables()  
		sess.run(init)

		start = 0
		if LOAD_MODEL:
			ckpt = tf.train.get_checkpoint_state(train_dir)
			
			if ckpt and ckpt.model_checkpoint_path:
				ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
				saver.restore(sess, os.path.join(train_dir, ckpt_name))
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
			print('loading success')	
			start = int(global_step)
		
		for epoch in range(EPOCH):
			batch_idxs = 400
			
			for idx in range(batch_idxs):
				batch_z = np.random.uniform(-1, 1, size = (BATCH_SIZE, Z_DIM))
				
				#判别器更新
				_, summary_str = sess.run([d_optim, d_sum], feed_dict = {z:batch_z})
				writer.add_summary(summary_str, idx+1)
				
				#生成器更新两次
				_, summary_str = sess.run([g_optim, g_sum], feed_dict = {z:batch_z})
				writer.add_summary(summary_str, idx+1)
				
				_, summary_str = sess.run([g_optim, g_sum], feed_dict = {z:batch_z})
				writer.add_summary(summary_str, idx+1)
				
				errD_fake = d_loss_fake.eval({z: batch_z})
				errD_real = d_loss_real.eval()
				errG = g_loss.eval({z: batch_z})
				
				if idx % 50 == 0:
					print("[%4d/%4d] d_loss: %.8f, g_loss: %.8f" % (idx, batch_idxs, errD_fake+errD_real, errG))
				
				if idx % 100 == 0:
					sample = sess.run(samples, feed_dict = {samplez:sample_z})
					samples_path = CURRENT_DIR + '/samples/'
					save_images(sample, [8,8], samples_path + 'sample_%d_epoch_%d.png' % (epoch, idx))
					
					print('\n')
					print('===========    %d_epoch_%d.png save down    ===========' %(epoch, idx))
					print('\n')
				
			checkpoint_path = os.path.join(train_dir, 'my_dcgan_tfrecords.ckpt')
			saver.save(sess, checkpoint_path, global_step = idx + 1)
			print('********* model saved *********')
			
		coord.request_stop()
		coord.join(threads)
		sess.close()
	
train()
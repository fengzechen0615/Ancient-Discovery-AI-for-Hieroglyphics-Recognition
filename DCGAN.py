from __future__ import division, print_function, absolute_import

import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PIL.Image as Image

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
	
#Generator Network
#Input: Noise Output:Image
def generator(x,reuse = False):
	with tf.variable_scope('Generator',reuse = reuse):
		#TensorFlow Layers automatically create variables and calculate their
		#shape, based on the input_data
		x = tf.layers.dense(x,units = 23 * 23 * 64) #全连接层 输出维度为23*23*64
		x = tf.nn.tanh(x)
		#Reshape to a 4-D array of images:(batch,height,width,channels)
		#New shape:(batch,23,23,128)
		x = tf.reshape(x,[-1,23,23,64])
		#Deconvolution,image shape:(batch,48,48,64)
		x = tf.layers.conv2d_transpose(x,8,4,strides = 2)
		#Deconvolution,image shape:(batch,96,96,1)
		x = tf.layers.conv2d_transpose(x,1,2,strides = 2)
		#Apple sigmoid to clip values between 0 and 1
		x = tf.nn.sigmoid(x)
	return x
		
#Discriminator Network
#Input: Image, Output: Prediction Real/Fake Image
def discriminator(x,reuse = False):
	with tf.variable_scope ('Discriminator', reuse = reuse):
		#Typical convolutional neural network to classify images
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

#Training Params
num_step = 80
batch_size = 32

#Network Params
image_size = 9216
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200 #Noise data point

log_dir = "mnist_logs"

#Build Networks
#Network Inputs
noise_input = tf.placeholder(tf.float32, shape = [None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape = [None,96,96,1])

#Build Generator Network
gen_sample = generator(noise_input) #shape:[None,28,28,1]

#Build 2 Discriminator Networks(one from noise input, one from generated samples)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse = True)
disc_concat = tf.concat([disc_real,disc_fake], axis = 0)

#Build the stacked generator/discriminator
stacked_gan = discriminator(gen_sample, reuse = True)

#Build Targets
disc_target = tf.placeholder(tf.int32,shape = [None])
gen_target = tf.placeholder(tf.int32,shape = [None])
real_target = tf.placeholder(tf.int32,shape = [None])
fake_target = tf.placeholder(tf.int32,shape = [None])

#Build Loss
disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = disc_concat, labels = disc_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = stacked_gan, labels = gen_target))

fake_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = disc_fake, labels = fake_target))
real_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = disc_real, labels = real_target))
tf.summary.scalar('loss_fake',fake_loss)
tf.summary.scalar('loss_real',real_loss)

#Build Optimizer
optimizer_gen = tf.train.AdamOptimizer(learning_rate = 0.0001)
optimizer_disc = tf.train.AdamOptimizer(learning_rate = 0.0001)

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
#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()

#Start training
cpu_config = tf.ConfigProto(intra_op_parallelism_threads = 4, inter_op_parallelism_threads = 4, device_count = {'CPU': 4})
with tf.Session(config  = cpu_config) as sess:
	
	sess.run(init)
	merged = tf.summary.merge_all()
	threads = tf.train.start_queue_runners(sess=sess)
	
	#断点继续训练
	saver = tf.train.Saver(max_to_keep = 1)
	ckpt = tf.train.get_checkpoint_state('/Users/glh/Desktop/甲骨文收集/代码/mnist_logs/model/')
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		
	train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
	
	for epoch in range(num_step):
		batch_idx = 416
		for i in range(batch_idx):
			batch_x, l = sess.run([image_batch,label_batch])
			batch_x = np.reshape(batch_x,newshape = [-1,96,96,1])
			z = np.random.uniform(-1.,1.,size = [batch_size, noise_dim])
			batch_disc_y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)], axis = 0)
			batch_disc_real_y = np.ones([batch_size])
			batch_disc_fake_y = np.zeros([batch_size])
			batch_gen_y = np.ones([batch_size])
			
			feed_dict = {real_image_input : batch_x, noise_input : z, disc_target : batch_disc_y, gen_target : batch_gen_y, fake_target: batch_disc_fake_y, real_target: batch_disc_real_y}
			_,_,gl,dl,fl,rl = sess.run([train_gen,train_disc,gen_loss,disc_loss,fake_loss, real_loss], feed_dict = feed_dict)
			result = sess.run(merged, feed_dict = feed_dict)
			if i % 50 == 0:
				print('step %i: Fake Loss: %f, Real Loss: %f, Loss: %f' % (i, fl, rl, dl))
				train_writer.add_summary(result,i)
	
	saver.save(sess, '/Users/glh/Desktop/甲骨文收集/代码/mnist_logs/model/gan.ckpt', global_step = epoch)	
	f,a = plt.subplots(4,10)
	for i in range(10):
		z = np.random.uniform(-1.,1.,size = [4,noise_dim])
		g = sess.run(gen_sample, feed_dict = {noise_input : z})
		print('g.size:')
		fig_count = 0
		for j in range(4):
			img = np.reshape(np.repeat(g[j][:,:,np.newaxis],3,axis = 2), newshape = (96,96,3))
			a[j][i].imshow(img)
			print('save images')
			scipy.misc.imsave(str(i)+str(j)+'.jpg', img)
			#scipy.misc.imsave('restmp.jpg', img)
	f.show()
	plt.draw()
	plt.waitforbuttonpress()
	
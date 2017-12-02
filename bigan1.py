import tensorflow as tf 
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

x_dim = 784
latent_dim = 64


def discriminator(x, latent, reuse=False):
	dis_input = tf.concat((x, latent), 1) 
	with tf.variable_scope('discriminator', reuse=reuse):
		dis1 = slim.fully_connected(dis_input, 128, activation_fn=tf.nn.relu, reuse=reuse, scope='dis1')
		dis_output = slim.fully_connected(dis1, 1, activation_fn=tf.nn.sigmoid, reuse=reuse, scope='dis_output')
	return dis_output

def encoder(x, reuse=False):
	with tf.variable_scope('encoder', reuse=reuse):
		enc1 = slim.fully_connected(x, 128, activation_fn=tf.nn.relu, reuse=reuse, scope='enc1')
		enc_output = slim.fully_connected(enc1, latent_dim, activation_fn=None, reuse=reuse, scope='enc_output')
	return enc_output
	
def generator(latent, reuse=False):
	with tf.variable_scope('generator', reuse=reuse):
		gen1 = slim.fully_connected(latent, 128, activation_fn=tf.nn.relu, reuse=reuse, scope='gen1')
		gen_output = slim.fully_connected(gen1, x_dim, activation_fn=tf.nn.sigmoid, reuse=reuse, scope='gen_output')
	return gen_output

if __name__=="__main__":
	data = input_data.read_dataset('../DL/data')

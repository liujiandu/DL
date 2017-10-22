from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("data", one_hot = True)

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x,w):
    return tf.nn.conv2d(x,w, strides=[1,1,1,1], padding="SAME")

def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides=[1,2,2,1], padding="SAME")

#network
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28,28,1])
y_ = tf.placeholder(tf.float32, [None, 10])

w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
h_pool1 = max_pool_2_2(h_conv1)


w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
h_pool2 = max_pool_2_2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = weight_variable([1024])
h_fc1 = tf.relu(tf.matmul(h_pool2_falt, w_fc1)_b_fc1)

w_fc2 = weight_variable([1024, 10])
b_fc2 = weight_variable([10])
y = tf.softmax(tf.matmul(h_fc1, w_fc2)+b_fc2)

cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_*tf.log(y_), reduction-dices=[1]))

#train
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correnct_prediction, tf.float32))

#initialize
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(10):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict = ({x: batch_xs, y_:batch_ys}))
    print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))


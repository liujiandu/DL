from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#data
mnist = input_data.read_data_sets("data", one_hot=True)

#network
input_neuron_num = 784
hidden1_neuron_num = 300
output_neuron_num = 10
w1 = tf.Variable(tf.truncated_normal([input_neuron_num, hidden1_neuron_num],stddev = 0.1))
b1 = tf.Variable(tf.zeros([hidden1_neuron_num]))
w2 = tf.Variable(tf.truncated_normal([hidden1_neuron_num, output_neuron_num],stddev=0.1))
b2 = tf.Variable(tf.zeros([output_neuron_num]))

x = tf.placeholder(tf.float32, [None, input_neuron_num])
hidden1 = tf.nn.relu(tf.matmul(x,w1)+b1)
keep_prob = tf.placeholder(tf.float32)
hidden1_drop  = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)
y_ = tf.placeholder(tf.float32,[None, output_neuron_num])
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_*tf.log(y),reduction_indices = [1]))


#train
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

#evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#initialize
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict=({x:batch_xs, y_:batch_ys, keep_prob:0.75}))
    print(sess.run(accuracy, feed_dict=({x:mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0})))


from tensorflow.examples.tutorials.mnist import input_data
mnist  = input_data.read_data_sets("data", one_hot=True)


import tensorflow as tf

#session
sess = tf.InteractiveSession()

#network
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(tf.float32, [None,10])

cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_*tf.log(y), reduction_indices = [1]))

#initial
#tf.global_variables_initializer().run()  #new version
tf.initialize_all_variables().run()  #new version


#train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

#evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))




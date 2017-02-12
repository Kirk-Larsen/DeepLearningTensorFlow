import tensorflow as tf

#import mnist images of numbers 0-9 (drawn by hand)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#set up a linear classifier
x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,W) + b

#compute softmax loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))

#minimize cross entropy via gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initialize and run session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#train classifier
for i in range(1000):
	batch = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict = {x: batch[0], y_: batch[1]})

#display linear classifier performance
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))
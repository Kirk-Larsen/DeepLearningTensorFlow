import tensorflow as tf

#import mnist images of numbers 0-9 (drawn by hand)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#define 3-layer neural network function, using rectified linear unit non-linearity
def neuralNet(x, W1, W2, W3, b1, b2, b3, p_keep_in, p_keep_out):
	x = tf.nn.dropout(x, p_keep_in)
	l1 = tf.nn.relu(tf.matmul(x, W1) + b1)

	l1 = tf.nn.dropout(l1, p_keep_out)
	l2 = tf.nn.relu(tf.matmul(l1,W2) + b2)

	l2 = tf.nn.dropout(l2, p_keep_out)
	return tf.matmul(l2, W3) + b3

#initialize placeholders and variables for data and matrix weights + biases + dropout parameters
x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])
W1 = tf.Variable(tf.random_normal(shape = [784, 600], stddev = 0.01))
W2 = tf.Variable(tf.random_normal(shape = [600,600], stddev = 0.01))
W3 = tf.Variable(tf.random_normal(shape = [600,10], stddev = 0.01))
b1 = tf.Variable(tf.random_normal(shape = [600], stddev = 0.01))
b2 = tf.Variable(tf.random_normal(shape = [600], stddev = 0.01))
b3 = tf.Variable(tf.random_normal(shape = [10], stddev = 0.01))
p_keep_in = tf.placeholder("float")
p_keep_out = tf.placeholder("float")

activate = neuralNet(x, W1, W2, W3, b1, b2, b3, p_keep_in, p_keep_out)

#compute softmax loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = activate, labels = y))

#minimize loss function via gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#initialize and run session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
		batch = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict = {x: batch[0], y: batch[1], p_keep_in: 0.8, p_keep_out: 0.5})
	#display neural network performance
	correct_prediction = tf.equal(tf.argmax(activate, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels, p_keep_in: 0.8, p_keep_out: 0.5}))
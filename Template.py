import tensorflow as tf
import data
import numpy as np

training = data.Data(mode="img", batch_size=2)
inputs = tf.placeholder(tf.int32, (batch_size, max_length))
one_hot = tf.one_hot(inputs, vocabsize)  # batch x length x vocab
outputs = tf.placeholder(tf.int32, (batch_size, max_length))

W = tf.get_variable(name="W", shape=[vocab, 1])
one_hot = tf.reshape(one_hot, [batch_size*max_length, -1]) # batch*L x V
logits = tf.matmul(inputs, W)   # batch*Length x 1
outputs = tf.reshape(outputs, [batch_size*max_length, 1])
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, outputs))
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

sess = tf.Session()

inp, out = training.get_batch()
feed_dict = {model.inputs: inp, model.outputs: out}
loss, _ = sess.run([loss, train_op], feed_dict)

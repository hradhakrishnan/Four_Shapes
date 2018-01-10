# coding: utf-8

# In[22]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import time


# In[23]:

beginTime = time.time()


# In[24]:

#------------------------------------------------------------
# Variables
#------------------------------------------------------------
# Parameter definitions
batch_size = 1
learning_rate = 0.01
max_steps = 100

#Image dimensions
img_height=100
img_width=100

# File TF REcords
filenames = ['all.tfrecords']
tfrecords_validation = ['validation.tfrecords']
# Object Classifications as Array
classes = ['circle','square','star','triangle']


# In[25]:

#---------------------------------
# TENSORFLOW GRAPH DEFINITION
#---------------------------------
# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None,10000])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# Define variables (these are the values we want to optimize)
weights = tf.Variable(tf.zeros([10000, 4]))
biases = tf.Variable(tf.zeros([4]))
tf.summary.histogram("weights", weights)
tf.summary.histogram("biases", biases)
print(weights)
# Define the classifier's result
logits = tf.matmul(images_placeholder, weights) + biases
print(logits)
# Define the loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
  labels=labels_placeholder))
print(loss)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
print(train_step)
# Operation comparing prediction with true label
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

# Operation calculating the accuracy of our predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#---------------------------------
# END OF GRAPH DEFINITION
#---------------------------------

# Operation merging summary data for TensorBoard
#summary = tf.summary.merge_all()

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())


# In[26]:

#------------------------------------------------------------
# Function to parse features for each object in the tfrecord
#------------------------------------------------------------

def _parse_record(tf_record):
    print('Parse Fn tf_record Called')
    features={
             'height': tf.FixedLenFeature([], tf.int64),
             'width': tf.FixedLenFeature([], tf.int64),
             'image_raw': tf.FixedLenFeature([], tf.string),
             'label': tf.FixedLenFeature([], tf.int64)
             }
    record = tf.parse_single_example(tf_record, features)

    height = tf.cast(record['height'], tf.int32)
    width = tf.cast(record['width'], tf.int32)

    image_raw = tf.decode_raw(record['image_raw'], tf.uint8)
    print(image_raw.shape)
    image_decoded = tf.reshape(image_raw, [-1,100,100])
    image_summary = tf.reshape(image_raw, [-1,100,100,1])
    tf.summary.image('input', image_summary, 10)
    image_decoded = tf.reshape(image_decoded,[10000])
    print(image_decoded.shape)

    label = tf.cast(record['label'], tf.int32)
    return image_decoded, label

#------------------------------------------------------------
# end of Parse function
#------------------------------------------------------------


# In[ ]:

#------------------------------------------------------------
# TFRECORD Dataset Iterator
#------------------------------------------------------------
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_record)
dataset = dataset.batch(batch_size)
dataset = dataset.shuffle(buffer_size=10000)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

#------------------------------------------------------------
# TFRECORD Validation Dataset Iterator
#------------------------------------------------------------
validation_dataset = tf.data.TFRecordDataset(tfrecords_validation)
validation_dataset = validation_dataset.map(_parse_record)
validation_dataset = validation_dataset.batch(batch_size)
validation_iterator = validation_dataset.make_initializable_iterator()
next_element_val = validation_iterator.get_next()


# In[ ]:

saver = tf.train.Saver()

with tf.Session() as sess:

    summary_writer = tf.summary.FileWriter("/../shapes/log",graph=tf.get_default_graph())
    # Loop over each example in batch
    # Compute for 100 epochs.
    sess.run(init_op)

    for i in range(100):

        sess.run(iterator.initializer)
        while True:
            try:
                images_batch, labels_batch = sess.run(next_element)
                print(images_batch.shape)
                print(labels_batch)
                feed_dict = {
                  images_placeholder: images_batch,
                  labels_placeholder: labels_batch
                            }
                # Periodically print out the model's current accuracy
                #if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
                print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

                #Training STEP
                sess.run(train_step, feed_dict=feed_dict)
            except tf.errors.OutOfRangeError:
                  break

        sess.run(validation_iterator.initializer)
        img_val, label_val = sess.run(next_element_val)
        feed_dict_val = {
                  images_placeholder: img_val,
                  labels_placeholder: label_val
                            }
        #for i in range(img_val.shape[0]):
        # After finishing the training, evaluate on the test set
        test_accuracy = sess.run(accuracy, feed_dict={images_placeholder: img_val,labels_placeholder: label_val})
        print('Test accuracy {:g}'.format(test_accuracy))


    save_path = saver.save(sess, "shapes.ckpt")
    print("Model saved in file: %s" % save_path)

    summary_writer.close()

    sess.close()

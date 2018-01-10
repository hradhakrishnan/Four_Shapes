
# coding: utf-8

# In[ ]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import sys
import tensorflow as tf
from PIL import Image,ImageFilter
import skimage.io as io
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import numpy as np
import time

#example execution: python predict.py /../data/circle/3.png

# In[ ]:

beginTime = time.time()


# In[ ]:
# Parameter definitions
batch_size = 10
learning_rate = 0.05
max_steps = 100

img_height=100
img_width=100
# Object Classifications as Array
classes = ['circle','square','star','triangle']

# In[103]:

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def imagereshape(argv):
    if os.path.exists(argv):
        img = Image.open(argv)
        img = img.resize((img_width, img_height), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        img.show()
        shape_img = np.array(img)
        shape_img = shape_img.flatten()
        return [shape_img]


# In[ ]:
def predictclass(imgvalue):

    image_raw = imgvalue
    images_placeholder = tf.placeholder(tf.float32, shape=[None,10000])
    labels_placeholder = tf.placeholder(tf.int64, shape=[None])

    weights = tf.Variable(tf.zeros([10000, 4]))
    biases = tf.Variable(tf.zeros([4]))
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", biases)

    logits = tf.matmul(images_placeholder, weights) + biases
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels_placeholder))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    saver = tf.train.Saver()



    with tf.Session() as sess:
        sess.run(init_op)

        saver.restore(sess, "shapes.ckpt")

        for i in range(1):
            feed_dict = {images_placeholder: image_raw}
            print('printing image')
            print(feed_dict)


            #print ("Model restored.")
            print("prediction happening here")
            prediction=tf.argmax(logits,1)
            return prediction.eval(feed_dict=feed_dict, session=sess)

# In[ ]:

def main(argv):
    """
    Main function.
    """
    imgvalue = imagereshape(argv)
    prediction = predictclass(imgvalue)
    print(prediction)
    print("-------------------------------------")
    print("MODELS PREDICTION")
    print("-------------------------------------")
    print(classes[prediction[0]])
    print("-------------------------------------")


if __name__ == "__main__":
    main(sys.argv[1])


# In[ ]:

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))

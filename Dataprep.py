
# coding: utf-8

# In[22]:

import os
import csv
import tensorflow as tf
import numpy as np
from PIL import Image
import skimage.io as io
from skimage.transform import rescale, resize, downscale_local_mean



# In[23]:

source_folderpath='/../data/triangle'
target_folderpath='/../shapes/'
tfrecords_filename = '/../shapes/all.tfrecords'
img_height=100
img_width=100
label=3


# In[24]:

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# In[25]:

classes = ['circle','square','star','triangle']
num_classes = len(classes)
print(num_classes)


# In[26]:

print(classes[1])
print(classes.index('triangle'))


# In[27]:

#Convert images into tfrecords - Hari

writer = tf.python_io.TFRecordWriter(tfrecords_filename)
for subdir, dirs, files in os.walk(source_folderpath):
    for file in files:
        filepath = subdir + os.sep + file
        print (filepath)
        img = Image.open(filepath)
        img = img.resize((img_width, img_height), Image.ANTIALIAS)
        shape_img = np.array(img)
        img_raw = shape_img.tostring()
        height = shape_img.shape[0]
        width = shape_img.shape[1]

        example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(label),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'image_raw': _bytes_feature(img_raw)}))

        writer.write(example.SerializeToString())
        print("writing tfrecord" + tfrecords_filename)

    writer.close()


# In[28]:

reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
output='/../shapes/data/out.png'
for string_record in record_iterator:

    example = tf.train.Example()
    example.ParseFromString(string_record)

    height = int(example.features.feature['height']
                                 .int64_list
                                 .value[0])

    width = int(example.features.feature['width']
                                .int64_list
                                .value[0])

    img_string = (example.features.feature['image_raw']
                                  .bytes_list
                                  .value[0])

    label = int(example.features.feature['label']
                                .int64_list
                                .value[0])
    print(classes[label])

    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((-1, width, height ))
    print(reconstructed_img.shape)
    print(reconstructed_img.dtype)
    io.imsave(output,reconstructed_img[0])

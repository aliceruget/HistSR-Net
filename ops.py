import math
import numpy as np 
import tensorflow as tf
import random
from tensorflow.python.framework import ops

from utils import *

try:
  image_summary = tf.compat.v1.summary.image
  scalar_summary = tf.compat.v1.summary.scalar
  histogram_summary = tf.compat.v1.summary.histogram
  merge_summary = tf.compat.v1.summary.merge
  SummaryWriter = tf.compat.v1.summary.FileWriter
except:
  image_summary = tf.compat.v1.summary.image
  scalar_summary = tf.compat.v1.summary.scalar
  histogram_summary = tf.compat.v1.summary.histogram
  merge_summary = tf.compat.v1.summary.merge
  SummaryWriter = tf.compat.v1.summary.FileWriter

if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):

    with tf.compat.v1.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

# def conv2d(input_, input_dim,output_dim, 
#        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#        name="conv2d"):
#   #with tf.device("/job:ps/task:0/cpu:0"):
#     with tf.compat.v1.variable_scope(name):
    
#       w = tf.compat.v1.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
#               initializer=tf.truncated_normal_initializer(stddev=stddev))
#       biases = tf.compat.v1.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))

#       conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
#       conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

#     return conv,w, biases
def conv2d(input_, input_dim,output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.compat.v1.variable_scope(name):

    w = tf.compat.v1.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.compat.v1.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    #biases = biases + (-1+2*np.random.rand(output_dim))*5/100*biases
    #biases = biases - 5/100*biases
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv,w, biases

def conv2d_1st_conv(input_, input_dim,output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.compat.v1.variable_scope(name):

    w = tf.compat.v1.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
    conv_before_bias = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.compat.v1.get_variable('biases', [output_dim], initializer=tf.compat.v1.constant_initializer(0.0))
    #biases = biases + (-1+2*np.random.rand(output_dim))*5/100*biases
    #biases = biases - 5/100*biases
    conv = tf.reshape(tf.nn.bias_add(conv_before_bias, biases), conv_before_bias.get_shape())

    return conv,w, biases, conv_before_bias
    
def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.compat.v1.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.compat.v1.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.compat.v1.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.compat.v1.get_variable('biases', [output_shape[-1]], initializer=tf.compat.v1.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv, w, biases

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.compat.v1.variable_scope(scope or "Linear"):
    matrix = tf.compat.v1.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.compat.v1.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
def max_pool_2x2(x):
  return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.compat.v1.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.compat.v1.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def conv(inputs, kernel_size, output_num, stride_size=1, init_bias=0.0, conv_padding='SAME', stddev=0.01,
         activation_func=tf.nn.relu):
    input_size = inputs.get_shape().as_list()[-1]
    conv_weights = tf.Variable(
        tf.random_normal([kernel_size, kernel_size, input_size, output_num], dtype=tf.float32, stddev=stddev),
        name='weights')
    conv_biases = tf.Variable(tf.constant(init_bias, shape=[output_num], dtype=tf.float32), 'biases')
    conv_layer = tf.nn.conv2d(inputs, conv_weights, [1, stride_size, stride_size, 1], padding=conv_padding)
    conv_layer = tf.nn.bias_add(conv_layer, conv_biases)
    if activation_func:
        conv_layer = activation_func(conv_layer)
    return conv_layer

def fc(inputs, output_size, init_bias=0.0, activation_func=tf.nn.relu, stddev=0.01):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) == 4:
        fc_weights = tf.Variable(
            tf.random_normal([input_shape[1] * input_shape[2] * input_shape[3], output_size], dtype=tf.float32,
                             stddev=stddev),
            name='weights')
        inputs = tf.reshape(inputs, [-1, fc_weights.get_shape().as_list()[0]])
    else:
        fc_weights = tf.Variable(tf.random_normal([input_shape[-1], output_size], dtype=tf.float32, stddev=stddev),
                                 name='weights')

    fc_biases = tf.Variable(tf.constant(init_bias, shape=[output_size], dtype=tf.float32), name='biases')
    fc_layer = tf.matmul(inputs, fc_weights)
    fc_layer = tf.nn.bias_add(fc_layer, fc_biases)
    if activation_func:
        fc_layer = activation_func(fc_layer)
    return fc_layer


def lrn(inputs, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    return tf.nn.local_response_normalization(inputs, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=bias)



def center_of_mass(Histogram_training_depth_LR, type):
    if type =='dict':
        nb_images = len(Histogram_training_depth_LR)
        depth_LR = Histogram_training_depth_LR[0]
        Nx_LR = int(depth_LR.shape[0])
        Ny_LR = int(depth_LR.shape[1])
        Nbins = depth_LR.shape[2]
        Depth_images = {}
    elif type =='one_image':
        depth_LR = Histogram_training_depth_LR
        nb_images = 1
        Nx_LR = int(depth_LR.shape[0])
        Ny_LR = int(depth_LR.shape[1])
        Nbins = depth_LR.shape[2]
        Depth_images = np.zeros((nb_images, Nx_LR, Ny_LR))

    for index in range(nb_images):
        if type =='dict':
            depth_LR = Histogram_training_depth_LR[index]
        elif type =='one_image':
            depth_LR = Histogram_training_depth_LR
        depth_image = np.zeros((Nx_LR,Ny_LR))
        denominator = np.zeros((Nx_LR,Ny_LR))
        numerator = np.zeros((Nx_LR,Ny_LR))

        for i in range(Nx_LR):
            for j in range(Ny_LR):
                # Define maximum symmetric window (range_center_of_mass) around maximum (pos_max)
                pos_max = np.argmax(np.squeeze(depth_LR[i,j,:]))
                index_bin = 0
                while pos_max + index_bin < Nbins and pos_max - index_bin > 0 and index_bin < 2:  
                    index_bin = index_bin + 1

                if index_bin==0:
                    depth_image[i,j]= pos_max
                else:
                    range_center_of_mass = range(pos_max-index_bin , pos_max + index_bin, 1) 
                    #range_center_of_mass = range(pos_max - 2 , pos_max + 2, 1) 
                    
                    # Define b 
                    b = np.median(np.squeeze(depth_LR[i,j,:]))
                    for t in range_center_of_mass:     
                        numerator[i,j] = numerator[i,j] + t * np.maximum(depth_LR[i,j,t] - b, 0)
                        denominator[i,j] = denominator[i,j] + np.maximum(depth_LR[i,j,t] - b, 0)
                    if denominator[i,j] !=0:
                        depth_image[i,j] = numerator[i,j] / denominator[i,j]
                    else:
                        depth_image[i,j] = 0

                    
        #if np.mod(index, 100)==0:
        #    print(index)
        #    print("--- %s seconds ---" % (time.time() - start_time))
        depth_image = np.float32(depth_image)
        #max_d, min_d = np.max(depth_image), np.min(depth_image)
        #if max_d - min_d == 0:

        depth_image = depth_image / 15

        if type =='dict':
            Depth_images[index] = depth_image
        elif type == 'one_image':
            Depth_images[index, :,:] = depth_image
    return Depth_images


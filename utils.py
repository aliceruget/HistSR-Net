"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt
# import color 
from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
from skimage import io,data,color
import tensorflow as tf
import imageio

FLAGS = tf.compat.v1.flags.FLAGS

def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    depth_down = np.array(hf.get('depth_down'))
    depth_label = np.array(hf.get('depth_label'))
    I_add = np.array(hf.get('I_add'))
    return depth_down, depth_label , I_add

# def preprocess(path, scale=3):
#   """
#   Preprocess single image file 
#     (1) Read original image as YCbCr format (and grayscale as default)
#     (2) Normalize
#     (3) Apply image file with bicubic interpolation

#   Args:
#     path: file path of desired file
#     input_: image applied bicubic interpolation (low-resolution)
#     label_: image with original resolution (high-resolution)
#   """
#   # path_depth = 
#   # path_I = 
#   image = imread(path, is_grayscale=True)
#   #I_add = imread(path, is_grayscale=False)
#   label_ = modcrop(image, scale)

#   # Must be normalized.astype(np.uint8)


#   # label_ = image / 255.
#   label_ = label_ / 255.
#   # plt.imshow(image)
#   # plt.show()
#   input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
#   # input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)
#   # input_= image / 255.
#   return input_, label_

def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  if FLAGS.is_train:
    #filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data = data+glob.glob(os.path.join(data_dir, "*.jpg"))+glob.glob(os.path.join(data_dir, "*.tif"))

  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
    data = glob.glob(os.path.join(data_dir, "*.bmp"))

  return data

def make_data(sess, data, label):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  if FLAGS.is_train:
    savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return imageio.imread(path, pilmode = 'RGB', as_gray = True).astype(np.float)
  else:
    return imageio.imread(path, pilmode = 'RGB').astype(np.float)
 
# 
def modcrop(image, scale=8):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def get_image_batch(train_list,start_id,end_id):
  target_list = train_list[start_id:end_id]
  input_list  = []

  for pair in target_list:
    input_img_ob = scipy.io.loadmat(pair)
    dlist = [key for key in input_img_ob if not key.startswith('__')]
    input_img = input_img_ob[dlist[0]]
    input_list.append(input_img)

  input_list = np.array(input_list)
  if len(input_list.shape) == 3:
    input_list.resize([end_id-start_id, input_list.shape[1], input_list.shape[2], 1])
  # import pdb  
  # pdb.set_trace()
  return input_list

def get_image_batch_new(train_list):
  #print(train_list)
  input_batch  = []
  input_img_ob = scipy.io.loadmat(train_list)
  
  dlist=[key for key in input_img_ob if not key.startswith('__')]
  input_img = np.array(input_img_ob[dlist[0]])

  if len(input_img.shape) == 3:
    input_img.resize([input_img.shape[0], input_img.shape[1], input_img.shape[2], 1])
  # import pdb  
  # pdb.set_trace()
  return input_img

def input_setup(sess, config):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  # Load data path
  if config.is_train:
    data = prepare_data(sess, dataset="Train_small")
  else:
    data = prepare_data(sess, dataset="Test")

  sub_input_sequence = []
  sub_label_sequence = []
  padding = 0 # 6

  if config.is_train:
    for i in range(len(data)):
      input_, label_ = preprocess(data[i], config.scale)

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape

      for x in range(0 , h - config.image_size + 1, config.image_size):
        for y in range(0 , w - config.image_size + 1, config.image_size):
          sub_input = input_[x:x+config.image_size, y:y+config.image_size,:] # [33 x 33]
          sub_label = label_[x:x+config.image_size, y:y+config.image_size,:] # [21 x 21]
          A=1-0.2*np.random.rand(1)
          t=0.05+0.95*np.random.rand(1)
          sub_input[:,:,1] = sub_input[:,:,1]*t+(1-t)*A
          sub_input[:,:,2] = sub_input[:,:,2]*t+(1-t)*A
          sub_input[:,:,0] = sub_input[:,:,0]*t+(1-t)*A
          # Make channel value
          # plt.imshow(sub_label)
          # plt.show()
          # plt.imshow(sub_input)
          # plt.show()
          # import pdb 	
          # pdb.set_trace()
          sub_label=(sub_label*255).astype(np.uint8)
          sub_input=(sub_input*255).astype(np.uint8)
          # sub_label=color.rgb2hsv(sub_label)
          # sub_input=color.rgb2hsv(sub_input)
          sub_label=color.rgb2lab(sub_label)
          sub_input=color.rgb2lab(sub_input)
          sub_label[:,:,0]=(sub_label[:,:,0]-50)/50
          sub_input[:,:,0]=(sub_input[:,:,0]-50)/50
          sub_label[:,:,1]=(sub_label[:,:,1])/128
          sub_label[:,:,2]=(sub_label[:,:,2])/128
          sub_input[:,:,1]=(sub_input[:,:,1])/128
          sub_input[:,:,2]=(sub_input[:,:,2])/128
          # sub_label=(sub_label)-0.5
          # sub_input=(sub_input)-0.5
          sub_input = sub_input.reshape([config.image_size, config.image_size, 3])  
          sub_label = sub_label.reshape([config.label_size, config.label_size, 3])
          
          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)
          
  else:
    input_, label_ = preprocess(data[2], config.scale)
    print('I AM ASKING FOR PREPROCESSING ............')
    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape

    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    nx = ny = 0 
    for x in range(0, h-config.image_size+1, config.image_size):
      nx += 1; ny = 0
      for y in range(0, w-config.image_size+1, config.stride):
        ny += 1
        sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
        sub_label = label_[x+padding:x+padding+config.label_size, y+padding:y+padding+config.label_size] # [21 x 21]
        
        sub_input = sub_input.reshape([config.image_size, config.image_size, 3])  
        sub_label = sub_label.reshape([config.label_size, config.label_size, 3])

        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)

  """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """
  # Make list to numpy array. With this transform
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 3]
  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 3]

  make_data(sess, arrdata, arrlabel)

  if not config.is_train:
    return nx, ny
def rmse(im1,im2):
  # import pdb  
  # pdb.set_trace()
  diff=np.square(im1.astype(np.float)-im2.astype(np.float))
  diff_sum=np.mean(diff)
  rmse=np.sqrt(diff_sum)
  return rmse    
def imsave(image, path):
  return imageio.imwrite(path, image)
# def rgb2ycbcr(x):
#   r,g,b=x[:,:,0]*255,x[:,:,1]*255,x[:,:,2]*255
#   y=(0.257*r+0.564*g+0.098*b+16)/255
#   cb=(-0.148*r-0.291*g+0.439*b+128)/255
#   cr=(0.439*r-0.368*g-0.071*b+128)/255
#   out[:,:,0]=y
#   out[:,:,1]=cb
#   out[:,:,2]=cr
#   return out
def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img
def loss_gradient_difference(true, generated):
   true_x_shifted_right = true[:,1:,:,:]
   true_x_shifted_left = true[:,:-1,:,:]
   true_x_gradient = tf.abs(true_x_shifted_right - true_x_shifted_left)

   generated_x_shifted_right = generated[:,1:,:,:]
   generated_x_shifted_left = generated[:,:-1,:,:]
   generated_x_gradient = tf.abs(generated_x_shifted_right - generated_x_shifted_left)

   loss_x_gradient = tf.reduce_mean(tf.square(true_x_gradient - generated_x_gradient))

   true_y_shifted_right = true[:,:,1:,:]
   true_y_shifted_left = true[:,:,:-1,:]
   true_y_gradient = tf.abs(true_y_shifted_right - true_y_shifted_left)

   generated_y_shifted_right = generated[:,:,1:,:]
   generated_y_shifted_left = generated[:,:,:-1,:]
   generated_y_gradient = tf.abs(generated_y_shifted_right - generated_y_shifted_left)
    
   loss_y_gradient = tf.reduce_mean(tf.square(true_y_gradient - generated_y_gradient))

   loss = loss_x_gradient + loss_y_gradient
   return loss

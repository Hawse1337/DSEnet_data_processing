#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
import scipy.io.wavfile as wavfile
import math


# In[2]:


def norm(inp):
    M = np.amax(np.absolute(inp))
    return inp/M


# In[3]:


#### write ####

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# In[4]:


save_path = './TFR'
out_file_name = 'DSEnet_DATA2_DEMAND_4_train.tfrecords'

target_dir = './DSEnet_train_DATA_1cm_4/LookD'
noisy_m1_dir = './DSEnet_train_DATA_1cm_4/MICR'
noisy_m2_dir = './DSEnet_train_DATA_1cm_4/MICL'

window = 16384
stride = int((0.5)*window)
print(stride)


# In[5]:


out_filepath= os.path.join(save_path, out_file_name)
out_file = tf.python_io.TFRecordWriter(out_filepath)


# In[6]:


target_ = os.listdir(target_dir)
noisy_m1_ = os.listdir(noisy_m1_dir)
noisy_m2_ = os.listdir(noisy_m2_dir) 
print(len(target_))
print(len(noisy_m1_))
print(len(noisy_m2_))


# In[7]:


total_chunks = 0
for i in target_:
    fm, target = wavfile.read('./DSEnet_train_DATA_1cm_4/LookD/%s'%i)
#     print(type(target[-1]))
    fm, noisy_m1 = wavfile.read('./DSEnet_train_DATA_1cm_4/MICR/%s'%i)
#     print(type(noisy_m1[-1]))
    
    fm, noisy_m2 = wavfile.read('./DSEnet_train_DATA_1cm_4/MICL/%s'%i)
#     print(type(noisy_m2[-1]))
    
  
    #### check data ####
    if fm != 16000:
        raise ValueError('Not 16kHz')
    if len(noisy_m2) != len(noisy_m1):
        raise ValueError('Clean and noise m1 are different')
    if len(target) != len(noisy_m2):
        raise ValueError('Clean and noise m2 are different')
    #### check data ####

    if int(len(target)/stride) == float(len(target))/stride:
        n_slice = int((len(target)-stride)/stride)
    else:
        n_slice = int(len(target)/stride)

    total_chunks = total_chunks + n_slice
print( 'Total chunks : ', total_chunks)


# In[11]:


indx = 0
for i in target_:
    if indx % 100 == 0:
        print('{}/{}'.format(indx + 1, total_chunks))

    fm, target = wavfile.read('./DSEnet_train_DATA_1cm_4/LookD/%s'%i)
#     target = norm(target)
    fm, noisy_m1 = wavfile.read('./DSEnet_train_DATA_1cm_4/MICR/%s'%i)
    fm, noisy_m2 = wavfile.read('./DSEnet_train_DATA_1cm_4/MICL/%s'%i)
    #### slice data ####
    if int(len(target)/stride) == float(len(target))/stride:
        n_slice = int((len(target)-stride)/stride)
    else:
        n_slice = int(len(target)/stride)
    
    #print(n_slice)
    
    for j in range(n_slice):
        if j == n_slice-1:
            start_slice = j*stride
            n_pad = window-(len(target)-start_slice)
            end_slice = len(target)
            pad = np.zeros(n_pad)
            pad = np.array(pad, dtype=np.float32)
            
            target_sliced = target[start_slice:end_slice]
            target_sliced = np.append(target_sliced,pad)
            
            noisy_m1_sliced = noisy_m1[start_slice:end_slice]
            noisy_m1_sliced = np.append(noisy_m1_sliced,pad)
            
            noisy_m2_sliced = noisy_m2[start_slice:end_slice]
            noisy_m2_sliced = np.append(noisy_m2_sliced,pad)

            target_sliced = target_sliced.tostring()
            noisy_m1_sliced = noisy_m1_sliced.tostring()
            noisy_m2_sliced = noisy_m2_sliced.tostring()
        else:
            start_slice = j*stride
            end_slice = window + j*stride

            target_sliced = target[start_slice:end_slice]
            noisy_m1_sliced = noisy_m1[start_slice:end_slice]
            noisy_m2_sliced = noisy_m2[start_slice:end_slice]

            target_sliced = target_sliced.tostring()
            noisy_m1_sliced = noisy_m1_sliced.tostring()
            noisy_m2_sliced = noisy_m2_sliced.tostring()
      #### slice data ####
        if( indx < 100*(total_chunks//100)):
            example = tf.train.Example(features = tf.train.Features(feature={
                'target_raw' : _bytes_feature(target_sliced),
                'noisy_m1_raw' : _bytes_feature(noisy_m1_sliced),
                'noisy_m2_raw' : _bytes_feature(noisy_m2_sliced)}))
            out_file.write(example.SerializeToString())
            indx = indx + 1
        
        

out_file.close()
print('indx : ',indx,'DONE')


# In[7]:





# In[ ]:





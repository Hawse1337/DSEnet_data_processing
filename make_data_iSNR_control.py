#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import scipy.io.wavfile as wf
import numpy as np
import random
import matplotlib.pyplot as plt


# In[2]:


####
#

#normalize a single array
def qnorm(inp):
    M = np.amax(np.absolute(inp))
    return inp*((2**15-1)/M)

def norm(inp):
    M = np.amax(np.absolute(inp))
    return inp/M

def iSNR_control(signal,noise,SNR):
    
    #now A
    A_now = 0
    for i in noise :
        A_now = A_now + i**2
    A_now = np.sqrt(A_now/len(noise))
    
    #aim A
    A_sig = 0
    for i in signal :
        A_sig = A_sig + i**2
    A_sig = np.sqrt(A_sig/len(signal))
    
    A_aim = A_sig/(10**(SNR/20))
    
    k = A_aim/A_now
#     print(k)
    noise = k*noise
    return noise

def get_n_sample(signal,noiseR,noiseL):
    
    Ls = len(signal)
    Ln = len(noiseR)
    
    Ln = Ln-Ls
    p = random.randrange(0,Ln)
    
    return noiseR[p:p+Ls],noiseL[p:p+Ls]

#
####


# In[3]:


NOISE_NAMES={0:'DLIVING',
             1:'DWASHING',
             2:'NFIELD',
             3:'NPARK',
             4:'NRIVER',
             5:'OHALLWAY',
             6:'OMEETING',
             7:'OOFFICE',
             8:'PCAFETER',
             9:'PRESTO',
             10:'PSTATION',
             11:'SPSQUARE',
             12:'STRAFFIC',
             13:'TBUS',
             14:'TCAR',
             15:'TMETRO'}


SNR_train_list = [15,10,5,0]
SNR_test_list = [17.5,12.5,7.5,2.5]
print(NOISE_NAMES[4])


# In[4]:


LD = os.listdir('./clean_testset_wav_16k')
print(len(LD))
    
I = []
II=[]
III=[]
t = True
while t:
    n1 = random.randrange(0,16)
    n2 = random.randrange(0,16)
    n3 = random.randrange(0,16)

    d1 = random.randrange(0,6)
    d2 = random.randrange(6,12)
    d3 = random.randrange(12,18)
      
    k = [n1,n2,n3,d1,d2,d3]

    I.append(k)
    II.append(n1)
    II.append(n2)
    II.append(n3)
    
    III.append(d1)
    III.append(d2)
    III.append(d3)
    if len(I) == len(LD):
        t = False
print(len(I))

# plt.hist(II,16)
# plt.show()
# plt.hist(III,18)
# plt.show()


# In[5]:


indx = 0
print(LD[0])
for f_name in LD:
    print(f_name)
    SR,clean_wav = wf.read('./clean_testset_wav_16k/%s'%(f_name))
    clean_wav = norm(clean_wav)
#     print('clean_wav : ',type(clean_wav[0]))
    
    nn1,nn2,nn3,nd1,nd2,nd3 = I[indx]

    if nd1 >= 9:
        nd1 = nd1+1
    if nd2 >= 9:
        nd2 = nd2+1
    if nd3 >= 9:
        nd3 = nd3+1
    
    SR,noise1R = wf.read('./DEMAND_16k_4/%s_%dd_R.wav'%(NOISE_NAMES[nn1],nd1))
    if SR != 16000:
        print(f_name)
    SR,noise1L = wf.read('./DEMAND_16k_4/%s_%dd_L.wav'%(NOISE_NAMES[nn1],nd1))
    if SR != 16000:
        print(f_name)
        
    SR,noise2R = wf.read('./DEMAND_16k_4/%s_%dd_R.wav'%(NOISE_NAMES[nn2],nd2))
    if SR != 16000:
        print(f_name)
    SR,noise2L = wf.read('./DEMAND_16k_4/%s_%dd_L.wav'%(NOISE_NAMES[nn2],nd2))
    if SR != 16000:
        print(f_name)
        
    SR,noise3R = wf.read('./DEMAND_16k_4/%s_%dd_R.wav'%(NOISE_NAMES[nn3],nd3))
    if SR != 16000:
        print(f_name)
    SR,noise3L = wf.read('./DEMAND_16k_4/%s_%dd_L.wav'%(NOISE_NAMES[nn3],nd3))
    if SR != 16000:
        print(f_name)
        
    indx = indx+1
    

    n1R_sample,n1L_sample = get_n_sample(clean_wav,noise1R,noise1L)
    
    n2R_sample,n2L_sample = get_n_sample(clean_wav,noise3R,noise2L)
    
    n3R_sample,n3L_sample = get_n_sample(clean_wav,noise3R,noise3L)
    
#     print(len(clean_wav))
#     print('n1R_sample : ',type(n1R_sample[0]))
#     print(len(n1L_sample))
    
    noiseR = n1R_sample+n2R_sample+n3R_sample
    noiseL = n1L_sample+n2L_sample+n3L_sample
    
    noiseR = iSNR_control(signal=clean_wav,noise=noiseR,SNR=SNR_test_list[indx%4])
    noiseL = iSNR_control(signal=clean_wav,noise=noiseL,SNR=SNR_test_list[indx%4])
    
    noisy_wav1 = norm(np.array(clean_wav+noiseR))
    noisy_wav2 = norm(np.array(clean_wav+noiseL))

    
    if len(noisy_wav1) != len(noisy_wav2):
        raise ValueError
    if len(noisy_wav1) != len(clean_wav):
        raise ValueError
        
    noisy_wav1 = noisy_wav1.astype(np.float32)
    noisy_wav2 = noisy_wav2.astype(np.float32)
    clean_wav = clean_wav.astype(np.float32)

        
    wf.write('./DSEnet_test_DATA_1cm_4/MICR/%s'%f_name,SR,noisy_wav1)
    wf.write('./DSEnet_test_DATA_1cm_4/MICL/%s'%f_name,SR,noisy_wav2)
    wf.write('./DSEnet_test_DATA_1cm_4/LookD/%s'%f_name,SR,clean_wav)
    
#     print('noisy_wav[0] : ',type(noisy_wav1[0]))
    
#     wav=[]
#     for ii in range(len(noisy_wav[0])):
#         wav.append([noisy_wav[0][ii],noisy_wav[1][ii]])
#     wf.write('./DSEnet_test_DATA_1cm_2/stereo/%s'%f_name,SR,np.array(wav))
    print(indx,'/',len(LD))
    
    


# In[6]:


LD = os.listdir('./clean_trainset_wav_16k')
print(len(LD))
    
I = []
II=[]
III=[]
t = True
while t:
    n1 = random.randrange(0,16)
    n2 = random.randrange(0,16)
    n3 = random.randrange(0,16)

    d1 = random.randrange(0,6)
    d2 = random.randrange(6,12)
    d3 = random.randrange(12,18)
      
    k = [n1,n2,n3,d1,d2,d3]

    I.append(k)
    II.append(n1)
    II.append(n2)
    II.append(n3)
    
    III.append(d1)
    III.append(d2)
    III.append(d3)
    if len(I) == len(LD):
        t = False
print(len(I))

# plt.hist(II,16)
# plt.show()
# plt.hist(III,18)
# plt.show()


# In[7]:


indx = 0
print(LD[0])
for f_name in LD:
#     print(f_name)
    SR,clean_wav = wf.read('./clean_trainset_wav_16k/%s'%(f_name))
    clean_wav = norm(clean_wav)
    
#     print(clean_wav)
    
    nn1,nn2,nn3,nd1,nd2,nd3 = I[indx]

    if nd1 >= 9:
        nd1 = nd1+1
    if nd2 >= 9:
        nd2 = nd2+1
    if nd3 >= 9:
        nd3 = nd3+1
    
    SR,noise1R = wf.read('./DEMAND_16k_4/%s_%dd_R.wav'%(NOISE_NAMES[nn1],nd1))
    if SR != 16000:
        print(f_name)
    SR,noise1L = wf.read('./DEMAND_16k_4/%s_%dd_L.wav'%(NOISE_NAMES[nn1],nd1))
    if SR != 16000:
        print(f_name)
        
    SR,noise2R = wf.read('./DEMAND_16k_4/%s_%dd_R.wav'%(NOISE_NAMES[nn2],nd2))
    if SR != 16000:
        print(f_name)
    SR,noise2L = wf.read('./DEMAND_16k_4/%s_%dd_L.wav'%(NOISE_NAMES[nn2],nd2))
    if SR != 16000:
        print(f_name)
        
    SR,noise3R = wf.read('./DEMAND_16k_4/%s_%dd_R.wav'%(NOISE_NAMES[nn3],nd3))
    if SR != 16000:
        print(f_name)
    SR,noise3L = wf.read('./DEMAND_16k_4/%s_%dd_L.wav'%(NOISE_NAMES[nn3],nd3))
    if SR != 16000:
        print(f_name)
        
    indx = indx+1
    

    n1R_sample,n1L_sample = get_n_sample(clean_wav,noise1R,noise1L)
    
    n2R_sample,n2L_sample = get_n_sample(clean_wav,noise3R,noise2L)
    
    n3R_sample,n3L_sample = get_n_sample(clean_wav,noise3R,noise3L)
    
#     print(len(clean_wav))
#     print(n1R_sample)
#     print(len(n1L_sample))
    
    noiseR = n1R_sample+n2R_sample+n3R_sample
    noiseL = n1L_sample+n2L_sample+n3L_sample
    
    noiseR = iSNR_control(signal=clean_wav,noise=noiseR,SNR=SNR_train_list[indx%4])
    noiseL = iSNR_control(signal=clean_wav,noise=noiseL,SNR=SNR_train_list[indx%4])
    
    noisy_wav1 = norm(np.array(clean_wav+noiseR))
    noisy_wav2 = norm(np.array(clean_wav+noiseL))
    
    noisy_wav1 = noisy_wav1.astype(np.float32)
    noisy_wav2 = noisy_wav2.astype(np.float32)
    clean_wav = clean_wav.astype(np.float32)
    
    
    if len(noisy_wav1) != len(noisy_wav2):
        raise ValueError
    if len(noisy_wav1) != len(clean_wav):
        raise ValueError

        
    wf.write('./DSEnet_train_DATA_1cm_4/MICR/%s'%f_name,SR,noisy_wav1)
    wf.write('./DSEnet_train_DATA_1cm_4/MICL/%s'%f_name,SR,noisy_wav2)
    wf.write('./DSEnet_train_DATA_1cm_4/LookD/%s'%f_name,SR,clean_wav)
    
#     wav=[]
#     for ii in range(len(noisy_wav[0])):
#         wav.append([noisy_wav[0][ii],noisy_wav[1][ii]])
#     wf.write('./DSEnet_train_DATA_1cm_2/stereo/%s'%f_name,SR,np.array(wav))
    
    print(indx,'/',len(LD))
    


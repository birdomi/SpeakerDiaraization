import tensorflow as tf
import numpy as np
import a
import scipy.io.wavfile as wav
import time
import random
import sys
import os


wavPath='Signals'
labPath='transcripts'
mfcc_path=os.listdir(wavPath)
label_path=os.listdir(labPath)

datanum=40
train_rate=1
d={}

for i in range(1):
    d.update({i:a.ICSI_Data()})
    d[i].Get_Data(wavPath+'/'+mfcc_path[i],labPath+'/'+label_path[i])
print(d[0]._LABEL[0].shape)


sess=tf.Session()
rnn=a.RNN_Model(sess,'RNN',0.1)
sess.run(tf.global_variables_initializer())

for i in range(1000):
    cost_sum=0
    for j in range(d[0].numberBatch):
        cost,_=rnn.Train(d[0]._MFCC[j],d[0]._LABEL[j])
        cost_sum+=cost
    print('step: ',i)
    print('cost: ',cost_sum)

    if(i%50==0):
        for j in range(d[0].numberBatch):
            print(rnn.Accuracy(d[0]._MFCC[j],d[0]._LABEL[j]))

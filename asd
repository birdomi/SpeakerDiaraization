import tensorflow as tf
import numpy as np
import a
import scipy.io.wavfile as wav
import time
import random
import os
from utils import Logger
from utils import Time
from plot_graph import plot_static

mfcc_path=os.listdir('Signals')
label_path=os.listdir('transcripts')

wavPath='Signals'
labPath='transcripts'
mfcc_path=os.listdir(wavPath)
label_path=os.listdir(labPath)

datanum=10

train_rate=0.8

trainRange=range(0,int(datanum*train_rate))
testRange=range(int(datanum*train_rate),datanum)

dH={}

for i in testRange:
    dH.update({i:a.ICSI_Data()})
    dH[i].MakeData(wavPath+'/'+mfcc_path[i],labPath+'/'+label_path[i],'On')

dN={}

for i in testRange:
    dN.update({i:a.ICSI_Data()})
    dN[i].MakeData(wavPath+'/'+mfcc_path[i],labPath+'/'+label_path[i],'N')

sess=tf.Session()
rnn=a.RNN_Model_usingMiddle(sess,'RNNAll',0.001)
sess.run(tf.global_variables_initializer())

rnn.Restore('Model/RNNMiddle_Hamming_10_20180303212832')
print('RNNMiddle_Hamming_10_20180303212832')
for num in testRange:
    for batch in range(dH[num].numberBatch):
        rnn.return_ResultPerfection(dH[num]._MFCC[batch],dH[num]._LABEL[batch],dH[num]._LABEL_Perfection[batch])
tf.reset_default_graph()


sess=tf.Session()
rnn=a.RNN_Model_usingMiddle(sess,'RNNAll',0.001)
sess.run(tf.global_variables_initializer())
rnn.Restore('Model/RNNMiddle_NoHamming_10_20180303202033')
print('RNNMiddle_NoHamming_10_20180303202033')
for num in testRange:
    for batch in range(dN[num].numberBatch):
        rnn.return_ResultPerfection(dN[num]._MFCC[batch],dN[num]._LABEL[batch],dN[num]._LABEL_Perfection[batch])
tf.reset_default_graph()

sess=tf.Session()
rnn=a.RNN_Model_usingBoundary(sess,'RNNAll',0.001)
sess.run(tf.global_variables_initializer())
rnn.Restore('Model/RNNBoundary_Hamming_10_20180303234612')
print('RNNBoundary_Hamming_10_20180303234612')
for num in testRange:
    for batch in range(dH[num].numberBatch):
        rnn.return_ResultPerfection(dH[num]._MFCC[batch],dH[num]._LABEL[batch],dH[num]._LABEL_Perfection[batch])
tf.reset_default_graph()

sess=tf.Session()
rnn=a.RNN_Model_usingBoundary(sess,'RNNAll',0.001)
sess.run(tf.global_variables_initializer())
rnn.Restore('Model/RNNBoundary_NoHamming_10_20180303223608')
print('RNNBoundary_NoHamming_10_20180303223608')
for num in testRange:
    for batch in range(dN[num].numberBatch):
        rnn.return_ResultPerfection(dN[num]._MFCC[batch],dN[num]._LABEL[batch],dN[num]._LABEL_Perfection[batch])
tf.reset_default_graph()




def matrixAccuracy(matrix):
    number_0=matrix[0][0]+matrix[0][1]
    number_1=matrix[1][0]+matrix[1][1]

    acc0_0=matrix[0][0]/number_0
    acc0_1=matrix[0][1]/number_0
    acc1_0=matrix[1][0]/number_1
    acc1_1=matrix[1][1]/number_1

    accMatrix=[[acc0_0,acc0_1],
               [acc1_0,acc1_1]]
    
    return accMatrix







import tensorflow as tf
import numpy as np
import a
import scipy.io.wavfile as wav
import time
import random
import sys
import os
"""
a1=[[[0,1]],
    [[1,0]],
    [[0,1]]]
a2=[[[0,1]],
    [[0,1]],
    [[0,1]]]
a3=[[[1,0]],
    [[1,0]],
    [[1,0]]]

sess=tf.Session()
a1=sess.run(tf.argmax(a1,2))
a2=sess.run(tf.argmax(a2,2))
a3=sess.run(tf.argmax(a3,2))

print(a1,a2,a3)
print(sess.run(tf.concat([a1,a2,a3],1)))


"""
wavPath='Signals'
labPath='transcripts'
mfcc_path=os.listdir(wavPath)
label_path=os.listdir(labPath)

datanum=40
train_rate=0.7
trainRange=range(0,int(datanum*train_rate))
testRange=range(int(datanum*train_rate),datanum)

d={}

for i in range(datanum):
    d.update({i:a.ICSI_Data()})
    d[i].Get_Data(wavPath+'/'+mfcc_path[i],labPath+'/'+label_path[i])

sess=tf.Session()
rnn=a.RNN_Model(sess,'RNN',0.001)
sess.run(tf.global_variables_initializer())
rnn.Restore()
print(rnn.Show_Shape(d[0]._MFCC[0],d[0]._LABEL[0]))
for i in range(5001):
    cost_sum=0
    starttime=time.time()
    for number in trainRange:
        for index in range(d[number].numberBatch):
            cost,_=rnn.Train(d[number]._MFCC[index],d[number]._LABEL[index])
            cost_sum+=cost
    trainTime=time.time()-starttime
    print('step: ',i,' time: ',trainTime)
    print('cost: ',cost_sum)
    print()

    if(i%100==0):
        print('####train####')
        for number in trainRange:
            print(number)
            for index in range(d[number].numberBatch):
                print(rnn.Accuracy(d[number]._MFCC[index],d[number]._LABEL[index]))
        print('#####test####')
        for number in testRange:
            print(number)
            for index in range(d[number].numberBatch):
                print(rnn.Accuracy(d[number]._MFCC[index],d[number]._LABEL[index]))
    if(i%100==0 and i!=0):
        rnn.Save()

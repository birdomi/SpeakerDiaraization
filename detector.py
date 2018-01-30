
import tensorflow as tf
import numpy as np
import a
import scipy.io.wavfile as wav
import time
import random
import sys

d=a.Data()
mfccname=['Bdb001.interaction.wav','Bed003.interaction.wav','Bed004.interaction.wav','Bed005.interaction.wav',
          'Bed006.interaction.wav','Bed008.interaction.wav','Bed009.interaction.wav','Bed010.interaction.wav',
          'Bed011.interaction.wav','Bed012.interaction.wav','Bed013.interaction.wav','Bed014.interaction.wav',
          'Bed015.interaction.wav','Bed016.interaction.wav','Bed017.interaction.wav']
labelname=['Bdb001.txt','Bed003.txt','Bed004.txt','Bed005.txt','Bed006.txt','Bed008.txt',
           'Bed009.txt','Bed010.txt','Bed011.txt','Bed012.txt','Bed013.txt','Bed014.txt',
           'Bed015.txt','Bed016.txt','Bed017.txt']
datanum=15

"""
mfcc=[]
for i in range(datanum):
    mfcc.append(d.Mfcc(mfccname[i]))
"""
###
sess=tf.Session()
rnn_model=a.RNN_model(sess,'rnn')
sess.run(tf.global_variables_initializer())
rnn_model.restore()

for i in range(datanum):
    d.Load_Data(mfccname[i],labelname[i],i)

for i in range(datanum):
    for k in range(d.data[i].data_length):
        print(d.data[i].mfcc_data[k].shape)
        print(rnn_model.accuracy(d.data[i].mfcc_data[k],d.data[i].label_data[k]))











"""
for i in range(datanum):
    d.Load_Data(mfccname[i],labelname[i],i)
for i in range(datanum):    
    print('Bed'+str(i))
    for k in range(d.data[i].data_length):
        print(rnn_model.accuracy(d.data[i].mfcc_data[k],d.data[i].label_data[k]))

X=[]
for i in range(datanum):
    X.append(rnn_model.Make_Result(mfcc[i]))

sess=tf.Session()
timeLiner=a.Time_liner(sess,'timeLiner')
sess.run(tf.global_variables_initializer())

for i in range(datanum):
    X[i]=timeLiner.Line_Predict(X[i])
Y=[]
for i in range(datanum):
    Y.append(timeLiner.Label_Time(labelname[i]))

for i in range(100000):
    r=random.sample(range(datanum),datanum)
    for j in range(datanum):
        cost,_=timeLiner.train(X[r[j]],Y[r[j]])

    if(i%500==0):
        print('##',i)
        for j in range(datanum):
            print(timeLiner.accuracy(X[j],Y[j]))

for i in range(datanum):
    print(timeLiner.accuracy(X[i],Y[i]))
timeLiner.SaveModel()
"""

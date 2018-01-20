import tensorflow as tf
import numpy as np
import a
import scipy.io.wavfile as wav
import time
import random
import sys

"""
if(len(sys.argv)==1):
    print('detector.exe inputWaveFileDir.wav')
    #sys.exit()
waveFile=sys.argv[1]
"""
dataClass=a.Data()
###
X2=dataClass.Mfcc('Bed017.interaction.wav')

sess=tf.Session()
rnn_model=a.RNN_model(sess,'rnn')
sess.run(tf.global_variables_initializer())
rnn_model.restore()

X1=rnn_model.Make_Result(X1)
X2=rnn_model.Make_Result(X2)
X3=rnn_model.Make_Result(X3)
X4=rnn_model.Make_Result(X4)


sess1=tf.Session()
timeLiner=a.Time_liner(sess1,'timeLiner')
sess1.run(tf.global_variables_initializer())
timeLiner.RestoreModel()

Y1=timeLiner.Label_Time('Bed011.txt')
Y2=timeLiner.Label_Time('Bed012.txt')
Y3=timeLiner.Label_Time('Bed013.txt')
Y4=timeLiner.Label_Time('Bed014.txt')

print('test accuracy : ',timeLiner.accuracy(X3,Y3))

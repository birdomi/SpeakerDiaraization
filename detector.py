import tensorflow as tf
import numpy as np
import a
import scipy.io.wavfile as wav
import time
import random
import sys


if(len(sys.argv)==1):
    print('detector.exe inputWaveFileDir.wav')
    #sys.exit()
waveFile=sys.argv[1]

dataClass=a.Data()
###
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
rnn_model=a.RNN_model(sess,'rnn')
sess.run(tf.global_variables_initializer())
rnn_model.restore()

sess1=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
timeLiner=a.Time_liner(sess1,'timeLiner')
sess1.run(tf.global_variables_initializer())
timeLiner.RestoreModel()

mfcc=dataClass.Mfcc(waveFile)
mfcc=timeLiner.Line_Predict(rnn_model.Make_Result(mfcc))
timeLiner.Save_Result(mfcc)

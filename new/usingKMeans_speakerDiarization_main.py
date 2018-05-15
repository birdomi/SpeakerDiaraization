import tensorflow as tf
import numpy as np
import os
import ut
import sklearn.cluster

CALLHOME=ut.CALLHOME_Data()

wavPath='CALLHOME/Signals'
labPath='CALLHOME/transcripts'
mfcc=os.listdir(wavPath)
label_path=os.listdir(labPath)

mfcc_path=[]
for i in range(len(mfcc)):
    if(mfcc[i][-4:]=='.csv'):
        pass
    else:
        mfcc_path.append(mfcc[i]) 

print(len(mfcc_path),len(label_path))
datanum=1#len(mfcc_path)

sess=tf.Session()
rnn=ut.VAD_Model(sess,'RNN_',0.001)
sess.run(tf.global_variables_initializer())
rnn.Restore('VADM/VAD')#application_path+
for i in range(datanum):
    mfcc_x=ut.INPUT_Data().Get_Data('4104.wav')
    print(mfcc_x.shape)
    segment=rnn.Make_Segment(mfcc_x)
    
    openSmile=ut.OpenSmile('list',segment)
    segmentFeature=openSmile.Get_features('waveTemp.wav')
    print(len(segment))
    print(segmentFeature.shape)

    cluster=sklearn.cluster.k_means(segmentFeature,2)
    print(cluster[1].shape)

for i in range(len(segment)):
    print(segment[i],cluster[1][i])





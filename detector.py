import tensorflow as tf
import numpy as np
import a
import scipy.io.wavfile as wav
import time
import random

mfccpath_train=['Bed003.interaction.wav','Bed004.interaction.wav','Bed005.interaction.wav',
                'Bed006.interaction.wav']
labelpath_train=['Bed003.txt','Bed004.txt','Bed005.txt',
                'Bed006.txt']

mfcc_path_dev=['Bed011.interaction.txt','Bed012.interaction.txt','Bed013.interaction.txt','Bed014.interaction.txt']
label_path_dev=['Bed011.txt','Bed012.txt','Bed013.txt','Bed014.txt']

mfcc_path_test=['Bed016.interaction.wav','Bed017.interaction.wav']
label_path_test=['Bed016.txt','Bed017.txt']

DataModule=a.Data()                                  
DataModule.Load_Data(mfccpath_train,labelpath_train,'train')
DataModule.Load_Data(mfcc_path_test,label_path_test,'test')
###
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
rnn_model=a.RNN_model(sess,'rnn')
sess.run(tf.global_variables_initializer())
rnn_model.restore()


print(DataModule.data['train'].data_length)
for i in range(1,1501):
    avg_cost=0
    avg_train1=0.0
    avg_train2=0.0
    avg_test1=0.0
    avg_test2=0.0
    start=time.time()
    
    randomIndex=random.sample(range(DataModule.data['train'].data_length),DataModule.data['train'].data_length)
    for j in range(DataModule.data['train'].data_length):        
        cost,out,_=rnn_model.train(DataModule.data['train'].mfcc_data[randomIndex[j]],DataModule.data['train'].label_data[randomIndex[j]],len(DataModule.data['train'].mfcc_data[randomIndex[j]]))
        avg_cost+=cost
    traintime=time.time()-start
    
    if(i%450==0):
        rnn_model.save()
        print('model saved')
    
    if(i%75==0):                
        for j in range(DataModule.data['train'].data_length):
            ac1,ac2=rnn_model.accuracy(DataModule.data['train'].mfcc_data[j],DataModule.data['train'].label_data[j])
            avg_train1+=ac1
            avg_train2+=ac2
        for j in range(DataModule.data['test'].data_length):
            ac1,ac2=rnn_model.accuracy(DataModule.data['test'].mfcc_data[j],DataModule.data['test'].label_data[j])
            avg_test1+=ac1
            avg_test2+=ac2
        avg_train1=avg_train1/DataModule.data['train'].data_length
        avg_train2=avg_train2/DataModule.data['train'].data_length
        avg_test1=avg_test1/DataModule.data['test'].data_length
        avg_test2=avg_test2/DataModule.data['test'].data_length    
        print('accuracy_train1:',avg_train1,'accuracy_train2',avg_train2)
        print('accuracy_test1:',avg_test1,'accuracy_test2',avg_test2)    
    
    avg_cost=avg_cost/DataModule.data['test'].data_length
    
    print('step',i)
    print('cost1: ',avg_cost)
    print('time',traintime)
print('train run')
for i in range(DataModule.data['train'].data_length):
    ac1,ac2=rnn_model.accuracy(DataModule.data['train'].mfcc_data[i],DataModule.data['train'].label_data[i])
    print(a.file.Return_Num_1(np.reshape(rnn_model.predict(DataModule.data['train'].mfcc_data[i]),[-1])))
    print(a.file.Return_Num_1(np.reshape(DataModule.data['train'].label_data[i],[-1])))
    print(ac1)
    print(ac2)
    print(' ')

print('test run')
for j in range(DataModule.data['test'].data_length):
    ac1,ac2=rnn_model.accuracy(DataModule.data['test'].mfcc_data[j],DataModule.data['test'].label_data[j])
    print(a.file.Return_Num_1(np.reshape(rnn_model.predict(DataModule.data['test'].mfcc_data[j]),[-1])))
    print(a.file.Return_Num_1(np.reshape(DataModule.data['test'].label_data[j],[-1])))
    print(ac1)
    print(ac2)
    print(' ')

rnn_model.save()
a.beep()

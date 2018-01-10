import tensorflow as tf
import numpy as np
import a
import scipy.io.wavfile as wav
import time
import random

randomindex=random.sample(range(10),10)
for i in range(10):
    print(str(randomindex[i]))

data=a.Data()
data.Data_Append('Bed004.interaction.txt','Bed004_test.txt')
data.Data_Load_forSmall_Label('Bed005.interaction.txt','Bed005_test.txt')
data.Make_Batch()

###
sess=tf.Session()
rnn_model=a.RNN_model(sess,'rnn')
sess.run(tf.global_variables_initializer())
#rnn_model.restore()

print(data.train.data_length)
for i in range(1,501):
    avg_cost=0
    avg_train1=0.0
    avg_train2=0.0
    avg_test1=0.0
    avg_test2=0.0
    start=time.time()
    
    randomIndex=random.sample(range(data.train.data_length),data.train.data_length)
    for j in range(data.train.data_length):        
        cost,out,_=rnn_model.train(data.train.mfcc_data[randomIndex[j]],data.train.label_data[randomIndex[j]],len(data.train.mfcc_data[randomIndex[j]]))
        avg_cost+=cost
    traintime=time.time()-start
    """
    if(i%100==0):
        rnn_model.save()
        print('model saved')
    """
    if(i%10==0):                
        for j in range(data.train.data_length):
            ac1,ac2=rnn_model.accuracy(data.train.mfcc_data[j],data.train.label_data[j])
            avg_train1+=ac1
            avg_train2+=ac2
        for j in range(data.test.data_length):
            ac1,ac2=rnn_model.accuracy(data.test.mfcc_data[j],data.test.label_data[j])
            avg_test1+=ac1
            avg_test2+=ac2
        avg_train1=avg_train1/data.train.data_length
        avg_train2=avg_train2/data.train.data_length
        avg_test1=avg_test1/data.test.data_length
        avg_test2=avg_test2/data.test.data_length    
        print('accuracy_train1:',avg_train1,'accuracy_train2',avg_train2)
        print('accuracy_test1:',avg_test1,'accuracy_test2',avg_test2)

    if(i%10==0):
        print('test Small Label')
        ac1_test_small_Label=0.0
        ac2_test_small_Label=0.0
        for j in range(data.test_small_label.data_length):
            ac1,ac2=rnn_model.accuracy(data.test_small_label.mfcc_data[j],data.test_small_label.label_data[j])
            ac1_test_small_Label+=ac1
            ac2_test_small_Label+=ac2
        ac1_test_small_Label=ac1_test_small_Label/data.test_small_label.data_length
        ac2_test_small_Label=ac2_test_small_Label/data.test_small_label.data_length

        print(ac1_test_small_Label)
        print(ac2_test_small_Label)
    
    
    avg_cost=avg_cost/data.train.data_length
    
    print('step',i)
    print('cost1: ',avg_cost)
    print('time',traintime)
print('train run')
for i in range(data.train.data_length):
    ac1,ac2=rnn_model.accuracy(data.train.mfcc_data[i],data.train.label_data[i])
    print(a.file.Return_Num_1(np.reshape(rnn_model.predict(data.train.mfcc_data[i]),[-1])))
    print(a.file.Return_Num_1(np.reshape(data.train.label_data[i],[-1])))
    print(ac1)
    print(ac2)
    print(' ')

print('test run')
for j in range(data.test.data_length):
    ac1,ac2=rnn_model.accuracy(data.test.mfcc_data[j],data.test.label_data[j])
    print(a.file.Return_Num_1(np.reshape(rnn_model.predict(data.test.mfcc_data[j]),[-1])))
    print(a.file.Return_Num_1(np.reshape(data.test.label_data[j],[-1])))
    print(ac1)
    print(ac2)
    print(' ')

rnn_model.save()
a.beep()

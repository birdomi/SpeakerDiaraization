import tensorflow as tf
import numpy as np
import a
import scipy.io.wavfile as wav
import time



Data=a.file.Extract_Mfcc_OneFile('Bdb001.interaction.wav')
print(Data)
length_Data=(int)(len(Data)/10)
TrainingData=np.ndarray((length_Data,a.vec*10))

for i in range(0,length_Data):
    temp=Data[i*10:i*10+10]
    temp=np.reshape(temp,(1,-1))
    TrainingData[i]=temp

label_scripts=np.loadtxt('Label.txt',dtype=int)
label_scripts=np.reshape(label_scripts,(-1,1))
out_info_fc=tf.Session().run(tf.one_hot(label_scripts,a.no_output))
out_info_fc=np.reshape(out_info_fc,(-1,a.no_output))

seq_length=a.seq_length
label_length=(int)(seq_length/10)
data_length=len(TrainingData)-label_length
print(label_length)

dataX=np.ndarray((data_length,label_length,a.vec*10))
dataY=np.ndarray((data_length,label_length,1))
print(dataY)
for i in range(0,data_length):
    dataX[i]=TrainingData[i:i+label_length]    
    dataY[i]=label_scripts[i:i+label_length]
out_info_label_scripts=tf.Session().run(tf.one_hot(dataY,a.no_output))
out_info_label_scripts=np.reshape(out_info_label_scripts,(-1,label_length,a.no_output))
dataX=np.reshape(dataX,(-1,seq_length,a.vec))
dataY=np.reshape(dataY,(-1,label_length))

print("data ready")
#a.beep()

###

test_rate=0.7
mfcc_train=dataX[0:(int)(len(dataX)*test_rate),:,:]
out_info_train=out_info_label_scripts[0:(int)(len(out_info_label_scripts)*test_rate),:]
tds_train=dataY[0:(int)(len(dataY)*test_rate),:]

batch_size=500
total_data_length=len(dataX)
num_batch=(int)(total_data_length/batch_size)

mfcc_train_array=[]
out_info_train_array=[]
tds_train_array=[]


for i in range(num_batch):
    mfcc_train_array.append(dataX[i*batch_size:(i+1)*batch_size])
    out_info_train_array.append(out_info_label_scripts[i*batch_size:(i+1)*batch_size])
    tds_train_array.append(dataY[i*batch_size:(i+1)*batch_size])

print(num_batch)

mfcc_test=dataX[(int)(len(dataX)*test_rate):len(dataX)-1,:,:]
out_info_test=out_info_label_scripts[(int)(len(out_info_label_scripts)*test_rate):len(out_info_label_scripts)-1,:]
tds_test=dataY[(int)(len(dataY)*test_rate):len(dataY)-1,:]

print(dataX.shape)
print(dataY.shape)
###
sess=tf.Session()
rnn_model=a.RNN_model(sess,'rnn')
fc_model=a.FC_model(sess,'m1')
sess.run(tf.global_variables_initializer())

train_range=(int)(num_batch*0.7)

for i in range(101):
    avg_cost=0
    start=time.time()
    for j in range(train_range):
        cost,_=rnn_model.train(mfcc_train_array[j],tds_train_array[j],len(tds_train_array[j]))
        avg_cost+=cost
    traintime=time.time()-start
    avg_cost=avg_cost/(train_range)
    print('step',i)
    print('cost1: ',avg_cost)
    print('time',traintime)\

print('train run')
for i in range(train_range):
    ac1,ac2=rnn_model.accuracy(mfcc_train_array[i],tds_train_array[i])
    print(a.file.Return_Num_1(np.reshape(rnn_model.predict(mfcc_train_array[i]),[-1])))
    print(a.file.Return_Num_1(np.reshape(tds_train_array[i],[-1])))
    print(ac1)
    print(ac2)
    print(' ')

print('test run')
for j in range(train_range,num_batch):
    ac1,ac2=rnn_model.accuracy(mfcc_train_array[j],tds_train_array[j])
    print(a.file.Return_Num_1(np.reshape(rnn_model.predict(mfcc_train_array[j]),[-1])))
    print(a.file.Return_Num_1(np.reshape(tds_train_array[j],[-1])))
    print(ac1)
    print(ac2)
    print(' ')
a.beep()

"""
for i in range(51):
    avg_cost=0
    total_batch=int(num_exampes/num_batches)
    start=time.time()
    
    for j in range(1,3):
        cost1,_=fc_model.train(mfcc_train_fc1,out_info_train_fc1,len(mfcc_train_fc1))   
        cost2,_=fc_model.train(mfcc_train_fc2,out_info_train_fc2,len(mfcc_train_fc2)) 
    traintime=time.time()-start
    print('step',i)
    print('cost1: ',cost1,'cost2: ',cost2)
    print('time',traintime)

print('train1 run')
ac1,ac2=fc_model.accuracy(mfcc_train_fc1,tds_train_fc1)
print(a.file.Return_Num_1(np.reshape(fc_model.predict(mfcc_train_fc1),[-1])))
print(a.file.Return_Num_1(np.reshape(tds_train_fc1,[-1])))
print(ac1)
print(ac2)

print('train2 run')
ac1,ac2=fc_model.accuracy(mfcc_train_fc2,tds_train_fc2)
print(a.file.Return_Num_1(np.reshape(fc_model.predict(mfcc_train_fc2),[-1])))
print(a.file.Return_Num_1(np.reshape(tds_train_fc2,[-1])))
print(ac1)
print(ac2)

print('test run')
ac1,ac2=fc_model.accuracy(mfcc_test_fc,tds_test_fc)
print(a.file.Return_Num_1(np.reshape(fc_model.predict(mfcc_test_fc),[-1])))
print(a.file.Return_Num_1(np.reshape(tds_test_fc,[-1])))
print(ac1)
print(ac2)
"""

import tensorflow as tf
import numpy as np
import a
import scipy.io.wavfile as wav
import time



Data=a.file.Extract_Mfcc_OneFile('Bdb001.interaction.wav')
print(Data)
length_Data=(int)(len(Data)/10)
TrainingData=np.ndarray((length_Data,130))

for i in range(0,length_Data):
    temp=Data[i*10:i*10+10]
    temp=np.reshape(temp,(1,-1))
    print(i)
    TrainingData[i]=temp

print(TrainingData.shape);

label_scripts=np.loadtxt('Label.txt',dtype=int)
print(label_scripts)
print(label_scripts.shape)
label_scripts=np.reshape(label_scripts,(-1,1))



seq_length=30
data_length=len(TrainingData)-seq_length

dataX=np.ndarray((data_length,30,130))
dataY=np.ndarray((data_length,30,1))
print(dataY)
for i in range(0,data_length):
    dataX[i]=TrainingData[i:i+seq_length]    
    dataY[i]=label_scripts[i:i+seq_length]
out_info_label_scripts=tf.Session().run(tf.one_hot(dataY,a.no_output))
out_info_label_scripts=np.reshape(out_info_label_scripts,(-1,seq_length,a.no_output))
dataY=np.reshape(dataY,(-1,seq_length))

print(dataY)
print(dataY.shape)
print("data ready")
"""
###################
#훈련용 데이터 로드
###################X_for_Train
mfcc=a.file.extract_mfcc_for_fc(0,600,'gilmore/data')
print(mfcc)
print(mfcc.shape)
print(len(mfcc))
###################Y_for_Train
tds=a.file.open_speakerinfo('gilmore/wave_info')
tds=np.reshape(tds,(-1,1))
out_info=tf.Session().run(tf.one_hot(tds,a.no_output))
out_info=np.reshape(out_info,(-1,a.no_output))
###################
"""
"""
test_rate=0.7
mfcc_train=mfcc[0:(int)(len(mfcc)*test_rate)]
out_info_train=out_info[0:(int)(len(out_info)*test_rate),:]
tds_train=tds[0:(int)(len(out_info)*test_rate),:]

mfcc_test=mfcc[(int)(len(mfcc)*test_rate):len(mfcc)-1,:,:]
out_info_test=out_info[(int)(len(out_info)*test_rate):len(out_info)-1,:]
tds_test=tds[(int)(len(tds)*test_rate):len(tds)-1,:]
"""
###

test_rate=0.7
mfcc_train=dataX[0:(int)(len(dataX)*test_rate),:,:]
out_info_train=out_info_label_scripts[0:(int)(len(out_info_label_scripts)*test_rate),:]
tds_train=dataY[0:(int)(len(dataY)*test_rate),:]


mfcc_test=TrainingData[(int)(len(TrainingData)*test_rate):len(TrainingData)-1,:]
out_info_test=out_info_label_scripts[(int)(len(out_info_label_scripts)*test_rate):len(out_info_label_scripts)-1,:]
tds_test=dataY[(int)(len(dataY)*test_rate):len(dataY)-1,:]

#
sess=tf.Session()
fc_model=a.RNN_model(sess,'m1')
sess.run(tf.global_variables_initializer())

###
print(mfcc_train.shape)
print(out_info_train.shape)
print(tds_train.shape)
print(len(mfcc_train))
for i in range(101):
    start=time.time()
    l,_=fc_model.train(mfcc_train,tds_train,len(mfcc_train))
    traintime=time.time()-start
    print('step',i)
    print('loss',l)
    print('time',traintime)
    

    if(i%10==0):
        print(fc_model.predict(mfcc_train))
        
    
print(fc_model.accuracy(mfcc_train,out_info_train))
print(fc_model.accuracy(mfcc_test,out_info_test))
print(fc_model.accuracy(TrainingData,out_info_label_scripts))

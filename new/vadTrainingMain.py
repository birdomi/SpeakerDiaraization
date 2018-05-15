# -*- coding: utf-8 -*-
import tensorflow as tf
import model
import numpy as np
import os
import sys
from utils import *
from plot_graph import plot_static
import gc

batch_size=1500
shuffle_buffer_size=10000000

train_rate=0.8
path='CALLHOME'
mfcc_path=os.listdir(path+'/'+'Signals')
label_path=os.listdir(path+'/'+'transcripts')

#####
mfcc_train=[]
label_train=[]
mfcc_test=[]
label_test=[]
print('trainSet Loading')
for i in range(int(train_rate*len(mfcc_path))):    
    lb,s,e=model.CALLHOME_Data().return_Labeling(path+'/'+'transcripts'+'/'+label_path[i])
    mf=model.CALLHOME_Data().Get_Data(path+'/'+'Signals'+'/'+mfcc_path[i],s,e)
    print(mf.shape,lb.shape)
    if(mf.shape[0]==lb.shape[0]):
        mfcc_train=np.append(mfcc_train,mf)
        label_train=np.append(label_train,lb)
    else:
        print('error: ',mfcc_path[i],label_path[i])
print('trainSet Loading Finished')
print('testSet Loading')
for i in range(int(train_rate*len(mfcc_path)),len(mfcc_path)):    
    lb,s,e=model.CALLHOME_Data().return_Labeling(path+'/'+'transcripts'+'/'+label_path[i])
    mf=model.CALLHOME_Data().Get_Data(path+'/'+'Signals'+'/'+mfcc_path[i],s,e)
    print(mf.shape)
    mfcc_test=np.append(mfcc_test,mf)
    label_test=np.append(label_test,lb)
print('testSet Loading Finished')
####
mfcc_train=np.reshape(mfcc_train,(-1,300,39))
label_train=np.reshape(label_train,(-1,300,1))
mfcc_test=np.reshape(mfcc_test,(-1,300,39))
label_test=np.reshape(label_test,(-1,300,1))
trainNumber=mfcc_train.shape[0]
testNumber=mfcc_test.shape[0]
print('훈련 데이터셋 :',mfcc_train.shape,label_train.shape)
print('평가 데이터셋 :',mfcc_test.shape,label_test.shape)
#####
print('shuffle Train')
mfcc_train=np.reshape(mfcc_train,(-1,300,39))
label_train=np.reshape(label_train,(-1,300,1))
trainSet=np.append(mfcc_train,label_train,2)
mfcc_train=None;label_train=None;
gc.collect()

print('shuffle Test')
testSet=np.append(mfcc_test,label_test,2)
mfcc_test=None;label_test=None;
gc.collect()
#####

def runTrain(sess,rnn,msg):
    sess.run(tf.global_variables_initializer())
    
    experiment='{}_{}'.format(rnn.name,Time.now())
    model_path="./model/{}.ckpt".format(experiment)
    log_path="SAVE_Logs/{}.txt".format(experiment)
    stat_path="SAVE_Logs/{}.stat".format(experiment)
        
    logger=Logger(log_path)
    stat={"tests":0}    

    num_epoch=50
    totalTime=Time()
    for curr_epoch in range(0,num_epoch):
        #
        np.random.shuffle(trainSet)
        np.random.shuffle(testSet)       
        #
        train_acc_sum=0
        test_acc_sum=0
        trian_cost_sum=0
        test_cost_sum=0
        train_count=0
        test_count=0
        trainTime=Time()        
        
        for i in range(int(trainNumber/batch_size)):
            x=trainSet[i*batch_size:(i+1)*batch_size,:,:-1]
            y=np.array(trainSet[i*batch_size:(i+1)*batch_size,:,[-1]],int)
            #print(x[0],y[0])
            #print(x.shape,y.shape)
            rnn.Train(x,y)   
            trian_cost_sum+=rnn.Cost(x,y)
            train_count+=1        
        
        avg_cost=trian_cost_sum/train_count


        for i in range(int(trainNumber/batch_size)):
            x=trainSet[i*batch_size:(i+1)*batch_size,:,:-1]
            y=np.array(trainSet[i*batch_size:(i+1)*batch_size,:,[-1]],int)
            acc0,acc1=rnn.Accuracy(x,y)
            train_acc_sum+=(acc0+acc1)/2

        avg_train_accuracy= train_acc_sum/train_count           

        for i in range(int(testNumber/batch_size)):
            x=testSet[i*batch_size:(i+1)*batch_size,:,:-1]
            y=np.array(testSet[i*batch_size:(i+1)*batch_size,:,[-1]],int)
            acc0,acc1=rnn.Accuracy(x,y)                
            test_cost_sum+=rnn.Cost(x,y)
            test_acc_sum+=(acc0+acc1)/2
            """
            for i in range(len(y)):
                print(rnn.Outputs(x)[i])
                print(y[i])
            """
            test_count+=1

        avg_test_accuracy=test_acc_sum/test_count
        avg_test_cost=test_cost_sum/test_count
        
        if(avg_test_accuracy>stat["tests"]):
            stat['tests']=avg_test_accuracy
            stat['trains']=avg_train_accuracy
            stat['epoch']=curr_epoch
            stat['cost']=avg_cost
            stat['traincost']=avg_test_cost
            rnn.Save(model_path)
    
        log="Epoch {}/{}, l_rate:{:.3f}, cost = {:>7.4f},train cost={:>7.4f}, accracy(train,test/best):({:.4f}, {:.4f}/{:.4f}),time = {}/{}\n".format(
        		    curr_epoch, num_epoch, rnn.learning_rate,avg_cost,avg_test_cost,
        			avg_train_accuracy,avg_test_accuracy,stat['tests'],trainTime.duration(), totalTime.duration())
        logger.write(log)
    summary ="""
    {}.{}.{}
            learning_rate : {} train_data_ratio : {}  num_epoch : {}	
            Best evaulation based on test_data  :  Accuracy_train  : {}    Accuracy_test :  {}  at epoch :{}
            \n
            """.format(
        	rnn.name,experiment,msg,
        	rnn.learning_rate, train_rate, num_epoch,	
        					stat['trains'],stat['tests'],stat['epoch'])
    print(summary)    
    logger.flush()
    logger.close()        
    plot_static(log_path)

    with open("SAVE_Logs/log.txt","a") as f:
        f.write(summary)
#########
#실행
sess=tf.Session()
VAD=model.RNN_Model(sess,'VAD',0.001)
runTrain(sess,VAD,'VAD')
tf.reset_default_graph()
"""
sess=tf.Session()
rnn=model.RNN_Model(sess,'RNN_',0.001)
sess.run(tf.global_variables_initializer())
rnn.Restore('VADM/VAD')#application_path+
rnn.Show_Reuslt(d._MFCC,borderFile,threshold,mdt)
#rnn.Show_Reuslt(d._MFCC,borderFile,threshold,mdt)
"""

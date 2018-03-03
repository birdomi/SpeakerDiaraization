import tensorflow as tf
import numpy as np
import a
import scipy.io.wavfile as wav
import time
import random
import os
from utils import Logger
from utils import Time
from plot_graph import plot_static

mfcc_path=os.listdir('Signals')
label_path=os.listdir('transcripts')

wavPath='Signals'
labPath='transcripts'
mfcc_path=os.listdir(wavPath)
label_path=os.listdir(labPath)

datanum=10

train_rate=0.8
trainRange=range(0,int(datanum*train_rate))
testRange=range(int(datanum*train_rate),datanum)

dN={}

for i in range(datanum):
    dN.update({i:a.ICSI_Data()})
    dN[i].MakeData(wavPath+'/'+mfcc_path[i],labPath+'/'+label_path[i],'N')

dH={}

for i in range(datanum):
    dH.update({i:a.ICSI_Data()})
    dH[i].MakeData(wavPath+'/'+mfcc_path[i],labPath+'/'+label_path[i],'On')

def matrixAccuracy(matrix):
    number_0=matrix[0][0]+matrix[0][1]
    number_1=matrix[1][0]+matrix[1][1]

    acc0_0=matrix[0][0]/number_0
    acc0_1=matrix[0][1]/number_0
    acc1_0=matrix[1][0]/number_1
    acc1_1=matrix[1][1]/number_1

    accMatrix=[[acc0_0,acc0_1],
               [acc1_0,acc1_1]]
    
    return accMatrix

def runTrain(sess,d,rnn,msg):
    sess.run(tf.global_variables_initializer()) 
    
    experiment='{}_{}_{}'.format(rnn.name,datanum,Time.now())
    model_path="model/{}".format(experiment)
    log_path="SAVE_Logs/{}.txt".format(experiment)
    stat_path="SAVE_Logs/{}.stat".format(experiment)
        
    logger=Logger(log_path)
    stat={"tests":0}
    stat_lowAbs={"dist":100}
        
    total_number_of_batch=0
    for number in trainRange:
        total_number_of_batch+=d[number].numberBatch
        
    total_number_of_batch_test=0
    for number in testRange:
        total_number_of_batch_test+=d[number].numberBatch
            
    num_epoch=100
    totalTime=Time()
    for curr_epoch in range(0,num_epoch):
        cost_sum=0
        test_cost_sum=0
        trainTime=Time()
        for number in trainRange:
            for index in range(d[number].numberBatch):
                cost,_=rnn.Train(d[number]._MFCC[index],d[number]._LABEL[index],0.8)
                cost_sum+=cost
        
        avg_cost=cost_sum/total_number_of_batch    
        acc1=0.0
        acc0=0.0
        for number in trainRange:
            for index in range(d[number].numberBatch):
                ac1,ac0=rnn.Accuracy(d[number]._MFCC[index],d[number]._LABEL[index])            
                acc1+=ac1
                acc0+=ac0
        avg_train_accuracy= (acc1/total_number_of_batch+acc0/total_number_of_batch)/2
       
        acc1=0.0
        acc0=0.0
        test_cost_sum=0
        resultMatrix=np.zeros([2,2],int)
        for number in testRange:
            for index in range(d[number].numberBatch):
                ac1,ac0=rnn.Accuracy(d[number]._MFCC[index],d[number]._LABEL[index])
                test_cost_sum+=rnn.Cost(d[number]._MFCC[index],d[number]._LABEL[index])
                resultMatrix+=rnn.return_ResultMatrix(d[number]._MFCC[index],d[number]._LABEL[index])
                acc1+=ac1
                acc0+=ac0
        avg_test_accuracy= (acc1/total_number_of_batch_test+acc0/total_number_of_batch_test)/2
        test_distance=np.abs(acc1/total_number_of_batch_test-acc0/total_number_of_batch_test)
        avg_test_cost=test_cost_sum/total_number_of_batch_test
        
        if(avg_test_accuracy>stat["tests"]):
            stat['tests']=avg_test_accuracy
            stat['trains']=avg_train_accuracy
            stat['epoch']=curr_epoch
            stat['cost']=avg_cost
            stat['traincost']=avg_test_cost
            stat['resultMatrix']=resultMatrix
            stat['dist']=test_distance
            rnn.Save(model_path)
    
        if(test_distance<stat_lowAbs['dist']):
            stat_lowAbs['tests']=avg_test_accuracy
            stat_lowAbs['trains']=avg_train_accuracy
            stat_lowAbs['epoch']=curr_epoch
            stat_lowAbs['cost']=avg_cost
            stat_lowAbs['traincost']=avg_test_cost
            stat_lowAbs['resultMatrix']=resultMatrix
            stat_lowAbs['dist']=test_distance
            rnn.Save(model_path+'lowdist')
    
        log="Epoch {}/{}, l_rate:{:.10f}, cost = {:>7.4f},train cost={:>7.4f}, accracy(train,test/best):({:.4f}, {:.4f}/{:.4f}), test_distance ={:.4f} ,time = {}/{}\n".format(
        		    curr_epoch, num_epoch, rnn.learning_rate,avg_cost,avg_test_cost,
        			avg_train_accuracy,avg_test_accuracy,stat['tests'],test_distance ,trainTime.duration(), totalTime.duration())
        logger.write(log)
    summary ="""
    {}.{}.{}
            learning_rate : {} train_data_ratio : {}  num_epoch : {}  batch_size : {}   windowsize : {} windowshift : {}		
            Best evaulation based on test_data  :  Accuracy_train  : {}    Accuracy_test :  {}  at epoch :{}
            Best evaulation based on test_data at lowest distance : Accuracy_train  : {}    Accuracy_test :  {} at epoch :{} \n
            best Result Matrix : \n{}{}\n
            best Reuslt Matrix at lowest distance : \n{}{}\n
            """.format(
        	rnn.name,experiment,msg,
        	rnn.learning_rate, train_rate, num_epoch,a.batch_size,a.windowsize,a.windowstep,		
        					stat["trains"],stat["tests"],stat['epoch'],stat_lowAbs['trains'],stat_lowAbs['tests'],stat_lowAbs['epoch'],
                            stat['resultMatrix'],matrixAccuracy(stat['resultMatrix']),stat_lowAbs['resultMatrix'],matrixAccuracy(stat_lowAbs['resultMatrix']))
    print(summary)
    logger.flush()
    logger.close()  
        
    plot_static(log_path)

    with open("SAVE_Logs/log.txt","a") as f:
        f.write(summary)

sess=tf.Session()
rnn=a.RNN_Model_usingAll(sess,'RNNAll_NoHamming',0.001)
runTrain(sess,dN,rnn,rnn.name)
tf.reset_default_graph()
  
sess=tf.Session()
rnn=a.RNN_Model_usingAll(sess,'RNNAll_Hamming',0.001)
runTrain(sess,dH,rnn,rnn.name)
tf.reset_default_graph()

sess=tf.Session()
rnn=a.RNN_Model_usingMiddle(sess,'RNNMiddle_NoHamming',0.001)
runTrain(sess,dN,rnn,rnn.name)
tf.reset_default_graph()
  
sess=tf.Session()
rnn=a.RNN_Model_usingMiddle(sess,'RNNMiddle_Hamming',0.001)
runTrain(sess,dH,rnn,rnn.name)
tf.reset_default_graph()

sess=tf.Session()
rnn=a.RNN_Model_usingBoundary(sess,'RNNBoundary_NoHamming',0.001)
sess.run(tf.global_variables_initializer())
runTrain(sess,dN,rnn,rnn.name)
tf.reset_default_graph()

sess=tf.Session()
rnn=a.RNN_Model_usingBoundary(sess,'RNNBoundary_Hamming',0.001)
runTrain(sess,dH,rnn,rnn.name)
tf.reset_default_graph()

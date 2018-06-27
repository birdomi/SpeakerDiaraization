import tensorflow as tf
import numpy as np
import re
import os
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import fbank
import scipy.io.wavfile as wav
import pydub
import librosa
from tensorflow.contrib import rnn
import sys

fsize=0.5
fstep=0.5

batch_size=1500
windowstep=50
windowmul=6
windowsize=windowstep*windowmul

def progressBar(value, endvalue, bar_length=20):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

class INPUT_Data():
    def Get_Data(self, mfcc_path):
        wavefile=pydub.AudioSegment.from_wav(mfcc_path)
        wavefile=wavefile.set_frame_rate(16000)
        wavefile=wavefile.set_channels(1)
        wavefile.export('waveTemp.wav',format='wav')

        self.__sr,self.__audio=wav.read('waveTemp.wav')
        self.__mfcc = mfcc(self.__audio,self.__sr,numcep=12,appendEnergy=False)
        self.__n,self.__mfcc_energy=fbank(self.__audio,self.__sr)
        self.__mfcc_energy=np.reshape(self.__mfcc_energy,(-1,1))
        self.__mfcc=np.append(self.__mfcc,self.__mfcc_energy,axis=1)

        self.__mfcc1=d_mfcc_feat = delta(self.__mfcc, 2)
        self.__mfcc2=d_mfcc_feat = delta(self.__mfcc, 3)
        self.__mfcc=np.append(self.__mfcc,self.__mfcc1,axis=1)
        self.__mfcc=np.append(self.__mfcc,self.__mfcc2,axis=1)
        self.__Tmfcc=self.__mfcc
        #self.__Tmfcc=sklearn.preprocessing.normalize(self.__Tmfcc)
        self.__data_length=int(len(self.__Tmfcc)/windowstep)
        self.__num_seq=self.__data_length-windowmul+1
        
        self.__tmp=[]
        for i in range(self.__num_seq):
            self.__tempSequence=self.__Tmfcc[i*windowstep:(i+windowmul)*windowstep]            
            #self.__tempSequence=np.multiply(self.__tempSequence,self.__hamming)
            self.__tmp.append(self.__tempSequence)

        self.__tmp=np.reshape(self.__tmp,(-1,300,39))        
        self.__batch_num=int(len(self.__tmp)/batch_size)
        self._MFCC=self.__tmp    
        return self._MFCC

    

class Model():
    def __init__(self,sess,name,learning_rate):
        self.sess=sess
        self.name=name
        self.learning_rate=learning_rate
        self._build_net()

    def _build_net(self):
        pass
    
    def Outputs(self,x,keep_prob=1.0):
        return self.sess.run(self.outputs,feed_dict={self.X:x,self.keep_prob:keep_prob})

    def Restore(self,name):
        saver=tf.train.Saver()
        saver.restore(self.sess,name)

class VAD_Model(Model):
    def _build_net(self):
        self.LSTM_layer1=35
        self.LSTM_layer2=30

        self.FC_layer1=50
        self.FC_layer2=40
        self.FC_layerOut=2

        self.X=tf.placeholder(tf.float32,[None,windowsize,39])
        self.Y=tf.placeholder(tf.int32,[None,windowsize,1])
        self.keep_prob=tf.placeholder(tf.float32)       

        self.RNN1_cell=tf.contrib.rnn.LSTMCell(self.LSTM_layer1,activation=tf.nn.tanh)
        self.RNN1_cell=tf.contrib.rnn.DropoutWrapper(self.RNN1_cell,output_keep_prob=self.keep_prob)

        self.RNN2_cell=tf.contrib.rnn.LSTMCell(self.LSTM_layer2,activation=tf.nn.tanh)
        self.RNN2_cell=tf.contrib.rnn.DropoutWrapper(self.RNN2_cell,output_keep_prob=self.keep_prob)

        self.cell_fw=tf.contrib.rnn.MultiRNNCell([self.RNN1_cell,self.RNN2_cell])
        self.cell_bw=tf.contrib.rnn.MultiRNNCell([self.RNN1_cell,self.RNN2_cell])
        self.RNN_output,_=tf.nn.bidirectional_dynamic_rnn(self.cell_fw,self.cell_bw,self.X,dtype=tf.float32)
        self.rnn_output=tf.concat(self.RNN_output,2)       

        self.fc_input=tf.reshape(self.rnn_output,[-1,self.LSTM_layer2*2])

        self.FC1=tf.contrib.layers.fully_connected(self.fc_input,self.FC_layer1)
        self.FC2=tf.contrib.layers.fully_connected(self.FC1,self.FC_layer2)
        self.FC1_out=tf.contrib.layers.fully_connected(self.FC2,self.FC_layerOut,activation_fn=None)
        
        self.FC1_out=tf.reshape(self.FC1_out,(-1,windowsize,2))

        self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.FC1_out,labels=tf.one_hot(self.Y,2)))
        
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
        self.outputs=tf.argmax(self.FC1_out,2)

    def Show_Shape(self,x,y,keep_prob=1.0):
        print(self.sess.run(tf.shape(self.RNN_output),feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))
        print(self.sess.run(tf.shape(self.__rnn_first),feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))
        print(self.sess.run(tf.shape(self.__rnnout),feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))
        print(self.sess.run(tf.shape(self.fc_input),feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))
        print(self.sess.run(self.outputs,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))

    def Show_Reuslt(self,x,saveFile,threshold=0.5,detectLength=0.1):
        self.__output=self.Outputs(x)
        self.__output=np.reshape(self.__output,(-1,1))
        self.shouldLongerThanThis=detectLength
        
        self.__outputArray=[];t=[]
        for i in range(int(len(self.__output)/windowsize)-windowmul):
            for j in range(windowstep):
                t.append(i*50+j)
            first_windowStart=windowsize*i
            outputArray=np.zeros((windowstep,1))
            for j in range(windowmul):          
                window=first_windowStart+windowsize*(j+1)
                windowStart=window-windowstep*(j+1)
                windowEnd=window-windowstep*j
                a1=self.__output[windowStart:windowEnd]
                outputArray+=a1
            self.__outputArray.append(outputArray/windowmul)

        self.__outputArray=np.reshape(self.__outputArray,(-1))
                
        self.currActivity=1

        f_short=open(saveFile+'_v1.bdr','w')        
        f_long=open(saveFile+'_v2.bdr','w')
        self.time=[]
        for i in range(len(self.__outputArray)):
            if(self.currActivity==0):
                if(self.__outputArray[i]>threshold):
                    self.time.append("{:.2f}".format(2.5+0.01*i))
                    self.currActivity=1
            if(self.currActivity==1):
                if(self.__outputArray[i]<threshold):
                    self.time.append("{:.2f}".format(2.5+0.01*i))
                    self.currActivity=0       

        
        for i in range(int(len(self.time)/2)):
            self.start=self.time[i*2]
            self.end=self.time[i*2+1]
            f_short.write(self.start+' '+self.end+'\n')
            if((float(self.end)-float(self.start))>=self.shouldLongerThanThis):
                print(self.start,self.end)
                f_long.write(self.start+' '+self.end+'\n')
        f_long.close()
        f_short.close()

    def Make_Segment(self,x,threshold=0.5,detectLength=0.1):
        self.__output=self.Outputs(x)
        self.__output=np.reshape(self.__output,(-1,1))
        self.shouldLongerThanThis=detectLength
        
        self.__outputArray=[];t=[]
        for i in range(int(len(self.__output)/windowsize)-windowmul):
            for j in range(windowstep):
                t.append(i*50+j)
            first_windowStart=windowsize*i
            outputArray=np.zeros((windowstep,1))
            for j in range(windowmul):          
                window=first_windowStart+windowsize*(j+1)
                windowStart=window-windowstep*(j+1)
                windowEnd=window-windowstep*j
                a1=self.__output[windowStart:windowEnd]
                outputArray+=a1
            self.__outputArray.append(outputArray/windowmul)

        self.__outputArray=np.reshape(self.__outputArray,(-1))
                
        self.currActivity=1
        
        self.time=[]
        self.startTime=[]
        self.endTime=[]
        for i in range(len(self.__outputArray)):
            if(self.currActivity==0):
                if(self.__outputArray[i]>threshold):
                    self.time.append("{:.2f}".format(2.5+0.01*i))
                    self.currActivity=1
            if(self.currActivity==1):
                if(self.__outputArray[i]<threshold):
                    self.time.append("{:.2f}".format(2.5+0.01*i))
                    self.currActivity=0       

        for i in range(int(len(self.time)/2)):
            self.start=self.time[i*2]
            self.end=self.time[i*2+1]
            if(float(self.end)-float(self.start)>=self.shouldLongerThanThis):
                self.startTime.append(self.end)
                self.endTime.append(self.start)

        print('###################')
        self.frameListArray=[]
        for i in range(1,len(self.startTime)-1):
            print(self.startTime[i-1],self.endTime[i])
            self.frameListArray.append(str(self.startTime[i-1])+'s-'+str(self.endTime[i])+'s,')
        return self.frameListArray

    def voiceTimeSegment(self,x,threshold=0.5,detectLength=0.1):
        self.__output=self.Outputs(x)
        self.__output=np.reshape(self.__output,(-1,1))
        self.shouldLongerThanThis=detectLength
        
        self.__outputArray=[];t=[]
        for i in range(int(len(self.__output)/windowsize)-windowmul):
            for j in range(windowstep):
                t.append(i*50+j)
            first_windowStart=windowsize*i
            outputArray=np.zeros((windowstep,1))
            for j in range(windowmul):          
                window=first_windowStart+windowsize*(j+1)
                windowStart=window-windowstep*(j+1)
                windowEnd=window-windowstep*j
                a1=self.__output[windowStart:windowEnd]
                outputArray+=a1
            self.__outputArray.append(outputArray/windowmul)

        self.__outputArray=np.reshape(self.__outputArray,(-1))
                
        self.currActivity=1
        
        self.time=[]
        self.startTime=[]
        self.endTime=[]
        for i in range(len(self.__outputArray)):
            if(self.currActivity==0):
                if(self.__outputArray[i]>threshold):
                    self.time.append("{:.2f}".format(2.5+0.01*i))
                    self.currActivity=1
            if(self.currActivity==1):
                if(self.__outputArray[i]<threshold):
                    self.time.append("{:.2f}".format(2.5+0.01*i))
                    self.currActivity=0       

        for i in range(int(len(self.time)/2)):
            self.start=self.time[i*2]
            self.end=self.time[i*2+1]
            if(float(self.end)-float(self.start)>=self.shouldLongerThanThis):
                self.startTime.append(self.end)
                self.endTime.append(self.start)

        print('###################')
        self.frameListArray=[]
        for i in range(1,len(self.startTime)-1):
            if(float(self.endTime[i])-float(self.startTime[i-1])>0.5):
                #print(float(self.startTime[i-1]),float(self.endTime[i]))
                self.frameListArray.append((float(self.startTime[i-1]),float(self.endTime[i])))
        return self.frameListArray
        
class VoiceFeature():
    def __init__(self,sess,model):
        self.SR=BILSTM_CNN_SR(sess,'sr',0,1166)
        self.SR.Restore(model)
        self.out=2048

    def Extract_timeSegment(self,audioDir,timeSegment):
        audio,sr=librosa.audio.load(audioDir,sr=16000)
        spec=librosa.feature.melspectrogram(y=audio,sr=sr,hop_length=160,n_fft=1024)
        spec=librosa.power_to_db(spec,ref=np.max)
        spec=spec.T

        r=np.reshape([],(0,self.out))

        timeSegment=[s for s in timeSegment]
        cut_per_segment=[]

        percentLength=len(timeSegment)+1
        percent=1
        progressBar(percent,percentLength,50)
        for s in timeSegment:
            specSegment=spec[int(s[0]*100):int(s[1]*100)+1]
            specLength=len(specSegment)
            cutCount=int((specLength-25)/25)
            #print(cutCount,specLength,(specLength-25)/25,s[0],s[1])
            cut_per_segment.append(cutCount)
            
            if(cutCount>0):
                x=[]                    
                for i in range(cutCount):
                    x.append(specSegment[i*25:50+i*25])
                x=np.reshape(x,(-1,50,128,1))
                x=self.SR.Feature(x)           
                x=np.reshape(x,(-1,self.out))            
                r=np.append(r,x,axis=0)

                percent+=1
                progressBar(percent,percentLength,50)

        r=np.reshape(r,(-1,self.out))
        #print(r.shape)
        #print("####추출된 벡터 개수, 컷 카운트 개수: ",r.shape,np.sum(cut_per_segment))
        return r,timeSegment,cut_per_segment
        
########################
class BILSTM_CNN_SR():
    def __init__(self,sess,name,learning_rate,speakerNo):
        self.sess=sess
        self.name=name
        self.learning_rate=learning_rate
        self.speakerNo=speakerNo
        self._build_net()

    def _build_net(self):
        audiocut_max=50
        LSTM1=64
        LSTM2=64
        LSTM3=64
        
        FC1=2048
        FC_Out=self.speakerNo

        self.X=tf.placeholder(tf.float32,[None,audiocut_max,128,1])
        self.Y=tf.placeholder(tf.int64,[None,1])
        
       
        #self.X1=tf.reshape(self.X,[-1,audiocut_max,128])
        self.keep_prob=tf.placeholder(tf.float32)  
       
        self.L1 = tf.layers.conv2d(inputs=self.X, filters=8, kernel_size=5,padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())            
        self.L2 = tf.layers.conv2d(inputs=self.L1, filters=8, kernel_size=5,padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.L3 = tf.layers.average_pooling2d(self.L2,2,2) 

        self.L4 = tf.layers.conv2d(inputs=self.L3, filters=32, kernel_size=32,padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.L5 = tf.layers.conv2d(inputs=self.L4, filters=32, kernel_size=32,padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        #self.L5 = tf.layers.max_pooling2d(self.L5,2,2)
       
        self.Conv_out=tf.reshape(self.L5,[-1,25,64*32])  
        
        with tf.variable_scope('rnn2'):
            self.RNN3_cell=rnn.LSTMCell(LSTM3)
            self.RNN3_cell=rnn.DropoutWrapper(self.RNN3_cell,1.0,self.keep_prob)
            self.cell_fwb=rnn.MultiRNNCell([self.RNN3_cell])
            self.cell_bwb=rnn.MultiRNNCell([self.RNN3_cell])
    
            self.RNN2_output,_=tf.nn.bidirectional_dynamic_rnn(self.cell_fwb,self.cell_bwb,self.Conv_out,dtype=tf.float32)
            self.RNN2_output=tf.concat(self.RNN2_output,2)
                        
            self.RNN2_output=tf.split(axis=1,value=self.RNN2_output,num_or_size_splits=25)
            self.RNN2_output=tf.concat([self.RNN2_output[-1],self.RNN2_output[0]],1)

            self.RNN2_Out=tf.layers.flatten(self.RNN2_output)
        
        self.FC=tf.layers.dense(self.RNN2_Out,FC1,activation=tf.nn.relu)
        self.FC=tf.layers.dropout(self.FC,self.keep_prob)
        self.FC_out=tf.layers.dense(self.FC,FC_Out)

        self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.FC_out,labels=tf.one_hot(self.Y,FC_Out)))
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.output=tf.argmax(self.FC_out,1)
        self.YOUT=tf.reshape(self.Y,[-1])
        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.output,tf.int32),tf.cast(self.YOUT,tf.int32)),tf.float32))

    def printConv(self,x,keep_prob=1.0):
        return self.sess.run([self.L1,self.L2,self.L3,self.L4],feed_dict={self.X:x,self.keep_prob:keep_prob})
            
    def Outputs(self,x,keep_prob=1.0):
        return self.sess.run(self.output,feed_dict={self.X:x,self.keep_prob:keep_prob})

    def Cost(self,x,y,keep_prob=1.0):
        return self.sess.run(self.cost,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})

    def Train(self,x,y,keep_prob=0.4):
        return self.sess.run(self.optimizer,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})

    def Accuracy(self,x,y,keep_prob=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})

    def Feature(self,x,keep_prob=1.0):
        return self.sess.run(self.FC,feed_dict={self.X:x,self.keep_prob:keep_prob})

    def Save(self,name):
        saver=tf.train.Saver(tf.trainable_variables())
        saver.save(self.sess, name)

    def Restore(self,name):
        saver=tf.train.Saver(tf.trainable_variables())
        saver.restore(self.sess,name)

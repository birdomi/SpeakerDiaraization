import numpy as np
import tensorflow as tf
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import fbank
import scipy.io.wavfile as wav
import re
import pydub

batch_size=1500
windowstep=50
windowmul=6
windowsize=windowstep*windowmul
labelLength=int(windowsize/windowmul)
class Data():
    def __init__(self):
        self._MFCC=[]
        self._LABEL=[]

class CALLHOME_Data():
    def Get_Data(self, mfcc_path,start,end):
        wavefile=pydub.AudioSegment.from_wav(mfcc_path)
        wavefile=wavefile.set_frame_rate(16000)
        wavefile=wavefile.set_channels(1)
        wavefile.export('waveTemp.wav',format='wav')

        self.__sr,self.__audio=wav.read('waveTemp.wav')
        self.__mfcc = mfcc(self.__audio,self.__sr)
        self.__mfcc1=d_mfcc_feat = delta(self.__mfcc, 1)
        self.__mfcc2=d_mfcc_feat = delta(self.__mfcc, 2)
        self.__mfcc=np.append(self.__mfcc,self.__mfcc1,axis=1)
        self.__mfcc=np.append(self.__mfcc,self.__mfcc2,axis=1)        
        self.__Tmfcc=self.__mfcc[start:end]
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

    def return_Labeling(self,label_path):
        file=open(label_path,'r',encoding='UTF8')
        lines=file.readlines()

        search=''
        number=re.compile("\d{2,}")

        self.startTime=[];self.endTime=[];self.speaker=[]
        for line in lines:
            if('*' in line):
                curr_speaker_atTime=line[1]
            if(search in line):        
                result=number.findall(line)
                if(line[1] == 'A' or line[1]=='B'):
                    self.speaker_atTime=line[1]
                else:
                    self.speaker_atTime=curr_speaker_atTime
                self.startTime.append(int(result[0]))
                self.endTime.append(int(result[1]))
                self.speaker.append(self.speaker_atTime)

        self.__labelstartTime=self.startTime[0]
        self.__labelendtime=0
        
        for i in range(len(self.endTime)):
            if(self.__labelendtime<self.endTime[i]):
                self.__labelendtime=self.endTime[i]

        
        
        self.__secStart=np.round(self.__labelstartTime/1000)+1
        self.__secEnd=np.round(self.__labelendtime/1000)-1        


        self.arrayLength=int((self.__labelendtime-self.__labelstartTime)/10)
        self.__array=[0]*(self.arrayLength+1)
        self.Speaker=['']*(self.arrayLength+1)
        

        for i in range(len(self.startTime)):
            #print(int((self.startTime[i]-self.__labelstartTime)/10),int((self.endTime[i]-self.__labelstartTime)/10))
            for j in range(int((self.startTime[i]-self.__labelstartTime)/10),int((self.endTime[i]-self.__labelstartTime)/10)):
                self.Speaker[j]+=self.speaker[i]
                self.__array[j]=1

        self.__array=np.reshape(self.__array,(-1,1))  
        self.Speaker=np.reshape(self.Speaker,(-1,1))

        self.__data_length=int(len(self.__array)/windowstep)
        self.__num_seq=self.__data_length-windowmul+1
        
        self.__tmp=[]
        for i in range(self.__num_seq):
            self.__tempSequence=self.__array[i*windowstep:(i+windowmul)*windowstep]            
            #self.__tempSequence=np.multiply(self.__tempSequence,self.__hamming)
            self.__tmp.append(self.__tempSequence)

        self.__tmp=np.reshape(self.__tmp,(-1,300,1))        
        self.__batch_num=int(len(self.__tmp)/batch_size)
        self.__resultArray=self.__tmp        
        
        return self.__resultArray,int(self.__labelstartTime/10),int(self.__labelendtime/10)


  
class INPUT_Data(Data):
    def Get_Data(self, mfcc_path):
        wavefile=pydub.AudioSegment.from_wav(mfcc_path)
        wavefile=wavefile.set_frame_rate(16000)
        wavefile=wavefile.set_channels(1)
        wavefile.export('waveTemp.wav',format='wav')

        self.__sr,self.__audio=wav.read('waveTemp.wav')
        self.__mfcc = mfcc(self.__audio,self.__sr)
        self.__mfcc1=d_mfcc_feat = delta(self.__mfcc, 1)
        self.__mfcc2=d_mfcc_feat = delta(self.__mfcc, 2)
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

        self.__tmp=np.reshape(self.__tmp,(-1,windowsize,39))        
        self.__batch_num=int(len(self.__tmp)/batch_size)
        self._MFCC=self.__tmp
        

class Model():
    def __init__(self,sess,name,learning_rate):
        self.sess=sess
        self.name=name
        self.learning_rate=learning_rate
        self._build_net()

    def _build_net(self):
        pass

    def Cost(self,x,y,keep_prob=1.0):
        return self.sess.run(self.cost,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})

    def Train(self,x,y,keep_prob=0.8):
        return self.sess.run(self.optimizer,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})
    
    def Outputs(self,x,keep_prob=1.0):
        return self.sess.run(self.outputs,feed_dict={self.X:x,self.keep_prob:keep_prob})

    def Save(self,name):
        saver=tf.train.Saver()
        saver.save(self.sess,name)

    def Restore(self,name):
        saver=tf.train.Saver()
        saver.restore(self.sess,name)

    def return_ResultMatrix(self,x,y):
        self.x_prediction=np.reshape(self.Outputs(x),[-1])
        self.y_prediction=np.reshape(y,[-1])

        self.resultMatrix=np.zeros([2,2],int)
        for i in range(len(self.x_prediction)):
            self.resultMatrix[self.y_prediction[i]][self.x_prediction[i]]+=1

        return self.resultMatrix

    def Accuracy(self,x,y):
        self.x_prediction=np.reshape(self.Outputs(x),[-1])
        self.y_prediction=np.reshape(y,[-1])
        
        self.check_for_changePoint=0
        self.check_for_falseAlarm=0
        self.total_changePoint=0
        self.total_non_changePoint=0
        self.accuracy_changePoint=0.0
        self.accuracy_non_changePoint=0.0
        for i in range(len(self.x_prediction)):            
            if(self.y_prediction[i]==1):
                self.total_changePoint=self.total_changePoint+1
                if(self.x_prediction[i]==1):
                    self.check_for_changePoint=self.check_for_changePoint+1
            else:
                self.total_non_changePoint=self.total_non_changePoint+1
                if(self.x_prediction[i]==0):
                    self.check_for_falseAlarm=self.check_for_falseAlarm+1
        #print('1의 총 개수, 맞은 개수: ',self.total_changePoint,self.check_for_changePoint)
        #print('0의 총 개수, 맞은 개수: ',self.total_non_changePoint,self.check_for_falseAlarm) 
        if(self.total_changePoint!=0):
            self.accuracy_changePoint=(float)(self.check_for_changePoint/self.total_changePoint)
        if(self.total_non_changePoint!=0):
            self.accuracy_non_changePoint=(float)(self.check_for_falseAlarm/self.total_non_changePoint)
                
        return self.accuracy_changePoint,self.accuracy_non_changePoint

class RNN_Model(Model):
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

        self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.FC1_out,labels=tf.one_hot(self.Y,2)))
        
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
            if(float(self.end)-float(self.start)>=self.shouldLongerThanThis):
                print(self.start,self.end)
                f_long.write(self.start+' '+self.end+'\n')
        f_long.close()
        f_short.close()

        
        
        
        
        

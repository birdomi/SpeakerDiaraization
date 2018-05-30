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

fsize=0.5
fstep=0.5

batch_size=1500
windowstep=50
windowmul=6
windowsize=windowstep*windowmul



class CALLHOME_Data():
    def Get_Data(self, mfcc_path):
        self.__sr,self.__audio=wav.read(mfcc_path)
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

    def return_Labeling(self,label_path,frameSize=fsize,frameStep=fstep):
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

        #print(self.__labelstartTime,self.__labelendtime)
        
        self.__secStart=self.__labelstartTime/1000
        self.__secEnd=self.__labelendtime/1000 

        self.__start=int(self.__secStart*100-self.__labelstartTime/10)
        self.__end=int(self.__labelendtime/10-self.__secEnd*100)
        

        #print(self.__secStart,self.__secEnd)

        self.arrayLength=int((self.__labelendtime-self.__labelstartTime)/10)
        self.__array=[0]*(self.arrayLength+1)
        self.Speaker=['']*(self.arrayLength+1)

        #print(self.arrayLength)
        for i in range(len(self.startTime)):
            #print(int((self.startTime[i]-self.__labelstartTime)/10),int((self.endTime[i]-self.__labelstartTime)/10))
            for j in range(int((self.startTime[i]-self.__labelstartTime)/10),int((self.endTime[i]-self.__labelstartTime)/10)):
                self.Speaker[j]+=self.speaker[i]

            self.__array[int((self.startTime[i]-self.__labelstartTime)/10)]=1
            self.__array[int((self.endTime[i]-self.__labelstartTime)/10)]=1

        self.__array=np.reshape(self.__array[self.__start:-self.__end-1],(-1,1))  
        self.Speaker=np.reshape(self.Speaker,(-1,1))
        #print(self.__array.shape)

        self.__noFeature=int(len(self.__array)/(100*frameStep))
        #print(self.__noFeature)
        
        check0=0;check1=0
        self.__resultArray=[]
        for i in range(int(self.__noFeature-frameSize/frameStep+1)):            
            self.__1check=False
            for j in range(int(100*frameSize)):
                if(self.__array[int(i*100*frameStep+j)]==1):
                    self.__1check=True
            if(self.__1check==True):
                self.__resultArray.append(1)
                check1+=1
            else:
                self.__resultArray.append(0)
                check0+=1
        print(check1,check0)

        self.__resultArray=np.reshape(self.__resultArray,(-1,1))
        
        
        return self.__resultArray,self.Speaker,self.__secStart,self.__secEnd

    def return_TimesetData(self,feature,label):
        temp_x=[]
        temp_y=[]

        for i in range(len(feature)-windowsize):
            for j in range(windowsize):
                temp_x.append(feature[i+j])
                temp_y.append(label[i+j])
        temp_x=np.reshape(temp_x,(-1,windowsize,feature.shape[1]))
        temp_y=np.reshape(temp_y,(-1,windowsize,1))

        print(temp_x.shape,temp_y.shape)
        return temp_x,temp_y

    def SegmentGroundTruth(self,timeSegment,segmentStart,segmentEnd,speaker):
        label=[]

        segment=[s for s in timeSegment if s[0]>segmentStart and s[1]<segmentEnd]
        for i in range(len(segment)):
            start=int((segment[i][0]-segmentStart)*100)
            end=int((segment[i][1]-segmentStart)*100)

            checkA=False;checkB=False
            for j in range(start,end):
                if(self.Speaker[j]=='A'):
                      checkA=True
                if(self.Speaker[j]=='B'):
                      checkB=True
            if(checkA==True and checkB==False):
                label.append(0);
            elif(checkA==False and checkB==True):
                label.append(1);
            elif(checkA==True and checkB==True):
                label.append(2);
            else:
                label.append(3);

        
        return segment,label

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
        return self.sess.run(self.output,feed_dict={self.X:x,self.keep_prob:keep_prob})

    def Cost(self,x,y,keep_prob=1.0):
        return self.sess.run(self.cost,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})

    def Train(self,x,y,keep_prob=0.9):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})
    
    def Accuracy(self,x,y,keep_prob=1.0):
        return self.sess.run(self.accuracy,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})
    
    def Save(self,name):
        saver=tf.train.Saver()
        saver.save(self.sess, name)

    def Restore(self,name):
        saver=tf.train.Saver()
        saver.restore(self.sess,name)

    

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
            if(float(self.end)-float(self.start)>=self.shouldLongerThanThis):
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
            #test print(self.startTime[i-1],self.endTime[i])
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
                self.frameListArray.append((float(self.startTime[i-1]),float(self.endTime[i])))
        return self.frameListArray
        
class VoiceFeature():
    def __init__(self,sess,model):
        self.SR=BILSTM_CNN_SR(sess,'sr',0,500)
        self.SR.Restore(model)

    def Extract(self,audioDir):
        audio,sr=librosa.audio.load(audioDir,sr=16000)
        spec=librosa.feature.melspectrogram(y=audio,sr=sr,hop_length=160,n_fft=1024)
        spec=librosa.power_to_db(spec,ref=np.max)
        spec=spec.T
        spec_length=len(spec)
        print(spec.shape)
        cutCount=int((spec_length-100)/25)
        if(cutCount<1):
            return

        x=[]        
        for i in range(cutCount):
            x.append(spec[i*25:100+i*25])
        x=np.reshape(x,(-1,100,128,1))

        x=self.SR.Feature(x)

        x=np.mean(x,0)
        return x

    def Extract_timeSegment(self,audioDir,timeSegment):
        audio,sr=librosa.audio.load(audioDir,sr=16000)
        spec=librosa.feature.melspectrogram(y=audio,sr=sr,hop_length=160,n_fft=1024)
        spec=librosa.power_to_db(spec,ref=np.max)
        spec=spec.T

        r=np.reshape([],(0,2048))

        timeSegment=[s for s in timeSegment if s[1]-s[0]>1.0]
        cut_per_segment=[]
        for s in timeSegment:
            specSegment=spec[int(s[0]*100):int(s[1]*100)]
            specLength=len(specSegment)
            cutCount=int((specLength-50)/50)

            cut_per_segment.append(cutCount)

            if(cutCount<1):
                continue
            x=[]        
            for i in range(cutCount):
                x.append(specSegment[i*25:100+i*25])
            x=np.reshape(x,(-1,100,128,1))
            x=self.SR.Feature(x)
            #x=np.mean(x,0)
            x=np.reshape(x,(-1,2048))
            r=np.append(r,x,axis=0)

        r=np.reshape(r,(-1,2048))
        print("####추출된 벡터 개수, 컷 카운트 개수: ",r.shape,np.sum(cut_per_segment))
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
        LSTM1=128
        LSTM2=128
        FC1=2048
        FC_Out=self.speakerNo

        self.X=tf.placeholder(tf.float32,[None,100,128,1])
        self.Y=tf.placeholder(tf.int64,[None,1])
        
        self.keep_prob=tf.placeholder(tf.float32)  

        self.X1=tf.reshape(self.X,[-1,100,128])
        self.RNN1_cell=tf.contrib.rnn.LSTMCell(LSTM1)
        self.RNN2_cell=tf.contrib.rnn.LSTMCell(LSTM2)

        self.cell_fw=tf.contrib.rnn.MultiRNNCell([self.RNN1_cell,self.RNN2_cell])
        self.cell_bw=tf.contrib.rnn.MultiRNNCell([self.RNN1_cell,self.RNN2_cell])

        self.RNN_output,_=tf.nn.bidirectional_dynamic_rnn(self.cell_fw,self.cell_bw,self.X1,dtype=tf.float32)
        self.rnn_output=tf.concat(self.RNN_output,2)
        self.rnn_output=tf.reshape(self.rnn_output,(-1,100,LSTM2*2,1))
        
        self.L1 = tf.layers.conv2d(inputs=self.X, filters=16, kernel_size=5,padding='valid',activation=tf.nn.elu,kernel_initializer=tf.contrib.layers.xavier_initializer())            
        self.L2 = tf.layers.conv2d(inputs=self.L1, filters=16, kernel_size=5,padding='valid',activation=tf.nn.elu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.L2 = tf.layers.max_pooling2d(self.L2,2,2) 

        self.L3 = tf.layers.conv2d(inputs=self.L2, filters=64, kernel_size=32,padding='same',activation=tf.nn.elu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.L4 = tf.layers.conv2d(inputs=self.L3, filters=64, kernel_size=32,padding='same',activation=tf.nn.elu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.L4 = tf.layers.max_pooling2d(self.L4,2,2)             
        
        self.Conv1_Out=tf.layers.flatten(self.L4)
        
        self.FC=tf.layers.dense(self.Conv1_Out,FC1,activation=tf.nn.relu)
        self.FC=tf.layers.dropout(self.FC,self.keep_prob)
        self.FC_out=tf.layers.dense(self.FC,FC_Out)

        self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.FC_out,labels=tf.one_hot(self.Y,FC_Out)))
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.output=tf.argmax(self.FC_out,1)
        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.output,self.Y),tf.float32))

    def printConv(self,x,keep_prob=1.0):
        return self.sess.run([self.L1,self.L2,self.L3,self.L4],feed_dict={self.X:x,self.keep_prob:keep_prob})
            
    def Outputs(self,x,keep_prob=1.0):
        return self.sess.run(self.output,feed_dict={self.X:x,self.keep_prob:keep_prob})

    def Cost(self,x,y,keep_prob=1.0):
        return self.sess.run(self.cost,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})

    def Train(self,x,y,keep_prob=0.2):
        return self.sess.run(self.optimizer,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})

    def Accuracy(self,x,y,keep_prob=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})

    def Feature(self,x,keep_prob=1.0):
        return self.sess.run(self.FC,feed_dict={self.X:x,self.keep_prob:keep_prob})

    def Save(self,name):

        saver=tf.train.Saver()
        saver.save(self.sess, name)

    def Restore(self,name):
        saver=tf.train.Saver()
        saver.restore(self.sess,name)

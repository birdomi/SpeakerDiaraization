import numpy as np
import tensorflow as tf
import os
import librosa
import sklearn.preprocessing

batch_size=1500
windowstep=100
windowmul=3
windowsize=windowstep*windowmul
labelLength=int(windowsize/3)
class Data():
    def __init__(self):
        self._MFCC=[]
        self._LABEL=[]
        self.numberBatch=0

    def Get_Data(self,mfcc_path,label_path='None'):
        """
        for i in range(len(mfcc_path)):
            if(mfcc_path[i]=='.'):
                self.__index=i
        self.__filename=mfcc_path[:self.__index]
        if(os.path.exists(self.__filename+'.mfc')and os.path.exists(self.__filename+'.lab')):
            self.LodaData(mfcc_path,label_path)
        else:
        """
        self.MakeData(mfcc_path,label_path)
        #self._SaveData(mfcc_path,label_path)
            
    def LodaData(self,mfcc_path,label_path):
        self.__mfcc=np.loadtxt(self.__filename+'.mfc')
        self.__label=np.loadtxt(self.__filename+'.lab',int)
        self.__mfcc=np.reshape(self.__mfcc,(-1,320,39))
        self.__label=np.reshape(self.__label,(-1,320))
        print(self.__mfcc)
        print(self.__label)
        
        self.__batch_num=int(len(self.__mfcc)/batch_size)
        for i in range(self.__batch_num):
            self._MFCC.append(self.__mfcc[i*batch_size:(i+1)*batch_size])
            self._LABEL.append(self.__label[i*batch_size:(i+1)*batch_size])
            self.numberBatch+=1
        self._MFCC.append(self.__mfcc[self.__batch_num*batch_size:])
        self._LABEL.append(self.__label[self.__batch_num*batch_size:])
        self.numberBatch+=1

    def MakeData(self,mfcc_path,label_path):
        pass

    def _ExtractMFCC(self,mfcc_path):
        self.__audio,self.__sr=librosa.load(mfcc_path,sr=16000)
        self.__mfcc=librosa.feature.mfcc(self.__audio,self.__sr,n_mfcc=12,hop_length=int(self.__sr/100), n_fft=int(self.__sr/40))
        self.__mfcc_energy=librosa.feature.rmse(S=self.__mfcc,hop_length=int(self.__sr/100), n_fft=int(self.__sr/40))
        self.__mfcc=np.append(self.__mfcc,self.__mfcc_energy,axis=0)

        self.__mfcc1=librosa.feature.delta(self.__mfcc)
        
        self.__mfcc2=librosa.feature.delta(self.__mfcc,order=2)
        

        self.__mfcc=np.append(self.__mfcc,self.__mfcc1,axis=0)
        self.__mfcc=np.append(self.__mfcc,self.__mfcc2,axis=0)
        self.__Tmfcc=np.transpose(self.__mfcc,(1,0))
        print(len(self.__Tmfcc))
        self.__Tmfcc=sklearn.preprocessing.normalize(self.__Tmfcc)
        
        return self.__Tmfcc
        
        return self.__tmp
    

    def _SaveData(self,mfcc_path,label_path):
        self.__mfcc=self._MFCC[0]
        self.__label=self._LABEL[0]

        for i in range(1,self.numberBatch):
            self.__mfcc=np.append(self.__mfcc,self._MFCC[i])
            self.__label=np.append(self.__label,self._LABEL[i])
        print(self.__mfcc.shape,self.__label.shape)

        for i in range(len(mfcc_path)):
            if(mfcc_path[i]=='.'):
                self.__index=i
        np.savetxt(mfcc_path[:self.__index]+'.mfc',self.__mfcc,delimiter=' ')
        np.savetxt(mfcc_path[:self.__index]+'.lab',self.__label,delimiter= ' ')



class ICSI_Data(Data):
    def MakeData(self, mfcc_path, label_path):
        self.__BatchMFCC(mfcc_path)
        self.__BatchMRT(label_path)

    def __BatchMFCC(self,mfcc_path):
        self.__hamming=np.reshape([1]*300,(300,1))-np.reshape(np.hamming(300),(300,1))
        self.__Tmfcc=self._ExtractMFCC(mfcc_path)
        
        self.__data_length=int(len(self.__Tmfcc)/windowstep)
        self.__num_seq=self.__data_length-windowmul+1
        
        self.__tmp=[]
        for i in range(self.__num_seq):
            self.__tempSequence=self.__Tmfcc[i*windowstep:(i+windowmul)*windowstep]            
            self.__tempSequence=np.multiply(self.__tempSequence,self.__hamming)
            self.__tmp.append(self.__tempSequence)

        self.__tmp=np.reshape(self.__tmp,(-1,windowsize,39))        
        self.__batch_num=int(len(self.__tmp)/batch_size)
        print(self.__tmp.shape)
        for i in range(self.__batch_num):
            self._MFCC.append(self.__tmp[i*batch_size:(i+1)*batch_size])
            self.numberBatch+=1
       
    def __LabelMRT(self,label_path):
        self.__file=open(label_path,'r')
        self.__lines=self.__file.readlines()
        for self.__line in self.__lines:
            if(self.__line[2:13]=='<Transcript'):
                self.__startTime,self.__endTime=self.__Get_Data_fromMRT(self.__line)
                break
        self.__Label=['']*int(self.__endTime*100+1)        
        self.__Label_01=[0]*int(self.__endTime*100+1)

        for i in range(len(self.__lines)):
            if(self.__lines[i][4:12]=='<Segment'):
                self.__startTime,self.__endTime,self.__speakerName=self.__Get_Data_fromMRT(self.__lines[i])
                if(self.__Is_VocalSound(self.__lines[i+1])):
                    for index in range(int(self.__startTime*100),int(self.__endTime*100)+1):
                        self.__Label[index]+=' '+self.__speakerName
        #print(self.__Label)
        
        self.__pastSpeaker=''
        for i in range(len(self.__Label)):
            self.__currnetSpeaker=self.__Label[i]
            if(self.__pastSpeaker!=self.__currnetSpeaker):
                self.__isLonger_than_1sec=True
                for time in range(0,15):
                    if(i+time+1<len(self.__Label)):
                        if(self.__Label[i+time]!=self.__Label[i+time+1]):
                            self.__isLonger_than_1sec=False

                if(self.__isLonger_than_1sec):
                    self.__pastSpeaker=self.__currnetSpeaker
                    self.__Label_01[i]=1
                    #print(i/100)

        return self.__Label_01
        

    def __Get_Data_fromMRT(self,line):
        self.__dataIndex=[]
        for i in range(len(line)):
            if(line[i]=='"'):
                self.__dataIndex.append(i)

        for i in range(len(self.__dataIndex)):
            if(i%2==0):
                self.__dataIndex[i]+=1

        self.__startTime=float(line[self.__dataIndex[0]:self.__dataIndex[1]])
        self.__endTime=float(line[self.__dataIndex[2]:self.__dataIndex[3]])
        if(len(self.__dataIndex)>4):
            self.__name=line[self.__dataIndex[4]:self.__dataIndex[5]]
            return self.__startTime,self.__endTime,self.__name
        return self.__startTime,self.__endTime

    def __Is_VocalSound(self,line):
        self.__Is_Vocal=True
        for i in range(7):
            if(line[i:i+2]=='<N'or line[i:i+2]=='<C'or line[i:i+2]=='<U'):
                self.__Is_Vocal=False
        return self.__Is_Vocal
    def __BatchMRT(self,label_path):        
        #
        self.__Label=self.__LabelMRT(label_path)
        print(len(self.__Label))

        self.__data_length=int(len(self.__Label)/windowstep)
        self.__num_seq=self.__data_length-windowmul+1
        self.__tmp=[]
        for i in range(self.__num_seq):
            self.__tempSequence=self.__Label[i*windowstep:(i+windowmul)*windowstep]
            self.__tmp.append(self.__tempSequence)
        self.__tmp=np.reshape(self.__tmp,(-1))
        self.__label=[]
        for i in range(int(len(self.__tmp)/labelLength)):
            self.__check=False
            for j in range(i*labelLength,(i+1)*labelLength):
                if(self.__tmp[j]==1):
                    self.__check=True
            if(self.__check):
                self.__label.append(1)
            else:
                self.__label.append(0)
        
        self.__label=np.reshape(self.__label,(-1,int(windowsize/labelLength),1))
        print(self.__label.shape)
        self.__batch_num=int(len(self.__label)/batch_size)
        for i in range(self.__batch_num):
            self._LABEL.append(self.__label[i*batch_size:(i+1)*batch_size])
        

class KVD_Data(Data):
    def MakeData(self, mfcc_path, label_path):
        self.__BatchMFCC(mfcc_path)
        self.__BatchMRT(label_path,10)

    def __BatchMFCC(self,mfcc_path):
        self.__tmp=self._ExtractMFCC(mfcc_path)
        print(self.__tmp.shape)
        self._MFCC=self.__tmp

    def __BatchMRT(self,label_path,label_weight):
        self.__file=open(label_path,'r')
        self.__lines=self.__file.readlines()
        self.__arrayLength=int(float(self.__lines[0])*100+1)
        
        self.__Label=[0]*self.__arrayLength

        for i in range(1,len(self.__lines)):
            self.__index=int(float(self.__lines[i])*100)
            self.__Label[self.__index]=1
            for j in range(label_weight):
                if(self.__index-j>=0):
                    self.__Label[self.__index-j]=1
                if(self.__index+j<len(self.__Label)):
                    self.__Label[self.__index+j]=1

        #
        self.__tmp=[]
        self.num_seq=int(len(self.__Label)/time_windowSteps)-4
        for i in range(self.num_seq):
            self.__tempSequence=self.__Label[i*time_windowSteps:(i+4)*time_windowSteps]
            self.__tmp.append(self.__tempSequence)
        self.__tmp=np.reshape(self.__tmp,(-1,time_windowSize,1))
        print(self.__tmp.shape)
        self._LABEL=self.__tmp

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

    def Train(self,x,y,keep_prob=0.8):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})

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
        print(self.total_changePoint,self.check_for_changePoint)
        if(self.total_changePoint!=0):
            self.accuracy_changePoint=(float)(self.check_for_changePoint/self.total_changePoint)
        if(self.total_non_changePoint!=0):
            self.accuracy_non_changePoint=(float)(self.check_for_falseAlarm/self.total_non_changePoint)
                
        return self.accuracy_changePoint,self.accuracy_non_changePoint

    def Save(self):
        saver=tf.train.Saver()
        saver.save(self.sess, self.name+'/model')

    def Restore(self):
        saver=tf.train.Saver()
        saver.restore(self.sess,self.name+'/model')

class RNN_Model(Model):
    def _build_net(self):
        self.LSTM_layer1=35
        self.LSTM_layer2=30

        self.FC_layer1=self.LSTM_layer2*2*4
        self.FC_layer2=80
        self.FC_layerOut=2

        self.X=tf.placeholder(tf.float32,[None,windowsize,39])
        self.Y=tf.placeholder(tf.int32,[None,3,1])
        self.keep_prob=tf.placeholder(tf.float32)
        
        self.Y1,self.Y2,self.Y3=tf.split(self.Y,3,1)

        self.RNN1_cell=tf.contrib.rnn.LSTMCell(self.LSTM_layer1,activation=tf.nn.tanh)
        self.RNN1_cell=tf.contrib.rnn.DropoutWrapper(self.RNN1_cell,output_keep_prob=self.keep_prob)

        self.RNN2_cell=tf.contrib.rnn.LSTMCell(self.LSTM_layer2,activation=tf.nn.tanh)
        self.RNN2_cell=tf.contrib.rnn.DropoutWrapper(self.RNN2_cell,output_keep_prob=self.keep_prob)

        self.cell_fw=tf.contrib.rnn.MultiRNNCell([self.RNN1_cell,self.RNN2_cell])
        self.cell_bw=tf.contrib.rnn.MultiRNNCell([self.RNN1_cell,self.RNN2_cell])
        self.RNN_output,_=tf.nn.bidirectional_dynamic_rnn(self.cell_fw,self.cell_bw,self.X,dtype=tf.float32)
        self.rnn_output=tf.concat(self.RNN_output,2)
        
        #tf.ones(300)-
        self.__rnn_first=tf.split(self.rnn_output,300,1)[0]
        self.__rnn_mid1=tf.split(self.rnn_output,300,1)[49]
        self.__rnn_mid2=tf.split(self.rnn_output,300,1)[99]
        self.__rnn_mid3=tf.split(self.rnn_output,300,1)[149]
        self.__rnn_mid4=tf.split(self.rnn_output,300,1)[199]
        self.__rnn_mid5=tf.split(self.rnn_output,300,1)[249]
        self.__rnn_mid6=tf.split(self.rnn_output,300,1)[299]
        self.__rnnout=tf.concat([self.__rnn_first,self.__rnn_mid1,self.__rnn_mid2,self.__rnn_mid3,self.__rnn_mid4,self.__rnn_mid5,self.__rnn_mid6],1)
        self.fc_input=tf.reshape(self.__rnnout,[-1,420])

        self.FC1_1=tf.contrib.layers.fully_connected(self.fc_input,self.FC_layer1)
        self.FC1_1=tf.nn.dropout(self.FC1_1,self.keep_prob)
        self.FC1_2=tf.contrib.layers.fully_connected(self.FC1_1,self.FC_layer2)
        self.FC1_2=tf.nn.dropout(self.FC1_2,self.keep_prob)
        self.FC1_out=tf.contrib.layers.fully_connected(self.FC1_2,self.FC_layerOut,activation_fn=None)

        self.FC2_1=tf.contrib.layers.fully_connected(self.fc_input,self.FC_layer1)
        self.FC2_1=tf.nn.dropout(self.FC2_1,self.keep_prob)
        self.FC2_2=tf.contrib.layers.fully_connected(self.FC2_1,self.FC_layer2)
        self.FC2_2=tf.nn.dropout(self.FC2_2,self.keep_prob)
        self.FC2_out=tf.contrib.layers.fully_connected(self.FC2_2,self.FC_layerOut,activation_fn=None)

        self.FC3_1=tf.contrib.layers.fully_connected(self.fc_input,self.FC_layer1)
        self.FC3_1=tf.nn.dropout(self.FC3_1,self.keep_prob)
        self.FC3_2=tf.contrib.layers.fully_connected(self.FC3_1,self.FC_layer2)
        self.FC3_2=tf.nn.dropout(self.FC3_2,self.keep_prob)
        self.FC3_out=tf.contrib.layers.fully_connected(self.FC3_2,self.FC_layerOut,activation_fn=None)        
        
        self.FC1_out=tf.reshape(self.FC1_out,(-1,1,2))
        self.FC2_out=tf.reshape(self.FC2_out,(-1,1,2))
        self.FC3_out=tf.reshape(self.FC3_out,(-1,1,2))

        self.cost1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.FC1_out,labels=tf.one_hot(self.Y1,2)))
        self.cost2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.FC2_out,labels=tf.one_hot(self.Y2,2)))
        self.cost3=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.FC3_out,labels=tf.one_hot(self.Y3,2)))

        self.cost=(self.cost1+self.cost2+self.cost3)
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
        self.out1=tf.argmax(self.FC1_out,2)
        self.out2=tf.argmax(self.FC2_out,2)
        self.out3=tf.argmax(self.FC3_out,2)
        self.outputs=tf.concat([self.out1,self.out2,self.out3],1)

    def Show_Shape(self,x,y,keep_prob=1.0):
        print(self.sess.run(tf.shape(self.RNN_output),feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))
        print(self.sess.run(tf.shape(self.rnn_output),feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))
        print(self.sess.run(tf.shape(self.__rnnout),feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))
        print(self.sess.run(tf.shape(self.fc_input),feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))
        print(self.sess.run(self.outputs,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))

    def Show_Reuslt(self,x,y,keep_prob=1.0):
        self.x_prediction=np.reshape(self.Outputs(x),[-1])
        self.y_prediction=np.reshape(y,[-1])

        self.__dataLength=int(len(self.y_prediction)/3)
        for i in range(self.__dataLength):
            print('테스트 구간 :',i,'~',(i+3))
            if(self.x_prediction[3*i]==self.y_prediction[3*i]):
                test1=True
            else:
                test1=False

            if(self.x_prediction[3*i+1]==self.y_prediction[3*i+1]):
                test2=True
            else:
                test2=False

            if(self.x_prediction[3*i+2]==self.y_prediction[3*i+2]):
                test3=True
            else:
                test3=False

            print(test1,test2,test3)
            print(self.x_prediction[3*i],self.x_prediction[3*i+1],self.x_prediction[3*i+2])
            print(self.y_prediction[3*i],self.y_prediction[3*i+1],self.y_prediction[3*i+2])
            print()

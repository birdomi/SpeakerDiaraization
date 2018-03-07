import numpy as np
import tensorflow as tf
import os
import librosa
import sklearn.preprocessing

batch_size=100
windowstep=300
windowmul=1
windowsize=windowstep*windowmul
labelLength=int(windowsize/windowmul)
class Data():
    def __init__(self):
        self._MFCC=[]
        self._LABEL=[]
        self._LABEL_Perfection=[]
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
        self.__mfcc_energy=librosa.feature.rmse(S=self.__mfcc,hop_length=int(self.__sr/100))
        self.__mfcc=np.append(self.__mfcc,self.__mfcc_energy,axis=0)

        self.__mfcc1=librosa.feature.delta(self.__mfcc)
        
        self.__mfcc2=librosa.feature.delta(self.__mfcc,order=2)
        

        self.__mfcc=np.append(self.__mfcc,self.__mfcc1,axis=0)
        self.__mfcc=np.append(self.__mfcc,self.__mfcc2,axis=0)
        self.__Tmfcc=np.transpose(self.__mfcc,(1,0))
        #self.__Tmfcc=sklearn.preprocessing.normalize(self.__Tmfcc)
        
        return self.__Tmfcc    

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
    def MakeData(self, mfcc_path, label_path,hamming):
        self.__BatchMFCC(mfcc_path,hamming)
        self.__BatchMRT(label_path)

    def __BatchMFCC(self,mfcc_path,hamming):
        if(hamming=='N'):
            self.__hamming=np.ones((windowsize,1))
        if(hamming=='On'):
            self.__hamming=np.reshape(np.hamming(windowsize),(windowsize,1))
        if(hamming=='1-On'):
            self.__hamming=np.ones((windowsize,1))-np.reshape(np.hamming(windowsize),(windowsize,1))

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
                    for index in range(int(self.__startTime*100)+25,int((self.__endTime*100)+1)-25):
                        self.__Label_01[index]=1
        
        self.__count0=0
        self.__count1=0
        for i in range(len(self.__Label_01)):
            if(self.__Label_01[i]==1):
                self.__count1+=1
            else:
                self.__count0+=1
        print(self.__count0,self.__count1)
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
        self.__label=self.__LabelMRT(label_path)
        self.__tmp=[]
        self.__data_length=int(len(self.__label)/windowstep)
        self.__num_seq=self.__data_length-windowmul+1

        for i in range(self.__num_seq):
            self.__tempSequence=self.__label[i*windowstep:(i+windowmul)*windowstep]
            self.__tmp.append(self.__tempSequence)
        self.__tmp=np.reshape(self.__tmp,(-1,windowsize,1))
        print(self.__tmp.shape)
        self.__batch_num=int(len(self.__tmp)/batch_size)
        for i in range(self.__batch_num):
             self._LABEL.append(self.__tmp[i*batch_size:(i+1)*batch_size])    

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

    def Cost(self,x,y,keep_prob=1.0):
        return self.sess.run(self.cost,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})

    def return_ResultMatrix(self,x,y):
        self.x_prediction=np.reshape(self.Outputs(x),[-1])
        self.y_prediction=np.reshape(y,[-1])

        self.resultMatrix=np.zeros([2,2],int)
        for i in range(len(self.x_prediction)):
            self.resultMatrix[self.y_prediction[i]][self.x_prediction[i]]+=1

        return self.resultMatrix

    def return_ResultPerfection(self,x,y,labelperfection):
        self.x_prediction=np.reshape(self.Outputs(x),[-1])
        self.y_prediction=np.reshape(y,[-1])

        numberLabelPerfect=0
        numberLabelNotPerfect=0

        numberCorrectatPerfect=0
        numberWrongatPerfect=0
        numberCorrectatWrong=0
        numberWrongatWrong=0

        for i in range(len(self.x_prediction)):
            if(labelperfection[i]==1):
                numberLabelPerfect+=1
                if(self.x_prediction[i]==self.y_prediction[i]):
                    numberCorrectatPerfect+=1
                else:
                    numberWrongatPerfect+=1
            else:
                numberLabelNotPerfect+=1
                if(self.x_prediction[i]==self.y_prediction[i]):
                    numberCorrectatWrong+=1
                else:
                    numberWrongatWrong+=1

        result=np.array([[1,numberCorrectatPerfect/numberLabelPerfect,numberWrongatPerfect/numberLabelPerfect],
                         [0,numberCorrectatWrong/numberLabelNotPerfect,numberWrongatWrong/numberLabelNotPerfect]])
        print(result)
        print(numberLabelPerfect,numberLabelNotPerfect)
        print(numberCorrectatPerfect,numberCorrectatWrong)
        return numberCorrectatPerfect/numberLabelPerfect,numberCorrectatWrong/numberLabelNotPerfect
        


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
        #print('1의 총 개수, 맞은 개수: ',self.total_changePoint,self.check_for_changePoint)
        #print('0의 총 개수, 맞은 개수: ',self.total_non_changePoint,self.check_for_falseAlarm) 
        if(self.total_changePoint!=0):
            self.accuracy_changePoint=(float)(self.check_for_changePoint/self.total_changePoint)
        if(self.total_non_changePoint!=0):
            self.accuracy_non_changePoint=(float)(self.check_for_falseAlarm/self.total_non_changePoint)
                
        return self.accuracy_changePoint,self.accuracy_non_changePoint

    def Save(self,name):
        saver=tf.train.Saver()
        saver.save(self.sess, name)

    def Restore(self,name):
        saver=tf.train.Saver()
        saver.restore(self.sess,name)


class RNN_Model_usingAll(Model):
    def _build_net(self):
        self.LSTM_layer1=35
        self.LSTM_layer2=30

        self.FC_layer1=40
        self.FC_layerOut=2

        self.X=tf.placeholder(tf.float32,[None,windowsize,39])
        self.Y=tf.placeholder(tf.int32,[None,windowsize,1])
        self.keep_prob=tf.placeholder(tf.float32)       

        self.RNN1_cell=tf.contrib.rnn.LSTMCell(self.LSTM_layer1)
        self.RNN1_cell=tf.contrib.rnn.DropoutWrapper(self.RNN1_cell,output_keep_prob=self.keep_prob)

        self.cell_fw=tf.contrib.rnn.MultiRNNCell([self.RNN1_cell])
        self.cell_bw=tf.contrib.rnn.MultiRNNCell([self.RNN1_cell])
        self.RNN_output,_=tf.nn.bidirectional_dynamic_rnn(self.cell_fw,self.cell_bw,self.X,dtype=tf.float32)
        self.rnn_output=tf.concat(self.RNN_output,2)

        self.FC_input=tf.reshape(self.rnn_output,(-1,self.LSTM_layer1*2))
        
        self.FC1=tf.contrib.layers.fully_connected(self.FC_input,self.FC_layer1)
        self.FC_out=tf.contrib.layers.fully_connected(self.FC1,self.FC_layerOut,activation_fn=None)
        self.FC_out=tf.reshape(self.FC_out,(-1,windowsize,2))

        self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.FC_out,labels=tf.one_hot(self.Y,2)))
        
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
        self.outputs=tf.argmax(self.FC_out,2)

    def Show_Shape(self,x,y,keep_prob=1.0):
        print(self.sess.run(tf.shape(self.rnn_output),feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))
        print(self.sess.run(tf.shape(self.__rnn_first),feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))
        print(self.sess.run(tf.shape(self.__rnnout),feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))
        print(self.sess.run(tf.shape(self.fc_input),feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))
        print(self.sess.run(self.outputs,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob}))

    def Show_Reuslt(self,x,y,mfcc_path,keep_prob=1.0):
        import pylab as plt
        self.__audio,self.__sr=librosa.load(mfcc_path,sr=16000)
        plt.figure(1,figsize=(1000,1))
        plt.xlim([16000*0,16000*4])
        plt.plot(self.__audio)
        self.x_out=self.sess.run(self.outputs,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})
        self.y_in=y
        self.x_out=np.reshape(self.x_out,(-1))
        self.y_in=np.reshape(self.y_in,(-1))
        x=[];y=[];t=[]
        for i in range(len(self.x_out)):
            x.append(self.x_out[i])
            y.append(self.y_in[i]+2)
            t.append(i)
        tmin=t[0]
        tmax=t[len(t)-1]
        tstep=round((tmax-tmin)/10)
        plt.figure(2,figsize=(1000,1))
        plt.plot(t,x,'r',label="x")
        plt.plot(t,y,label='y')
        plt.legend(loc='upper right')
        plt.xlabel('t')
        plt.ylabel('output')
        plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.9)
        plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
        plt.minorticks_on()
        plt.xticks(np.arange(tmin,tmax+tstep,tstep))
        plt.xlim([0,tmax])
        plt.ylim([0,5])
        #plt.savefig("{}-cost.png".format(fn.split(".")[:-1][0]), bbox_inches='tight')        
        plt.show()





import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import python_speech_features
import winsound
from sklearn.preprocessing import MinMaxScaler

#config
time_windowSteps=80#*0.01s
time_windowLength=4*80#*0.01s
learning_rate=0.0001
batch_size=1000
#

vec_per_frame=39
no_output=2
beep_frequency=1500
beep_duration=2000

def beep():
    winsound.Beep(beep_frequency,beep_duration)


class mfcc_label_data():
    mfcc_data=[]
    label_data=[]
    data_length=0
    
class Data(object):
    data={
        'train':mfcc_label_data(),
        'dev':mfcc_label_data(),
        'test':mfcc_label_data()
    }

    def Mfcc(self,s):        
        sig,f=wav.read('wave/'+s)
        mfcc=python_speech_features.base.mfcc(f,sig)
        delta=python_speech_features.base.delta(mfcc,1)
        ddelta=python_speech_features.base.delta(delta,1)
        

        Data_length=(int)(len(mfcc)/time_windowSteps)
        tmp_data_for_X=[]
        
        seq=(int)(time_windowLength/time_windowSteps)
        num_seq=Data_length-seq
        for i in range(Data_length):
            tmp_data_for_X.append(Data.Return_a_Sequence_data(mfcc,i*time_windowSteps))
            tmp_data_for_X.append(Data.Return_a_Sequence_data(delta,i*time_windowSteps))
            tmp_data_for_X.append(Data.Return_a_Sequence_data(ddelta,i*time_windowSteps))
        tmp_data_for_X=np.reshape(tmp_data_for_X,(-1,39))
        """
        scaler=MinMaxScaler()
        scaler.fit(tmp_data_for_X)
        tmp_data_for_X=scaler.transform(tmp_data_for_X)
       """
        tmp_data_for_X=np.reshape(tmp_data_for_X,(-1,time_windowSteps,vec_per_frame))
        tmp_seq=[]
        for i in range(num_seq):
            for j in range(seq):
                tmp_seq.append(tmp_data_for_X[i+j])
        
        tmp_seq=np.reshape(tmp_seq,(-1,time_windowLength,vec_per_frame))
        return tmp_seq

    def Labeling(self,s):
        tmp=np.loadtxt('Label/'+s,dtype=int)
        Data_length=(int)(len(tmp)/time_windowSteps)
                
        seq=(int)(time_windowLength/time_windowSteps)
        num_seq=Data_length-seq

        tmp_data_for_Y=[]
        for i in range(Data_length):
            tmp_data_for_Y.append(Data.Return_a_Sequence_data(tmp,i*time_windowSteps))        
        tmp_data_for_Y=np.reshape(tmp_data_for_Y,(-1,time_windowSteps))

        dataY=[]
        for i in range(num_seq):
            for j in range(seq):
                dataY.append(tmp_data_for_Y[i+j])
        dataY=np.reshape(dataY,(-1,time_windowLength))

        return dataY 
   
    def Return_a_Sequence_data(array,index):
        temp=array[index:(index+time_windowSteps)]
        temp=np.reshape(temp,(-1))
        return temp     

    def Load_Data(self,mfccpath,labelpath,dataname):
        for i in range(len(mfccpath)):
            append_mfcc=self.Mfcc(mfccpath[i])
            mfcc_data=append_mfcc
            mfcc_data=np.reshape(mfcc_data,(-1,time_windowLength,vec_per_frame))
            print(mfcc_data.shape)

            append_label=self.Labeling(labelpath[i])
            label_data=append_label
            label_data=np.reshape(label_data,(-1,time_windowLength))
            print(label_data.shape)

            data_length=len(label_data)
        
            if(data_length==0):
               print('Data not Ready')
               return
        
            number_batch=(int)(data_length/batch_size)

            tmp_mfcc=self.data[dataname].mfcc_data
            tmp_label=self.data[dataname].label_data

            for i in range((int)(number_batch)):
                tmp_mfcc.append(mfcc_data[i*batch_size:(i+1)*batch_size])
                tmp_label.append(label_data[i*batch_size:(i+1)*batch_size])
                self.data[dataname].data_length+=1
            self.data[dataname].mfcc_data=tmp_mfcc
            self.data[dataname].label_data=tmp_label
        

class file(object):
    def Return_Num_1(array):
        count=0
        for i in range(len(array)):
            if(array[i]==1):
                count=count+1
        return count

class RNN_model:
    
    def __init__(self, sess,name,):
        self.sess=sess
        self.name=name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):            
            layer1=32
            layer2=64
            layer3=34
            layer4=20
            
            self.layer_input=layer4*2
            self.layer1=40
            self.layer2=10
            self.layer_output=no_output

            self.X=tf.placeholder(tf.float32,[None,time_windowLength,vec_per_frame])
            self.Y_transcripts=tf.placeholder(tf.int32,[None,time_windowLength])
            self.keep_prob=tf.placeholder(tf.float32)
            self.data_length=tf.placeholder(tf.int32)

            
            
            layer1_cell=tf.contrib.rnn.LSTMCell(num_units=layer1,activation=tf.nn.tanh,initializer=tf.contrib.layers.xavier_initializer())
            layer1_cell=tf.contrib.rnn.DropoutWrapper(layer1_cell,self.keep_prob,self.keep_prob)
            layer2_cell=tf.contrib.rnn.LSTMCell(num_units=layer2,activation=tf.nn.tanh,initializer=tf.contrib.layers.xavier_initializer())
            layer3_cell=tf.contrib.rnn.LSTMCell(num_units=layer3,activation=tf.nn.tanh,initializer=tf.contrib.layers.xavier_initializer())
            layer4_cell=tf.contrib.rnn.LSTMCell(num_units=layer4,activation=tf.nn.tanh,initializer=tf.contrib.layers.xavier_initializer())
            layer4_cell=tf.contrib.rnn.DropoutWrapper(layer4_cell,self.keep_prob,self.keep_prob)

            multi_cell=tf.contrib.rnn.MultiRNNCell([layer1_cell,layer4_cell])

            layer2_output,_=tf.nn.bidirectional_dynamic_rnn(multi_cell,multi_cell,self.X,dtype=tf.float32)
            rnn_output=tf.concat(layer2_output,2)
            fc_inputs=tf.reshape(rnn_output,[-1,self.layer_input])
                        
            W1=tf.get_variable('W1',[self.layer_input,self.layer1],tf.float32,tf.contrib.layers.xavier_initializer())
            b1=tf.Variable(tf.random_normal([self.layer1]))
            L1=tf.nn.tanh(tf.matmul(fc_inputs,W1)+b1)
            L1=tf.nn.dropout(L1,self.keep_prob)

            W2=tf.get_variable('W2',[self.layer1,self.layer2],tf.float32,tf.contrib.layers.xavier_initializer())
            b2=tf.Variable(tf.random_normal([self.layer2]))
            L2=tf.nn.tanh(tf.matmul(L1,W2)+b2)
            L2=tf.nn.dropout(L2,self.keep_prob)

            W3=tf.get_variable('W3',[self.layer2,no_output],tf.float32,tf.contrib.layers.xavier_initializer())
            b3=tf.Variable(tf.random_normal([no_output]))
            L3=(tf.matmul(L2,W3)+b3)

            self.outputs=tf.reshape(L3,(-1,time_windowLength,no_output))            
            
            
        self.cost=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.outputs,labels=self.Y_transcripts))
        self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def show_cost(self,x,y,keep_prob=1.0):
        return self.sess.run(self.cost,feed_dict={self.X:x,self.Y_transcripts:y,self.keep_prob:keep_prob})
       
    def predict(self,x_test,keep_prob=1.0):
        return self.sess.run(tf.argmax(self.outputs,axis=2),feed_dict={self.X:x_test,self.keep_prob:keep_prob})
  
    def output(self,x_test,keep_prob=1.0):
        return self.sess.run(self.outputs,feed_dict={self.X:x_test,self.keep_prob:keep_prob})

    def train(self,x_data,y_data,data_length,keep_prob=1.0):
        return self.sess.run([self.cost,self.outputs,self.optimizer],feed_dict={self.X:x_data,self.Y_transcripts:y_data,self.data_length:data_length,self.keep_prob:keep_prob})

    def accuracy(self,x_data,y_data):
        x_prediction=np.reshape(self.predict(x_data),[-1])
        y_prediction=np.reshape(y_data,[-1])
        
        check_for_changePoint=0
        check_for_falseAlarm=0
        total_changePoint=0
        total_non_changePoint=0
        accuracy_changePoint=0.0
        accuracy_non_changePoint=0.0
        for i in range(self.sess.run(tf.size(y_prediction))):            
            if(y_prediction[i]==1):
                total_changePoint=total_changePoint+1
                if(x_prediction[i]==1):
                    check_for_changePoint=check_for_changePoint+1
            else:
                total_non_changePoint=total_non_changePoint+1
                if(x_prediction[i]==0):
                    check_for_falseAlarm=check_for_falseAlarm+1

        if(total_changePoint!=0):
            accuracy_changePoint=(float)(check_for_changePoint/total_changePoint)
        if(total_non_changePoint!=0):
            accuracy_non_changePoint=(float)(check_for_falseAlarm/total_non_changePoint)
                
        return accuracy_changePoint,accuracy_non_changePoint

    def Make_Result(self,x_test,keep_prob=1.0):
        result=np.reshape(self.predict(x_test),[-1])

        Temp=[]
        time=(int)(time_windowLength/time_windowSteps)
        
        for i in range(4):
            if(i==0):
                for j in range(time_windowSteps):
                    Temp.append(result[j])
            if(i==1):
                for j in range(time_windowSteps,2*time_windowSteps):
                    index=j
                    tmp_sum=0
                    for k in range(2):
                        tmp_sum+=result[index+3*k*time_windowSteps]
                    Temp.append(tmp_sum/2.0)
            if(i==2):
                for j in range(2*time_windowSteps,3*time_windowSteps):
                    index=j
                    tmp_sum=0
                    for k in range(3):
                        tmp_sum+=result[index+3*k*time_windowSteps]
                    Temp.append(tmp_sum/3.0)

        for i in range(len(x_test)-3):
            step=(i+1)*time_windowLength
            for j in range(step-time_windowSteps,step):
                index=j
                tmp_sum=0
                for k in range(4):
                    tmp_sum+=result[index+3*k*time_windowSteps]
                Temp.append(tmp_sum/4.0)

        for i in range(len(x_test)-3,len(x_test)):
            step=(i+1)*time_windowLength

            if(i==len(x_test)-3):
               for j in range(step-time_windowSteps,step):
                   index=j
                   tmp_sum=0
                   for k in range(3):
                       tmp_sum+=result[index+3*k*time_windowSteps]
                   Temp.append(tmp_sum/3.0)
            if(i==len(x_test)-2):
                for j in range(step-time_windowSteps,step):
                   index=j
                   tmp_sum=0
                   for k in range(2):
                       tmp_sum+=result[index+3*k*time_windowSteps]
                   Temp.append(tmp_sum/2.0)
            if(i==len(x_test)-1):
                step=(i+1)*time_windowLength
                for j in range(step-time_windowSteps,step):
                    Temp.append(result[j])
        return Temp

    def save(self):
        saver=tf.train.Saver()
        saver.save(self.sess,'Model/model')
    def restore(self):
        saver=tf.train.Saver()
        saver.restore(self.sess,'Model/model')
pass

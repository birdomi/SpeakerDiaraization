import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import python_speech_features
import winsound

#config
label_weight=5
no_output =2
vec_per_frame=39
vec_per_sequence=vec_per_frame*label_weight
seq_length=10*label_weight
learning_rate=1e-3
batch_size=1000
test_rate=0.7

beep_frequency=1500
beep_duration=2000

def beep():
    winsound.Beep(beep_frequency,beep_duration)


class mfcc_label_data():
    mfcc_data=[]
    label_data=[]
    data_length=0
    
class Data(object):
    mfcc_data=[]
    label_data=[]

    train=mfcc_label_data()
    test=mfcc_label_data()
    

    def Mfcc(self,s):
        fs,audio=wav.read(s)
        mfcc=python_speech_features.base.mfcc(audio,fs)
        delta=python_speech_features.base.delta(mfcc,2)
        ddelta=python_speech_features.base.delta(delta,2)

        Data_length=(int)(len(mfcc)/seq_length)
        tmp_data_for_X=[]

        for i in range(Data_length):
            tmp_data_for_X.append(Data.Return_a_Sequence_data(mfcc,i))
            tmp_data_for_X.append(Data.Return_a_Sequence_data(delta,i))
            tmp_data_for_X.append(Data.Return_a_Sequence_data(ddelta,i))
            
        tmp_data_for_X=np.reshape(tmp_data_for_X,(-1,seq_length,vec_per_frame))
        self.mfcc_data=tmp_data_for_X
        return

    def Labeling(self,s):
        tmp=np.loadtxt('Label.txt',dtype=int)
        Data_length=(int)(len(tmp)/label_weight)

        tmp_data_for_Y=[]
        
        for i in range(Data_length):
            is_1_in=False
            for j in range(label_weight):
                if(tmp[i*label_weight+j]==1):
                    is_1_in=True
            if(is_1_in):
                tmp_data_for_Y.append(1)
            else:
                tmp_data_for_Y.append(0)

        tmp_data_for_Y=np.reshape(tmp_data_for_Y,(-1,1))
        self.label_data=tmp_data_for_Y
        return

    def Make_Batch(self):
        data_length=len(self.label_data)
        
        if(data_length==0):
            print('Data not Ready')
            return
        
        number_batch=(int)(data_length/batch_size)

        tmp_mfcc=[]
        tmp_label=[]

        for i in range((int)(number_batch*test_rate)):
            tmp_mfcc.append(self.mfcc_data[i*batch_size:(i+1)*batch_size])
            tmp_label.append(self.label_data[i*batch_size:(i+1)*batch_size])
            self.train.data_length+=1
        self.train.mfcc_data=tmp_mfcc
        self.train.label_data=tmp_label

        tmp_mfcc=[]
        tmp_label=[]
        for i in range((int)(number_batch*test_rate),number_batch):
            tmp_mfcc.append(self.mfcc_data[i*batch_size:(i+1)*batch_size])
            tmp_label.append(self.label_data[i*batch_size:(i+1)*batch_size])
            self.test.data_length+=1
        self.test.mfcc_data=tmp_mfcc
        self.test.label_data=tmp_label
        return

    def Return_a_Sequence_data(array,index):
        temp=array[index*seq_length:(index+1)*seq_length]
        temp=np.reshape(temp,(-1))
        return temp     


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
            layer2=20
            layer3=1000
            layer4=10
            layer5=10
            layer6=100
            layer_output=no_output

            self.layer_input=seq_length*layer6
            self.layer1=10
            self.layer2=800
            self.layer3=100
            self.layer4=40            
            self.layer_output=no_output
            label_length=(int)(seq_length/10/label_weight)

            self.X=tf.placeholder(tf.float32,[None,seq_length,vec_per_frame])
            self.Y=tf.placeholder(tf.int32,[None,label_length,no_output])
            self.Y_transcripts=tf.placeholder(tf.int32,[None,label_length])
            self.keep_prob=tf.placeholder(tf.float32)
            self.data_length=tf.placeholder(tf.int32)
            
            layer1_cell=tf.contrib.rnn.BasicLSTMCell(num_units=layer1)
            layer2_cell=tf.contrib.rnn.BasicLSTMCell(num_units=layer2)
            layer3_cell=tf.contrib.rnn.BasicLSTMCell(num_units=layer3)
            layer4_cell=tf.contrib.rnn.BasicLSTMCell(num_units=layer4)
            layer5_cell=tf.contrib.rnn.BasicLSTMCell(num_units=layer5)
            layer6_cell=tf.contrib.rnn.BasicLSTMCell(num_units=layer6)            
                        
            multi_cell=tf.contrib.rnn.MultiRNNCell([
                                                    
                                                    layer6_cell])
            
            rnn_outputs,_=tf.nn.dynamic_rnn(multi_cell,self.X,dtype=tf.float32)

            fc_inputs=tf.reshape(rnn_outputs,[-1,self.layer_input])
           
            W=tf.get_variable('W',[self.layer_input,self.layer_output],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.Variable(tf.random_normal([self.layer_output]))
            L1=(tf.matmul(fc_inputs,W)+b)
            
            self.outputs=tf.reshape(L1,(-1,label_length,no_output))           
            
            
        self.weights=tf.ones([self.data_length,label_length])        
        self.cost=tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=self.outputs,targets=self.Y_transcripts,weights=self.weights))
        self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def show_cost(self,x,y,keep_prob=1.0):
        return self.sess.run(self.cost,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})
      
    def predict(self,x_test,keep_prob=1.0):

        return self.sess.run(tf.argmax(self.outputs,axis=2),feed_dict={self.X:x_test,self.keep_prob:keep_prob})
  
    def output(self,x_test,keep_prob=1.0):
        return self.sess.run(self.outputs,feed_dict={self.X:x_test,self.keep_prob:keep_prob})

    def train(self,x_data,y_data,data_length,keep_prob=0.8):
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
pass

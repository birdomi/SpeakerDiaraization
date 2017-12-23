import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import python_speech_features

no_output =2
vec_per_sec=10
seq_length=30
vec_pec_frame=33
learning_rate=0.0001

class file(object):
    def Return_Num_1(array):
        count=0
        for i in range(len(array)):
            if(array[i]==1):
                count=count+1
        return count

    def Extract_Mfcc_OneFile(s):
        fs,audio=wav.read(s)
        mfcc=python_speech_features.mfcc(audio,samplerate=fs)
        return mfcc

    def open_mfcc(start, end,s=''):
       sp = np.loadtxt(s+str(start)+'.txt')
       sp = sp[0:vec_per_sec]
       for i in range(start+1,end+1):
          temp=np.loadtxt(s+str(i)+'.txt')
          temp=temp[0:vec_per_sec]
          sp=np.append(sp,temp,axis=0)    
       return sp

    def extract_mfcc(start,end, s=''):
        fs,audio=wav.read(s+str(start)+'.wav')
        mfcc = python_speech_features.mfcc(audio,samplerate=fs)
        for i in range(start+1,end+1):
            temp_fs,temp_audio=wav.read(s+str(i)+'.wav')
            temp_mfcc=python_speech_features.mfcc(temp_audio,samplerate=temp_fs)
            mfcc=np.append(mfcc,temp_mfcc,axis=0)
        mfcc=np.reshape(mfcc,(-1,(int)(vec_per_sec),13))
        return mfcc

    def extract_mfcc_for_fc(start,end,s=''):
        fs,audio=wav.read(s+str(start)+'.wav')
        mfcc = python_speech_features.mfcc(audio,samplerate=fs)
        for i in range(start+1,end+1):
            temp_fs,temp_audio=wav.read(s+str(i)+'.wav')
            temp_mfcc=python_speech_features.mfcc(temp_audio,samplerate=temp_fs)
            mfcc=np.append(mfcc,temp_mfcc,axis=0)
        mfcc=np.reshape(mfcc,(-1,13*vec_per_sec))
        return mfcc

    def open_whole_mfcc(s):
        sp=np.loadtxt(s+'.txt')
        return sp
        
    def open_speakerinfo(filename):
        s=np.loadtxt(filename+".txt")
        return s

class Det_model:
    
    def __init__(self, sess,name):
        self.sess=sess
        self.name=name       
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):            
            self.X=tf.placeholder(tf.float32,[None,(int)(vec_per_sec),13])
            self.Y=tf.placeholder(tf.int32,[None,no_output])
            self.keep_prob=tf.placeholder(tf.float32)

            layer1_cell=tf.contrib.rnn.BasicRNNCell(num_units=layer1,activation=tf.nn.relu)
            
            multi_cell=tf.contrib.rnn.MultiRNNCell([layer1_cell])
            outputs,_=tf.nn.dynamic_rnn(multi_cell,self.X,dtype=tf.float32)
            outputs=tf.nn.dropout(outputs,keep_prob=self.keep_prob)

            sequence=layer1*(int)(vec_per_sec)
            X_for_FC=tf.reshape(outputs,[-1,sequence])

            W1=tf.get_variable("W1",[sequence,no_output],initializer=tf.contrib.layers.xavier_initializer())
            b1=tf.Variable(tf.random_normal([no_output]))
            L1=(tf.matmul(X_for_FC,W1)+b1)
            
            self.outputs=tf.reshape(L1,[-1,no_output])

        
        self.cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs,labels=self.Y))
        self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
    def show_cost(self,x,y,keep_prob=1.0):
        return self.sess.run(self.cost,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})
      
    def predict(self,x_test,keep_prob=1.0):
        return self.sess.run(tf.argmax(self.outputs,axis=1),feed_dict={self.X:x_test,self.keep_prob:keep_prob})
  
    def output(self,x_test,keep_prob=1.0):
        return self.sess.run(self.outputs,feed_dict={self.X:x_test,self.keep_prob:keep_prob})

    def train(self,x_data,y_data,keep_prob=0.7):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x_data,self.Y:y_data,self.keep_prob:keep_prob})

    def accuracy(self,x_data,y_data):
        x_prediction=tf.reshape(self.predict(x_data),(-1,1))
        y_prediction=tf.reshape(y_data,(-1,1))
        correct=tf.equal(x_prediction,tf.cast(y_prediction,tf.int64))
        return self.sess.run(tf.reduce_mean(tf.cast(correct,tf.float32)))

class RNN_model:
    
    def __init__(self, sess,name,):
        self.sess=sess
        self.name=name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):            
            layer1=100
            layer2=no_output
            FC1=50

            self.X=tf.placeholder(tf.float32,[None,seq_length,130])
            self.Y=tf.placeholder(tf.int32,[None,seq_length,no_output])
            self.Y_transcripts=tf.placeholder(tf.int32,[None,seq_length])
            self.keep_prob=tf.placeholder(tf.float32)
            self.data_length=tf.placeholder(tf.int32)

            layer1_cell=tf.contrib.rnn.BasicRNNCell(num_units=layer1,activation=tf.nn.relu)
            layer2_cell=tf.contrib.rnn.BasicRNNCell(num_units=layer2,activation=tf.nn.relu)
            
            multi_cell=tf.contrib.rnn.MultiRNNCell([layer1_cell,layer2_cell])
            
            outputs,_=tf.nn.dynamic_rnn(multi_cell,self.X,dtype=tf.float32)
            self.outputs=tf.nn.dropout(outputs,keep_prob=self.keep_prob)
            
        self.weights=tf.ones([self.data_length,seq_length])
        self.cost=tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=self.outputs,targets=self.Y_transcripts,weights=self.weights))
        self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def show_cost(self,x,y,keep_prob=1.0):
        return self.sess.run(self.cost,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})
      
    def predict(self,x_test,keep_prob=1.0):
        return self.sess.run(tf.argmax(self.outputs,axis=1),feed_dict={self.X:x_test,self.keep_prob:keep_prob})
  
    def output(self,x_test,keep_prob=1.0):
        return self.sess.run(self.outputs,feed_dict={self.X:x_test,self.keep_prob:keep_prob})

    def train(self,x_data,y_data,data_length,keep_prob=0.7):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x_data,self.Y_transcripts:y_data,self.data_length:data_length,self.keep_prob:keep_prob})

    def accuracy(self,x_data,y_data):
        x_prediction=tf.reshape(self.predict(x_data),(-1,1))
        y_prediction=tf.reshape(y_data,(-1,1))
        correct=tf.equal(x_prediction,tf.cast(y_prediction,tf.int64))
        return self.sess.run(tf.reduce_mean(tf.cast(correct,tf.float32)))

class FC_model:    
    def __init__(self, sess, name):
        self.sess=sess
        self.name=name       
        self._build_net()
        

    def _build_net(self):
        self.layer_input=13*vec_per_sec
        self.layer1=3000
        self.layer2=3000
        self.layer3=3000
        self.layer4=1000
        self.layer5=1000
        self.layer6=1000
        self.layer7=500
        self.layer8=500
        self.layer9=100
        self.layer10=100
        self.layer11=50
        self.layer12=25
        self.layer13=10
        self.layer_output=no_output

        self.X=tf.placeholder(tf.float32,[None,self.layer_input])
        self.Y=tf.placeholder(tf.float32,[None,self.layer_output])
        self.keep_prob=tf.placeholder(tf.float32)
        with tf.variable_scope(self.name):
            W1=tf.get_variable('W1',[self.layer_input,self.layer1],initializer=tf.contrib.layers.xavier_initializer())
            b1=tf.Variable(tf.random_normal([self.layer1]))
            L1=tf.nn.relu(tf.matmul(self.X,W1)+b1)

            W2=tf.get_variable('W2',[self.layer1,self.layer2],initializer=tf.contrib.layers.xavier_initializer())
            b2=tf.Variable(tf.random_normal([self.layer2]))
            L2=tf.nn.relu(tf.matmul(L1,W2)+b2)

            W3=tf.get_variable('W3',[self.layer2,self.layer3],initializer=tf.contrib.layers.xavier_initializer())
            b3=tf.Variable(tf.random_normal([self.layer3]))
            L3=tf.nn.relu(tf.matmul(L2,W3)+b3)

            W4=tf.get_variable('W4',[self.layer3,self.layer4],initializer=tf.contrib.layers.xavier_initializer())
            b4=tf.Variable(tf.random_normal([self.layer4]))
            L4=tf.nn.relu(tf.matmul(L3,W4)+b4)

            W5=tf.get_variable('W5',[self.layer4,self.layer5],initializer=tf.contrib.layers.xavier_initializer())
            b5=tf.Variable(tf.random_normal([self.layer5]))
            L5=tf.nn.relu(tf.matmul(L4,W5)+b5)

            W6=tf.get_variable('W6',[self.layer5,self.layer6],initializer=tf.contrib.layers.xavier_initializer())
            b6=tf.Variable(tf.random_normal([self.layer6]))
            L6=tf.nn.relu(tf.matmul(L5,W6)+b6)

            W7=tf.get_variable('W7',[self.layer6,self.layer7],initializer=tf.contrib.layers.xavier_initializer())
            b7=tf.Variable(tf.random_normal([self.layer7]))
            L7=tf.nn.relu(tf.matmul(L6,W7)+b7)

            W8=tf.get_variable('W8',[self.layer7,self.layer8],initializer=tf.contrib.layers.xavier_initializer())
            b8=tf.Variable(tf.random_normal([self.layer8]))
            L8=tf.nn.relu(tf.matmul(L7,W8)+b8)

            W9=tf.get_variable('W9',[self.layer8,self.layer9],initializer=tf.contrib.layers.xavier_initializer())
            b9=tf.Variable(tf.random_normal([self.layer9]))
            L9=tf.nn.relu(tf.matmul(L8,W9)+b9)

            W10=tf.get_variable('W10',[self.layer9,self.layer10],initializer=tf.contrib.layers.xavier_initializer())
            b10=tf.Variable(tf.random_normal([self.layer10]))
            L10=(tf.matmul(L9,W10)+b10)

            W11=tf.get_variable('W11',[self.layer10,self.layer11],initializer=tf.contrib.layers.xavier_initializer())
            b11=tf.Variable(tf.random_normal([self.layer11]))
            L11=(tf.matmul(L10,W11)+b11)

            W12=tf.get_variable('W12',[self.layer11,self.layer12],initializer=tf.contrib.layers.xavier_initializer())
            b12=tf.Variable(tf.random_normal([self.layer12]))
            L12=(tf.matmul(L11,W12)+b12)

            W13=tf.get_variable('W13',[self.layer12,self.layer13],initializer=tf.contrib.layers.xavier_initializer())
            b13=tf.Variable(tf.random_normal([self.layer13]))
            L13=(tf.matmul(L12,W13)+b13)

            W14=tf.get_variable('W14',[self.layer13,self.layer_output],initializer=tf.contrib.layers.xavier_initializer())
            b14=tf.Variable(tf.random_normal([self.layer_output]))
            self.hypothesis=(tf.matmul(L13,W14)+b14)

        self.cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.hypothesis,labels=self.Y))
        self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
    
    def show_cost(self,x,y,keep_prob=1.0):
        return self.sess.run(self.cost,feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})
    
    def predict(self,x_test,keep_prob=1.0):
        return self.sess.run(tf.argmax(self.hypothesis,axis=1),feed_dict={self.X:x_test,self.keep_prob:keep_prob})
    
    def output(self,x_test,keep_prob=1.0):
        return self.sess.run(self.outputs,feed_dict={self.X:x_test,self.keep_prob:keep_prob})

    def train(self,x_data,y_data,keep_prob=0.7):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x_data,self.Y:y_data,self.keep_prob:keep_prob})

    def accuracy(self,x_data,y_data):
        correct_prediction=tf.equal(tf.argmax(self.sess.run(self.hypothesis,feed_dict={self.X:x_data}),1),tf.argmax(y_data,1))
        return self.sess.run(tf.reduce_mean(tf.cast(correct_prediction,tf.float32)))
    pass




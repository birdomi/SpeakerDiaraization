import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import python_speech_features
import winsound
from sklearn.preprocessing import MinMaxScaler

#config
time_windowSteps=80#*0.01s
time_windowLength=4*time_windowSteps#*0.01s
learning_rate=0.01
batch_size=1500
test_rate=0.7
threshold=0.25
check_range=24
#

vec_per_frame=39
no_output=2
beep_frequency=1500
beep_duration=2000

def beep():
    winsound.Beep(beep_frequency,beep_duration)

class Time_liner(object):
    X_data=None
    Y_data=None

    def Label_Time(self,label_text):
        s=np.loadtxt('timeLabel/'+label_text)    
        len80=(int)(len(s)/80)
        s=s[0:(len80-1)*80]
        
        length=(int)(len(s)/50)
        result=[]
        for i in range(length):
            step=50*i
            Label={
                '1':False,
                '2':False,
                '3':False,
                '4':False,
                '5':False
                }
            for i in range(step,step+10):
                if(s[i]==1):Label['1']=True
            for i in range(step+10,step+20):
                if(s[i]==1):Label['2']=True
            for i in range(step+20,step+30):
                if(s[i]==1):Label['3']=True
            for i in range(step+30,step+40):
                if(s[i]==1):Label['4']=True
            for i in range(step+40,step+50):
                if(s[i]==1):Label['5']=True
            
            alreadyAppend=False
            for check in range(1,5):
                if(Label[str(check)]==True):
                    if(alreadyAppend==False):
                        result.append(check)
                        alreadyAppend=True
            if(alreadyAppend==False):
                result.append(0)
        result=result[0:length]
        self.Y_data=np.reshape(result,(-1))
        return result

    def Line_Predict(self,RNN_result):
        length=(int)(len(RNN_result)/50)
        tmp=RNN_result[0:length*50]
        self.X_data=np.reshape(tmp,(-1,50))
        return self.X_data

    def __init__(self, sess,name,):
        self.sess=sess
        self.name=name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.layer_input=50
            self.layer1=150
            self.layer2=150
            self.layer3=100
            self.layer4=100
            self.layer5=50
            self.layer6=50
            self.layer_output=6
                        
            self.X=tf.placeholder(tf.float32,[None,self.layer_input],'X')
            self.Y=tf.placeholder(tf.int32,[None],'Y')
            self.Y_onehot=tf.one_hot(self.Y,self.layer_output,dtype=tf.int64)
            self.keep_prob=tf.placeholder(tf.float32)
                        
            W1=tf.get_variable('W1',[self.layer_input,self.layer1],tf.float32,tf.contrib.layers.xavier_initializer())
            b1=tf.Variable(tf.random_normal([self.layer1]))
            L1=tf.nn.relu(tf.matmul(self.X,W1)+b1)
            L1=tf.nn.dropout(L1,self.keep_prob)
            
            W2=tf.get_variable('W2',[self.layer1,self.layer2],tf.float32,tf.contrib.layers.xavier_initializer())
            b2=tf.Variable(tf.random_normal([self.layer2]))
            L2=tf.nn.relu(tf.matmul(L1,W2)+b2)
            L2=tf.nn.dropout(L2,self.keep_prob)
            
            W3=tf.get_variable('W3',[self.layer2,self.layer3],tf.float32,tf.contrib.layers.xavier_initializer())
            b3=tf.Variable(tf.random_normal([self.layer3]))
            L3=tf.nn.relu(tf.matmul(L2,W3)+b3)
            L3=tf.nn.dropout(L3,self.keep_prob)
            
            W4=tf.get_variable('W4',[self.layer3,self.layer4],tf.float32,tf.contrib.layers.xavier_initializer())
            b4=tf.Variable(tf.random_normal([self.layer4]))
            L4=tf.nn.relu(tf.matmul(L3,W4)+b4)
            L4=tf.nn.dropout(L4,self.keep_prob)
            
            W5=tf.get_variable('W5',[self.layer4,self.layer5],tf.float32,tf.contrib.layers.xavier_initializer())
            b5=tf.Variable(tf.random_normal([self.layer5]))
            L5=tf.nn.relu(tf.matmul(L4,W5)+b5)
            L5=tf.nn.dropout(L5,self.keep_prob)

            W6=tf.get_variable('W6',[self.layer5,self.layer6],tf.float32,tf.contrib.layers.xavier_initializer())
            b6=tf.Variable(tf.random_normal([self.layer6]))
            L6=tf.nn.relu(tf.matmul(L5,W6)+b6)
            L6=tf.nn.dropout(L6,self.keep_prob)

            W7=tf.get_variable('W7',[self.layer6,self.layer_output],tf.float32,tf.contrib.layers.xavier_initializer())
            b7=tf.Variable(tf.random_normal([self.layer_output]))
            L7=(tf.matmul(L6,W7)+b7)
        
        self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L7,labels=self.Y_onehot))
        self.optimizer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)
        self.out=tf.argmax(L7,axis=1,output_type=tf.int32)
        self.correct=tf.equal(self.out,self.Y)
        self.acc=tf.reduce_mean(tf.cast(self.correct,tf.float32))
    
    def train(self,x,y,keep_prob=0.9):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x,self.Y:y,self.keep_prob:keep_prob})
    def output(self,x,keep_prob=1.0):
        return self.sess.run(self.out,feed_dict={self.X:x,self.keep_prob:keep_prob})
    def accuracy(self,x,y,keep_prob=1.0):
        x_prediction=np.reshape(self.output(x),[-1])
        y_prediction=np.reshape(y,[-1])
        
        check_for_changePoint=0
        check_for_falseAlarm=0
        total_changePoint=0
        total_non_changePoint=0                
        accuracy_changePoint=0.0
        accuracy_non_changePoint=0.0
        for i in range(self.sess.run(tf.size(y_prediction))):            
            if(y_prediction[i]==0):
                total_non_changePoint=total_non_changePoint+1
                if(x_prediction[i]==0):
                    check_for_falseAlarm=check_for_falseAlarm+1
            else:
                total_changePoint=total_changePoint+1
                if(x_prediction[i]==y_prediction[i]):
                    check_for_changePoint=check_for_changePoint+1

        if(total_changePoint!=0):
            accuracy_changePoint=(float)(check_for_changePoint/total_changePoint)
        if(total_non_changePoint!=0):
            accuracy_non_changePoint=(float)(check_for_falseAlarm/total_non_changePoint)
                
        print('ChangePoint Number: ',total_changePoint,' Correct: ',check_for_changePoint)
        print('NoneChangePoint Number: ',total_non_changePoint,' Correct: ',check_for_falseAlarm)
        return accuracy_changePoint,accuracy_non_changePoint
    def SaveModel(self):        
        saver=tf.train.Saver()
        saver.save(self.sess,'timeLinerModel/model')
    def RestoreModel(self):        
        saver=tf.train.Saver()
        saver.restore(self.sess,'timeLinerModel/model')

    def Save_Result(self,x,filename,keep_prob=1.0):
        outputArray=self.output(x)

        s=[]
        for i in range(len(outputArray)):
            for j in range(1,6):
                if(outputArray[i]==j):
                    s.append(str(-0.05+i*0.5+j*0.1))
        np.savetxt(filename+'.bdr',s,fmt='%s',delimiter='\n')




class mfcc_label_data():
    def __init__(self):
        self.mfcc_data=[]
        self.label_data=[]
        self.data_length=0
    
class Data(object):
    data={
        0:mfcc_label_data(),
        1:mfcc_label_data(),
        2:mfcc_label_data(),
        3:mfcc_label_data(),
        4:mfcc_label_data(),
        5:mfcc_label_data(),
        6:mfcc_label_data(),
        7:mfcc_label_data(),
        8:mfcc_label_data(),
        9:mfcc_label_data(),
        10:mfcc_label_data(),
        11:mfcc_label_data(),
        12:mfcc_label_data(),
        13:mfcc_label_data(),
        14:mfcc_label_data(),
        15:mfcc_label_data()
    }

    def Mfcc(self,s):        
        sig,f=wav.read('Wave/'+s)
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
        mfcc=self.Mfcc(mfccpath)
        label=self.Labeling(labelpath)        

        length=len(label)
        number_batch=(int)(length/batch_size)
        
        for i in range(number_batch):
            self.data[dataname].mfcc_data.append(mfcc[i*batch_size:(i+1)*batch_size])
            self.data[dataname].label_data.append(label[i*batch_size:(i+1)*batch_size])
            self.data[dataname].data_length+=1
        self.data[dataname].mfcc_data.append(mfcc[number_batch*batch_size:])
        self.data[dataname].label_data.append(label[number_batch*batch_size:])
        self.data[dataname].data_length+=1
        print(self.data[dataname].mfcc_data[number_batch].shape)
        

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
            self.learning_rate=tf.placeholder(tf.float32)

            
            
            layer1_cell=tf.contrib.rnn.LSTMCell(num_units=layer1,activation=tf.nn.relu,initializer=tf.contrib.layers.xavier_initializer())
            layer1_cell=tf.contrib.rnn.DropoutWrapper(layer1_cell,self.keep_prob,self.keep_prob)
            layer4_cell=tf.contrib.rnn.LSTMCell(num_units=layer4,activation=tf.nn.relu,initializer=tf.contrib.layers.xavier_initializer())
            layer4_cell=tf.contrib.rnn.DropoutWrapper(layer4_cell,self.keep_prob,self.keep_prob)

            multi_cell=tf.contrib.rnn.MultiRNNCell([layer1_cell,layer4_cell])

            layer2_output,_=tf.nn.bidirectional_dynamic_rnn(multi_cell,multi_cell,self.X,dtype=tf.float32)
            rnn_output=tf.concat(layer2_output,2)
            fc_inputs=tf.reshape(rnn_output,[-1,self.layer_input])
                        
            W1=tf.get_variable('W1',[self.layer_input,self.layer1],tf.float32,tf.contrib.layers.xavier_initializer())
            b1=tf.Variable(tf.random_normal([self.layer1]))
            L1=tf.nn.relu(tf.matmul(fc_inputs,W1)+b1)
            L1=tf.nn.dropout(L1,self.keep_prob)

            W2=tf.get_variable('W2',[self.layer1,self.layer2],tf.float32,tf.contrib.layers.xavier_initializer())
            b2=tf.Variable(tf.random_normal([self.layer2]))
            L2=tf.nn.relu(tf.matmul(L1,W2)+b2)
            L2=tf.nn.dropout(L2,self.keep_prob)

            W3=tf.get_variable('W3',[self.layer2,no_output],tf.float32,tf.contrib.layers.xavier_initializer())
            b3=tf.Variable(tf.random_normal([no_output]))
            L3=(tf.matmul(L2,W3)+b3)

            self.outputs=tf.reshape(L3,(-1,time_windowLength,no_output))            
            
            
        self.cost=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.outputs,labels=self.Y_transcripts))
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def show_cost(self,x,y,keep_prob=1.0):
        return self.sess.run(self.cost,feed_dict={self.X:x,self.Y_transcripts:y,self.keep_prob:keep_prob})
       
    def predict(self,x_test,keep_prob=1.0):
        return self.sess.run(tf.argmax(self.outputs,axis=2),feed_dict={self.X:x_test,self.keep_prob:keep_prob})
  
    def output(self,x_test,keep_prob=1.0):
        return self.sess.run(self.outputs,feed_dict={self.X:x_test,self.keep_prob:keep_prob})

    def train(self,x_data,y_data,learn,keep_prob=1.0):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x_data,self.Y_transcripts:y_data,self.learning_rate:learn,self.keep_prob:keep_prob})

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

    def Final_Accuracy(self,x_data,y_data):
        x_prediction=np.reshape(x_data,(-1))
        y_prediction=np.reshape(y_data,(-1))
        

        number_of_changePoint=0
        number_of_correct=0
        for i in range(len(y_prediction)):
            correct=False
            if(y_prediction[i]==1):
                number_of_changePoint+=1
                for chk in range(1,check_range):
                    if(i-chk>=0):
                        if(x_prediction[i-chk]==1):
                            correct=True
                if(x_prediction[i]==1):
                    correct=True
                for chk in range(1,check_range):
                    if(i+chk<len(y_prediction)):
                        if(x_prediction[i+chk]==1):
                            correct=True
            if(correct==True):
                number_of_correct+=1
        acc=number_of_correct/number_of_changePoint

        return acc
pass


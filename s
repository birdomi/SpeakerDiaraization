class ICSI_Data2(Data):
    def MakeData(self, mfcc_path, label_path):        
        self.__BatchMRT(mfcc_path,label_path)

    def __BatchMFCC(self,mfcc_path):
        self.__hamming=np.reshape(np.hamming(windowsize),(windowsize,1))
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
                for time in range(0,1):
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
    
    def __BatchMRT(self,mfcc_path,label_path):        
        #
        self.__Label=self.__LabelMRT(label_path)
        print(len(self.__Label))

        self.__data_length=int(len(self.__Label)/windowstep)
        self.__num_seq=self.__data_length-windowmul+1
        self.__label_in=[]
        self.__label=[]

        for i in range(self.__num_seq):
            self.__tempSequence=self.__Label[i*windowstep:(i+windowmul)*windowstep]
            self.__isThereOnly1=0
            self.__where1is=-1
            for j in range(windowsize):
                self.__isThereOnly1+=self.__tempSequence[j]
                if(self.__tempSequence[j]==1):
                    self.__where1is=j
            if(self.__isThereOnly1<=1):
                self.__label_in.append(i)
                if(self.__isThereOnly1==0):
                    self.__label.append(10)
                else:
                    self.__Label_of_1=0
                    if(self.__where1is>150):
                        self.__where1is=self.__where1is-150
                    else:
                        self.__where1is=150-self.__where1is

                    if(self.__where1is>135):
                        self.__Label_of_1=9
                    elif(self.__where1is>120):
                        self.__Label_of_1=8
                    elif(self.__where1is>105):
                        self.__Label_of_1=7
                    elif(self.__where1is>90):
                        self.__Label_of_1=6
                    elif(self.__where1is>75):
                        self.__Label_of_1=5
                    elif(self.__where1is>60):
                        self.__Label_of_1=4
                    elif(self.__where1is>45):
                        self.__Label_of_1=3
                    elif(self.__where1is>30):
                        self.__Label_of_1=2
                    elif(self.__where1is>15):
                        self.__Label_of_1=1
                    else:
                        self.__Label_of_1=0
                    self.__label.append(self.__Label_of_1)               
        
        self.__label=np.reshape(self.__label,(-1,1,1))
        print(self.__label.shape)
        
        self.__hamming=np.reshape(np.hamming(windowsize),(windowsize,1))
        self.__Tmfcc=self._ExtractMFCC(mfcc_path)
        
        self.__data_length=int(len(self.__Tmfcc)/windowstep)
        self.__num_seq=self.__data_length-windowmul+1
        
        self.__tmp=[]
        for i in range(self.__num_seq):
            self.__tempSequence=self.__Tmfcc[i*windowstep:(i+windowmul)*windowstep]            
            self.__tempSequence=np.multiply(self.__tempSequence,self.__hamming)
            self.__tmp.append(self.__tempSequence)

        self.__tmp=np.reshape(self.__tmp,(-1,windowsize,39))   
        
        self.__mfcc=[]
        for i in range(len(self.__label_in)):
            self.__mfcc.append(self.__tmp[self.__label_in[i]])


        self.__mfcc=np.reshape(self.__mfcc,(-1,windowsize,39))
        print(self.__mfcc.shape)
        self.__batch_num=int(len(self.__label)/batch_size)

        for i in range(self.__batch_num):
            self._MFCC.append(self.__mfcc[i*batch_size:(i+1)*batch_size])
            self.numberBatch+=1
                    
        for i in range(self.__batch_num):
            self._LABEL.append(self.__label[i*batch_size:(i+1)*batch_size])

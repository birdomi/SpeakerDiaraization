import tensorflow as tf
import numpy as np
import os
import ut
import sklearn.cluster

def Segment_Analysis(cluster,segment,cut):
	current_position=0
	segment_based_onCut=[]

	for i in range(len(cut)):
		start=segment[i][0]
		end=segment[i][1]
		for j in range(cut[i]):			
			if(j==0):
				segment_based_onCut.append([start,start+0.75,cluster[current_position]])
				current_position+=1
			elif(j==cut[i]-1):
				segment_based_onCut.append([start+0.25+0.5*j,end,cluster[current_position]])
				current_position+=1
			else:
				segment_based_onCut.append([start+0.25+0.5*j,start+0.25+(j+1)*0.5,cluster[current_position]])
				current_position+=1
			print(segment_based_onCut[current_position-1])
		
	k=len(segment_based_onCut)
	print(k)
	i=0
	while(i<k-1):
		i+=1		
		print(i,k)
		if(segment_based_onCut[i][2]==segment_based_onCut[i-1][2]):
			if(segment_based_onCut[i][0]==segment_based_onCut[i-1][1]):
				segment_based_onCut[i-1][1]=segment_based_onCut[i][1]
				del segment_based_onCut[i]
				k-=1;i-=1
	file=open('ttttt'+'.bdr','w')
	for i in range(len(segment_based_onCut)):
		print('('+str(segment_based_onCut[i][0])+', '+str(segment_based_onCut[i][1])+') '+str(segment_based_onCut[i][2])+'\n')
		file.write('('+str(segment_based_onCut[i][0])+', '+str(segment_based_onCut[i][1])+') '+str(segment_based_onCut[i][2])+'\n')
	file.close()






CALLHOME=ut.CALLHOME_Data()


wavPath='CALLHOME/Signals'
labPath='CALLHOME/transcripts'
mfcc=os.listdir(wavPath)
label_path=os.listdir(labPath)

mfcc_path=[]
for i in range(len(mfcc)):
    if(mfcc[i][-4:]=='.csv'):
        pass
    else:
        mfcc_path.append(mfcc[i]) 




print(len(mfcc_path),len(label_path))
datanum=1#len(mfcc_path)

acc=[];no=[]
for i in range(datanum):  
    #label,speaker,startPoint,endPoint=CALLHOME.return_Labeling(labPath+'/'+label_path[i])
    #print(startPoint,endPoint)
    #print(speaker.shape)

    sess=tf.Session()
    rnn=ut.VAD_Model(sess,'RNN_',0.001)
    rnn.Restore('model/VADM/VAD')#application_path+

    mfcc_x=ut.INPUT_Data().Get_Data('3_일반_구어_대만_고대_5급_인터뷰_0001.wav')
    print(mfcc_x.shape)
    timesegment=rnn.voiceTimeSegment(mfcc_x)
    #print(timesegment)
    #timesegment,groundTruth=CALLHOME.SegmentGroundTruth(timesegment,startPoint,endPoint,speaker)
    #print(len(timesegment),len(groundTruth))

    tf.reset_default_graph()
    sess=tf.Session()
    vf=ut.VoiceFeature(sess,'model/save/SR')
    f,segment,cut_segment=vf.Extract_timeSegment('3_일반_구어_대만_고대_5급_인터뷰_0001.wav',timesegment)
    tf.reset_default_graph()

    k=sklearn.cluster.k_means(f,2)

    Segment_Analysis(k[1],segment,cut_segment)
    


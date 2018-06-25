import tensorflow as tf
import numpy as np
import os
import sys
import ut
import sklearn.cluster



def Segment_Analysis2(cluster,segment,cut,border,speakerNumber):
    current_position=0
    result=[]
    #print(len(cluster),len(segment),len(cut))
    for i in range(len(cut)):        
        start=segment[i][0]
        end=segment[i][1]        
        
        if(cut[i]>1):
            s={}
            for i in range(speakerNumber):
                s[i]=0
            for j in range(cut[i]):
                s[cluster[current_position]]+=1
                current_position+=1
            speaker=[a for a,v in s.items() if v==max(s.values())][0]
            result.append((start,end,speaker,max(s.values())/sum(s.values())))
        else:
            result.append((start,end,speakerNumber,0.0))
            
    file=open(border+'.bdr','w')
    for i in range(len(result)):
        #print('({:.3f}, {:.3f}) {:d} {:.2f}'.format(result[i][0],result[i][1],result[i][2],result[i][3]))
        file.write('({:.3f}, {:.3f}) {:d} {:.2f}\n'.format(result[i][0],result[i][1],result[i][2],result[i][3]))
    file.close()


def readBDR(bdr):
    f=open(bdr,'r')
    lines=f.readlines()

    s=[]
    for line in lines:
        d=line.split(' ')
        s.append((float(d[0]),float(d[1][:-2])))
    return s

if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the pyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

print(application_path)
if(len(sys.argv)==4):
    waveFile=sys.argv[1]
    index=waveFile.find('.')
    waveName=waveFile[0:index]
    speakerNumber=sys.argv[2]
    segment=sys.argv[3]
    borderFile=waveName

elif(len(sys.argv)==4):
    waveFile=sys.argv[1]
    speakerNumber=sys.argv[2]
    segment=sys.argv[3]
    borderFile=sys.argv[4]

else:
    print('다음과 같이 실행시켜주세요.\n')
    print('detetor.exe    wavefileDirectory  SpeakerNumber  BoundaryFile (Option)borderfileName\n')
    print('디텍터.exe     입력시킬wav파일  wave안에 들어있는 화자 수   발화구간BDR파일  (옵션)출력시킬bdr파일이름\n')
    sys.exit()

ut.progressBar(0,100,50)
segment=readBDR(segment)
tf.reset_default_graph()
sess=tf.Session()
vf=ut.VoiceFeature(sess,str(application_path+'\\sr1166p\\sr').encode('UTF-8'))
f,segment,cut=vf.Extract_timeSegment(waveFile,segment)

k=sklearn.cluster.k_means(f,int(speakerNumber))
ut.progressBar(100,100,50)
Segment_Analysis2(k[1],segment,cut,borderFile,2)


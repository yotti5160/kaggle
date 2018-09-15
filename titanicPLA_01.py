import numpy as np
    
#sign
def sign(x): return 1 if x>0 else -1

#read file 
def readFile(location):
    f = open(location,"r")
    data = []
    for line in f:
        tmp=line.split(',')
#        tmpNum=[1] #x0=1 for all data
        tmpNum=[]
        for t in tmp:
            tmpNum.append(t)
        data.append(tmpNum) 
    f.close()
    return data

def processData(orgData):
    orgData=orgData[1:] #No first row
    # calculate avg age
    total, count=0, 0
    for d in orgData:
        if d[6]:
            total+=float(d[6])
            count+=1
    avgAge=total/count
        
    newData=[]
    for d in orgData:
        tmp=None
        if int(d[1])==1:
            tmp=[1, 1]
        else:
            tmp=[-1, 1]
        if d[2]=='1':
            tmp+=[1,0,0]
        elif d[2]=='2':
            tmp+=[0,1,0]
        else:
            tmp+=[0,0,1]
        if d[5]=='male':
            tmp+=[1,0]
        else:
            tmp+=[0,1]
        if d[6]:
            tmp.append(float(d[6]))
        else:
            tmp.append(avgAge)
        tmp.append(float(d[7]))
        tmp.append(float(d[8]))
        tmp.append(float(d[10]))
        # no cabin
        if d[12][0]=='S':
            tmp+=[1,0,0]
        elif d[12][0]=='C':
            tmp+=[0,1,0]
        else:
            tmp+=[0,0,1]
        newData.append(np.array(tmp))
    return newData

def processData2(orgData):
    orgData=orgData[1:] #No first row
    # calculate avg age
    total, count=0, 0
    for d in orgData:
        if d[6]:
            total+=float(d[6])
            count+=1
    avgAge=total/count
        
    newData=[]
    for d in orgData:
        tmp=[1]
        if d[1]=='1':
            tmp+=[1,0,0]
        elif d[1]=='2':
            tmp+=[0,1,0]
        else:
            tmp+=[0,0,1]
        if d[4]=='male':
            tmp+=[1,0]
        else:
            tmp+=[0,1]
        # Age
        if d[5]:
            tmp.append(float(d[5]))
        else:
            tmp.append(avgAge)
        tmp.append(float(d[6]))
        tmp.append(float(d[7]))
        if d[9]:
            tmp.append(float(d[9]))
        else:
            tmp.append(0)
        # no cabin
        if d[11][0]=='S':
            tmp+=[1,0,0]
        elif d[11][0]=='C':
            tmp+=[0,1,0]
        else:
            tmp+=[0,0,1]
        newData.append(np.array(tmp))
    return newData

#pocket PLA return pocketW
def PocketPLA(TrainingData, standard):
    updateCount=0
    w=np.zeros((len(TrainingData[0])-1,))
    pocketW=np.zeros((len(TrainingData[0])-1))
    pocketER=checkER(TrainingData, pocketW)
    allcorrect=False
    
    while updateCount<standard and not allcorrect:
        allcorrect=True
        for d in TrainingData:
            if d[0]!=sign(np.dot(w,d[1:])):
                for i in range(5):
                    w+=d[0]*d[1:]
                tmpER=checkER(TrainingData, w)
                if tmpER<pocketER:
                    pocketW=np.copy(w)
                    pocketER=tmpER
                allcorrect=False
                updateCount+=1
                if updateCount>=standard:
                    return pocketW
    return pocketW

#check error rate
def checkER(data, w):
    errorNum=0
    for d in data:
        if d[0]!=sign(np.dot(w,d[1:])):
            errorNum+=1
    return errorNum/len(data)

def runTest(data, w):
    ret=[]
    for d in data:
        ret.append(sign(np.dot(w,d)))
    return ret
        
def writeFile(startNum, result, location):
    f = open(location,"w")
    f.write('PassengerId,Survived\n')
    PassengerId=startNum
    for r in result:
        if r==1:
            f.write(str(PassengerId)+',1 \n')
        else:
            f.write(str(PassengerId)+',0 \n')
        PassengerId+=1
            
    f.close()
    return
        


# start here
tdLocation='C:/Users/Yotti/Desktop/kaggle_titanic/train.csv'
testLocation='C:/Users/Yotti/Desktop/kaggle_titanic/test.csv'
writeLocation='C:/Users/Yotti/Desktop/kaggle_titanic/results.csv'
def main():
    orgData=readFile(tdLocation)
    newData=processData(orgData)
    tmpw=PocketPLA(newData, 3000) #  PocketPLA with ??? updates
    
    testData=readFile(testLocation)
    newTestData=processData2(testData)
    result=runTest(newTestData, tmpw)
    writeFile(892, result, writeLocation)
    
    return checkER(newData, tmpw)

print(main())
    
            
        
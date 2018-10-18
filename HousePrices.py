import numpy as np
import random
  
traingDataLoc='C:/Users/Yotti/Desktop/homePrice/train.csv'

# =============================================================================
# OverallQual 17
# OverallCond 18
# YearBuilt 19
# ExterQual 27
# BsmtQual 30
# BsmtFinSF1 34
# TotalBsmtSF 38
# HeatingQC 40
# 1stFlrSF 43
# 2ndFlrSF 44
# GrLivArea 46
# KitchenQual 53
# TotRmsAbvGrd 54
# GarageCars 61
# GarageArea 62
#
# SalePrice 80
# =============================================================================

f = open(traingDataLoc, "r")
x, y = [], []
row_count=0
for line in f:
    if row_count==0:
        row_count+=1
        continue
    tmp=line.split(',')
    row=[]
    row.append(float(1))
    row.append(float(tmp[17])) # OverallQual 17
    row.append(float(tmp[18])) # OverallCond 18
    row.append(float(tmp[19])) # YearBuilt 19
    if tmp[27]=='Ex': # ExterQual 27
        row.append(5.0)
    elif tmp[27]=='Gd':
        row.append(4.0)
    elif tmp[27]=='TA':
        row.append(3.0)
    elif tmp[27]=='Fa':
        row.append(2.0)
    else:
        row.append(1.0)
        
    if tmp[30]=='Ex': # BsmtQual 30
        row+=[5.0, 0]
    elif tmp[30]=='Gd':
        row+=[4.0, 0]
    elif tmp[30]=='TA':
        row+=[3.0, 0]
    elif tmp[30]=='Fa':
        row+=[2.0, 0]
    elif tmp[30]=='Po':
        row+=[1.0, 0]
    else:
        row+=[0.0, 1]
        
    try:
        row+=[float(tmp[34]), 0] # BsmtFinSF1 34
    except:
        row+=[0, 1]
    try:
        row.append(float(tmp[38])) # TotalBsmtSF 38
    except:
        row.append(0.0)
    if tmp[40]=='Ex': # HeatingQC 40
        row.append(5.0)
    elif tmp[40]=='Gd':
        row.append(4.0)
    elif tmp[40]=='TA':
        row.append(3.0)
    elif tmp[40]=='Fa':
        row.append(2.0)
    else:
        row.append(1.0)
    row.append(float(tmp[43])) # 1stFlrSF 43
    row.append(float(tmp[44])) # 2ndFlrSF 44
    row.append(float(tmp[46])) # GrLivArea 46
    if tmp[53]=='Ex': # KitchenQual 53
        row.append(5.0)
    elif tmp[53]=='Gd':
        row.append(4.0)
    elif tmp[53]=='TA':
        row.append(3.0)
    elif tmp[53]=='Fa':
        row.append(2.0)
    else:
        row.append(1.0)
    row.append(float(tmp[54])) # TotRmsAbvGrd 54
    try:
        row.append(float(tmp[61])) # GarageCars 61
    except:
        row.append(0.0)
    try:
        row.append(float(tmp[62])) # GarageArea 62
    except:
        row.append(0)
    
    x.append(row)
    
    y.append(float(tmp[80]))
    row_count+=1
f.close()
# =============================================================================
# clean data(outlier)
# =============================================================================
rowsToDrop=[]
for i in range(len(x)):
    if x[i][13]>4000 and y[i]<200000:
        rowsToDrop.append(i)
while rowsToDrop:
    rowNum=rowsToDrop.pop()
    x.pop(rowNum)
    y.pop(rowNum)       


rowsToDrop=[]
avgPricePerArea=[]
for i in range(len(x)):
    val=y[i]/x[i][13]
    avgPricePerArea.append(val)
q25, q75 = np.percentile(avgPricePerArea, [25, 75])
iqr = q75 - q25
validRange=[q25-3*iqr, q75+3*iqr]
for i in range(len(avgPricePerArea)):
    if avgPricePerArea[i]<validRange[0] or avgPricePerArea[i]>validRange[1]:
        rowsToDrop.append(i)
while rowsToDrop:
    rowNum=rowsToDrop.pop()
    x.pop(rowNum)
    y.pop(rowNum)

# =============================================================================
# data feature scaling
# =============================================================================
x=np.array(x)
y=np.array(y)
l=len(x[0])
datasize=len(x)

meanStd=[[1, 0]]
# first term always 1, don't need scaling
for i in range(1,len(x[0])):
    tmp=[x[j][i] for j in range(datasize)]
    avg=np.mean(tmp)
    sd=np.std(tmp)
    meanStd.append([avg, sd])
    if sd==0:
        continue
    for j in range(datasize):
        x[j][i]=(x[j][i]-avg)/sd

# =============================================================================
# start training (Stochastic Gradient Descent)
# =============================================================================
w=[176560.30, 16086.82, 6347.22, 4656.87,\
    7894.87, 8429.57, 5855.59, 8254.60,\
   0.0, 8547.40, 914.98, 9094.54,\
   7153.32, 12681.19, 7721.99, 6472.99,\
   5632.95, 3725.10]
#w=np.zeros(l)
w=np.array(w)
repeat=50000
eta=0.1
etaLow=0

for i in range(repeat):
    #choose n for Stochastic Gradient Descent
    n=random.randint(0,datasize-1)
    grad=np.zeros(l)
    #compute grad
    for k in range(l):
        grad[k]=-2*x[n][k]*(y[n]-np.inner(w, x[n]))
    #updates etaLow
    etaLow+=grad.dot(grad)
    #updates w
    w=w-(eta/np.sqrt(etaLow))*grad
print(w)

# =============================================================================
# check error in training data    
# =============================================================================
error=0
for i in range(datasize):
    error+=abs(y[i]-np.inner(w, x[i]))
print(error/datasize)

# =============================================================================
#  out put prediction of test data
# =============================================================================
testDataLoc='C:/Users/Yotti/Desktop/homePrice/test.csv'

f = open(testDataLoc, "r")
x=[]
row_count=0
for line in f:
    if row_count==0:
        row_count+=1
        continue
    tmp=line.split(',')
    row=[]
    row.append(float(1))
    row.append(float(tmp[17])) # OverallQual 17
    row.append(float(tmp[18])) # OverallCond 18
    row.append(float(tmp[19])) # YearBuilt 19
    if tmp[27]=='Ex': # ExterQual 27
        row.append(5.0)
    elif tmp[27]=='Gd':
        row.append(4.0)
    elif tmp[27]=='TA':
        row.append(3.0)
    elif tmp[27]=='Fa':
        row.append(2.0)
    else:
        row.append(1.0)
        
    if tmp[30]=='Ex': # BsmtQual 30
        row+=[5.0, 0]
    elif tmp[30]=='Gd':
        row+=[4.0, 0]
    elif tmp[30]=='TA':
        row+=[3.0, 0]
    elif tmp[30]=='Fa':
        row+=[2.0, 0]
    elif tmp[30]=='Po':
        row+=[1.0, 0]
    else:
        row+=[0.0, 1]
        
    try:
        row+=[float(tmp[34]), 0] # BsmtFinSF1 34
    except:
        row+=[0, 1]
    try:
        row.append(float(tmp[38])) # TotalBsmtSF 38
    except:
        row.append(0.0)
    if tmp[40]=='Ex': # HeatingQC 40
        row.append(5.0)
    elif tmp[40]=='Gd':
        row.append(4.0)
    elif tmp[40]=='TA':
        row.append(3.0)
    elif tmp[40]=='Fa':
        row.append(2.0)
    else:
        row.append(1.0)
    row.append(float(tmp[43])) # 1stFlrSF 43
    row.append(float(tmp[44])) # 2ndFlrSF 44
    row.append(float(tmp[46])) # GrLivArea 46
    if tmp[53]=='Ex': # KitchenQual 53
        row.append(5.0)
    elif tmp[53]=='Gd':
        row.append(4.0)
    elif tmp[53]=='TA':
        row.append(3.0)
    elif tmp[53]=='Fa':
        row.append(2.0)
    else:
        row.append(1.0)
    row.append(float(tmp[54])) # TotRmsAbvGrd 54
    try:
        row.append(float(tmp[61])) # GarageCars 61
    except:
        row.append(0.0)
    try:
        row.append(float(tmp[62])) # GarageArea 62
    except:
        row.append(0)
    
    x.append(row)
    
    row_count+=1
f.close()

x=np.array(x)
l=len(x[0])
datasize=len(x)

# scaling by meanStd
for i in range(1,len(x[0])): # first term always 1, don't need scaling
    for j in range(datasize):
        if meanStd[i][1]==0:
            continue
        x[j][i]=(x[j][i]-meanStd[i][0])/meanStd[i][1]
        
result=[]
for i in range(datasize):
    result.append(np.inner(w, x[i]))

# write output file
rowNumber=1461
f=open('C:/Users/Yotti/Desktop/homePrice/output.csv', 'w')
f.write('Id,SalePrice\n')
for i in range(datasize):
    f.write(str(rowNumber)+','+str(result[i])+'\n')
    rowNumber+=1
f.close()

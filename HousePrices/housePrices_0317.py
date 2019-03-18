import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

pd.set_option('display.max_rows', 300)

trainingData = pd.read_csv('C:/Users/Yotti/Desktop/homePrice/train.csv')
testingData = pd.read_csv('C:/Users/Yotti/Desktop/homePrice/test.csv')

##missing data
#total = trainingData.isnull().sum().sort_values(ascending=False)
#percent = (trainingData.isnull().sum()/trainingData.shape[0]).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(25))

columnsToDrop=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage']
trainingData.drop(columnsToDrop, axis=1, inplace=True)
testingData.drop(columnsToDrop, axis=1, inplace=True)

##missing data
#total = trainingData.isnull().sum().sort_values(ascending=False)
#percent = (trainingData.isnull().sum()/trainingData.shape[0]).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(15))

##missing data
#total = testingData.isnull().sum().sort_values(ascending=False)
#percent = (testingData.isnull().sum()/testingData.shape[0]).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(30))


# fillna cause by no garage
trainingData['GarageType'].fillna('NA', inplace=True)
trainingData['GarageYrBlt'].fillna(trainingData['YearBuilt'], inplace=True)
trainingData['GarageFinish'].fillna('NA', inplace=True)
trainingData['GarageCond'].fillna('NA', inplace=True)
trainingData['GarageQual'].fillna('NA', inplace=True)

testingData['GarageType'].fillna('NA', inplace=True)
testingData['GarageYrBlt'].fillna(testingData['YearBuilt'], inplace=True)
testingData['GarageFinish'].fillna('NA', inplace=True)
testingData['GarageCond'].fillna('NA', inplace=True)
testingData['GarageQual'].fillna('NA', inplace=True)
testingData['GarageArea'].fillna(0.0, inplace=True)
testingData['GarageCars'].fillna(0.0, inplace=True)


# fillna cause by no basement
trainingData['BsmtFinType2'].fillna('NA', inplace=True)
trainingData['BsmtExposure'].fillna('NA', inplace=True)
trainingData['BsmtQual'].fillna('NA', inplace=True)
trainingData['BsmtFinType1'].fillna('NA', inplace=True)
trainingData['BsmtCond'].fillna('NA', inplace=True)

testingData['BsmtFinType2'].fillna('NA', inplace=True)
testingData['BsmtExposure'].fillna('NA', inplace=True)
testingData['BsmtQual'].fillna('NA', inplace=True)
testingData['BsmtFinType1'].fillna('NA', inplace=True)
testingData['BsmtCond'].fillna('NA', inplace=True)
testingData['BsmtFullBath'].fillna(0.0, inplace=True)
testingData['BsmtHalfBath'].fillna(0.0, inplace=True)
testingData['TotalBsmtSF'].fillna(0.0, inplace=True)
testingData['BsmtFinSF2'].fillna(0.0, inplace=True)
testingData['BsmtFinSF1'].fillna(0.0, inplace=True)
testingData['BsmtUnfSF'].fillna(0.0, inplace=True)


# fillna cause by no Masonry veneer
trainingData['MasVnrType'].fillna('None', inplace=True)
trainingData['MasVnrArea'].fillna(0.0, inplace=True)

testingData['MasVnrType'].fillna('None', inplace=True)
testingData['MasVnrArea'].fillna(0.0, inplace=True)


##missing data
#total = trainingData.isnull().sum().sort_values(ascending=False)
#percent = (trainingData.isnull().sum()/trainingData.shape[0]).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(5))
#
##missing data
#total = testingData.isnull().sum().sort_values(ascending=False)
#percent = (testingData.isnull().sum()/testingData.shape[0]).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(10))


#fill other missing data
trainingData['Electrical'].fillna(trainingData['Electrical'].mode()[0], inplace=True)

testingData['MSZoning'].fillna(trainingData['MSZoning'].mode()[0], inplace=True)
testingData['Utilities'].fillna(trainingData['Utilities'].mode()[0], inplace=True)
testingData['MSZoning'].fillna(trainingData['MSZoning'].mode()[0], inplace=True)
testingData['Exterior1st'].fillna(trainingData['Exterior1st'].mode()[0], inplace=True)
testingData['Exterior2nd'].fillna(trainingData['Exterior2nd'].mode()[0], inplace=True)
testingData['Functional'].fillna(trainingData['Functional'].mode()[0], inplace=True)
testingData['SaleType'].fillna(trainingData['SaleType'].mode()[0], inplace=True)
testingData['KitchenQual'].fillna(trainingData['KitchenQual'].mode()[0], inplace=True)


##missing data
#total = trainingData.isnull().sum().sort_values(ascending=False)
#percent = (trainingData.isnull().sum()/trainingData.shape[0]).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(5))
#
##missing data
#total = testingData.isnull().sum().sort_values(ascending=False)
#percent = (testingData.isnull().sum()/testingData.shape[0]).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(6))



#Add some combine attributes
trainingData['TotalSF'] = trainingData['TotalBsmtSF'] + trainingData['1stFlrSF'] + trainingData['2ndFlrSF']
trainingData['TotalBath'] = trainingData['BsmtFullBath'] + trainingData['BsmtHalfBath'] + trainingData['FullBath'] + trainingData['HalfBath']

testingData['TotalSF'] = testingData['TotalBsmtSF'] + testingData['1stFlrSF'] + testingData['2ndFlrSF']
testingData['TotalBath'] = testingData['BsmtFullBath'] + testingData['BsmtHalfBath'] + testingData['FullBath'] + testingData['HalfBath']
##drop some attributes
#trainingData.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1, inplace=True)
#testingData.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1, inplace=True)


#change some attributes to numerical
trainingData["ExterQual"] = trainingData["ExterQual"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
trainingData["ExterCond"] = trainingData["ExterCond"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
trainingData["HeatingQC"] = trainingData["HeatingQC"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
trainingData["KitchenQual"] = trainingData["KitchenQual"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})

testingData["ExterQual"] = testingData["ExterQual"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
testingData["ExterCond"] = testingData["ExterCond"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
testingData["HeatingQC"] = testingData["HeatingQC"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
testingData["KitchenQual"] = testingData["KitchenQual"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})

#change dtypes
trainingData['MSSubClass'] = trainingData['MSSubClass'].astype(str)


##check for outliers
#col=['GrLivArea', 'TotalSF']
#for c in col:
#    sns.relplot(x=c, y='SalePrice', data=trainingData)
#
#print(trainingData.shape)
trainingData.drop(trainingData[(trainingData['GrLivArea']>4000) & (trainingData['SalePrice']<300000)].index, inplace = True) 

#for c in col:
#    sns.relplot(x=c, y='SalePrice', data=trainingData)
#    
#print(trainingData.shape)




train_Y=trainingData['SalePrice']
train_X=trainingData.drop(['Id', 'SalePrice'], axis=1)

test_X=testingData.drop(['Id'], axis=1)

#print(train_X.dtypes)
#print(train_Y.shape)


#log transform
new_train_Y=np.log(train_Y)




#prosess categorical features
len_train_x=train_X.shape[0]
tmp_all=pd.concat(objs=[train_X, test_X], axis=0, sort=False).reset_index(drop=True)
tmp_all=pd.get_dummies(tmp_all)
tmp_all= tmp_all.astype(float)

new_train_X=tmp_all[:len_train_x]
new_test_X=tmp_all[len_train_x:]



# Splitting
x_train, x_test, y_train, y_test = train_test_split(new_train_X, new_train_Y, test_size=0.1, random_state=1)

GBR = ensemble.GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)


#print(GBR.score(x_test, y_test))

y_pred = GBR.predict(x_test)


rms = sqrt(mean_squared_error(y_test, y_pred))
print(rms)



    
#retraining with whole training data
GBR = ensemble.GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(new_train_X, new_train_Y)


# Getting our SalePrice estimation
result = GBR.predict(new_test_X)
#transform 
result=np.exp(result)

## Saving to CSV
pd.DataFrame({'Id': testingData.Id, 'SalePrice': result}).to_csv('C:/Users/Yotti/Desktop/homePrice/pred0317-01.csv', index =False)    











import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression


#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

featureArray_Sorted=['TotalSF', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBath', 'ExterQual', 'TotRmsAbvGrd', 'YearBuilt', 'KitchenQual', 'YearRemodAdd', 'Foundation', 'Fireplaces', 'HeatingQC', 'BsmtFinSF1', 'Neighborhood', 'SaleType', 'SaleCondition', 'WoodDeckSF', 'OpenPorchSF', 'Exterior2nd', 'Exterior1st', 'MSZoning', 'LotArea', 'LotShape', 'CentralAir', 'HouseStyle', 'Electrical', 'RoofStyle', 'PavedDrive', 'BsmtUnfSF', 'RoofMatl', 'BedroomAbvGr', 'LotConfig', 'ExterCond', 'BldgType', 'KitchenAbvGr', 'EnclosedPorch', 'LandContour', 'Condition1', 'Functional', 'ScreenPorch', 'PoolArea', 'Heating', 'MSSubClass', 'OverallCond', 'Condition2', 'LandSlope', 'MoSold', '3SsnPorch', 'Street', 'YrSold', 'LowQualFinSF', 'MiscVal', 'Utilities', 'BsmtFinSF2']

def changeDtypes(df):
    tmpDF=df.copy()
    
    #fillna
    tmpDF['MSZoning'].fillna(trainingData['MSZoning'].mode()[0], inplace=True)
    tmpDF['Functional'].fillna(trainingData['Functional'].mode()[0], inplace=True)
    tmpDF['Utilities'].fillna(trainingData['Utilities'].mode()[0], inplace=True)
    tmpDF['Utilities'].fillna(trainingData['Utilities'].mode()[0], inplace=True)
    tmpDF['KitchenQual'].fillna(trainingData['KitchenQual'].mode()[0], inplace=True)
    tmpDF['GarageCars'].fillna(0.0, inplace=True)
    tmpDF['GarageArea'].fillna(0.0, inplace=True)
    tmpDF['BsmtFinSF1'].fillna(trainingData['BsmtFinSF1'].mean(), inplace=True)
    tmpDF['BsmtUnfSF'].fillna(trainingData['BsmtUnfSF'].mean(), inplace=True)
    tmpDF['BsmtFinSF2'].fillna(trainingData['BsmtFinSF2'].mean(), inplace=True)
    tmpDF['Exterior2nd'].fillna(trainingData['Exterior2nd'].mode()[0], inplace=True)
    tmpDF['Exterior1st'].fillna(trainingData['Exterior1st'].mode()[0], inplace=True)
    tmpDF['SaleType'].fillna(trainingData['SaleType'].mode()[0], inplace=True)
    
    #standardlize here!!
    tmpDF['LotArea']=(tmpDF['LotArea']-trainingData['LotArea'].mean())/trainingData['LotArea'].std()
    tmpDF['YearBuilt']=(tmpDF['YearBuilt']-trainingData['YearBuilt'].mean())/trainingData['YearBuilt'].std()
    tmpDF['YearRemodAdd']=(tmpDF['YearRemodAdd']-trainingData['YearRemodAdd'].mean())/trainingData['YearRemodAdd'].std()
    tmpDF['BsmtFinSF1']=(tmpDF['BsmtFinSF1']-trainingData['BsmtFinSF1'].mean())/trainingData['BsmtFinSF1'].std()
    tmpDF['BsmtFinSF2']=(tmpDF['BsmtFinSF2']-trainingData['BsmtFinSF2'].mean())/trainingData['BsmtFinSF2'].std()
    tmpDF['BsmtUnfSF']=(tmpDF['BsmtUnfSF']-trainingData['BsmtUnfSF'].mean())/trainingData['BsmtUnfSF'].std()
    tmpDF['YearBuilt']=(tmpDF['YearBuilt']-trainingData['YearBuilt'].mean())/trainingData['YearBuilt'].std()
    tmpDF['LowQualFinSF']=(tmpDF['LowQualFinSF']-trainingData['LowQualFinSF'].mean())/trainingData['LowQualFinSF'].std()
    tmpDF['GrLivArea']=(tmpDF['GrLivArea']-trainingData['GrLivArea'].mean())/trainingData['GrLivArea'].std()
    tmpDF['BedroomAbvGr']=(tmpDF['BedroomAbvGr']-trainingData['BedroomAbvGr'].mean())/trainingData['BedroomAbvGr'].std()
    tmpDF['KitchenAbvGr']=(tmpDF['KitchenAbvGr']-trainingData['KitchenAbvGr'].mean())/trainingData['KitchenAbvGr'].std()
    tmpDF['TotRmsAbvGrd']=(tmpDF['TotRmsAbvGrd']-trainingData['TotRmsAbvGrd'].mean())/trainingData['TotRmsAbvGrd'].std()
    tmpDF['Fireplaces']=(tmpDF['Fireplaces']-trainingData['Fireplaces'].mean())/trainingData['Fireplaces'].std()
    tmpDF['GarageCars']=(tmpDF['GarageCars']-trainingData['GarageCars'].mean())/trainingData['GarageCars'].std()
    tmpDF['GarageArea']=(tmpDF['GarageArea']-trainingData['GarageArea'].mean())/trainingData['GarageArea'].std()
    tmpDF['WoodDeckSF']=(tmpDF['WoodDeckSF']-trainingData['WoodDeckSF'].mean())/trainingData['WoodDeckSF'].std()
    tmpDF['OpenPorchSF']=(tmpDF['OpenPorchSF']-trainingData['OpenPorchSF'].mean())/trainingData['OpenPorchSF'].std()
    tmpDF['EnclosedPorch']=(tmpDF['EnclosedPorch']-trainingData['EnclosedPorch'].mean())/trainingData['EnclosedPorch'].std()
    tmpDF['3SsnPorch']=(tmpDF['3SsnPorch']-trainingData['3SsnPorch'].mean())/trainingData['3SsnPorch'].std()
    tmpDF['ScreenPorch']=(tmpDF['ScreenPorch']-trainingData['ScreenPorch'].mean())/trainingData['ScreenPorch'].std()
    tmpDF['PoolArea']=(tmpDF['PoolArea']-trainingData['PoolArea'].mean())/trainingData['PoolArea'].std()
    tmpDF['MiscVal']=(tmpDF['MiscVal']-trainingData['MiscVal'].mean())/trainingData['MiscVal'].std()
    tmpDF['YrSold']=(tmpDF['YrSold']-trainingData['YrSold'].mean())/trainingData['YrSold'].std()
    tmpDF['TotalSF']=(tmpDF['TotalSF']-trainingData['TotalSF'].mean())/trainingData['TotalSF'].std()
    tmpDF['TotalBath']=(tmpDF['TotalBath']-trainingData['TotalBath'].mean())/trainingData['TotalBath'].std()
    
    tmpDF['ExterQual']=(tmpDF['ExterQual']-trainingData['ExterQual'].mean())/trainingData['ExterQual'].std()
    tmpDF['ExterCond']=(tmpDF['ExterCond']-trainingData['ExterCond'].mean())/trainingData['ExterCond'].std()
    tmpDF['HeatingQC']=(tmpDF['HeatingQC']-trainingData['HeatingQC'].mean())/trainingData['HeatingQC'].std()
    tmpDF['KitchenQual']=(tmpDF['KitchenQual']-trainingData['KitchenQual'].mean())/trainingData['KitchenQual'].std()
    
    #change dtypes
    tmpDF['MSSubClass'] = tmpDF['MSSubClass'].astype(str)
    tmpDF['MSZoning'] = tmpDF['MSZoning'].astype(str)
    tmpDF['Street'] = tmpDF['Street'].astype(str)
    tmpDF['LotShape'] = tmpDF['LotShape'].astype(str)
    tmpDF['LandContour'] = tmpDF['LandContour'].astype(str)
    tmpDF['LandSlope'] = tmpDF['LandSlope'].astype(str)
    tmpDF['Neighborhood'] = tmpDF['Neighborhood'].astype(str)
    tmpDF['Condition1'] = tmpDF['Condition1'].astype(str)
    tmpDF['Condition2'] = tmpDF['Condition2'].astype(str)
    tmpDF['BldgType'] = tmpDF['BldgType'].astype(str)
    tmpDF['HouseStyle'] = tmpDF['HouseStyle'].astype(str)
    tmpDF['RoofStyle'] = tmpDF['RoofStyle'].astype(str)
    tmpDF['RoofMatl'] = tmpDF['RoofMatl'].astype(str)
    tmpDF['Exterior1st'] = tmpDF['Exterior1st'].astype(str)
    tmpDF['Exterior2nd'] = tmpDF['Exterior2nd'].astype(str)
    tmpDF['Foundation'] = tmpDF['Foundation'].astype(str)
    tmpDF['Heating'] = tmpDF['Heating'].astype(str)
    tmpDF['CentralAir'] = tmpDF['CentralAir'].astype(str)
    tmpDF['Electrical'] = tmpDF['Electrical'].astype(str)
    tmpDF['Functional'] = tmpDF['Functional'].astype(str)
    tmpDF['PavedDrive'] = tmpDF['PavedDrive'].astype(str)
    tmpDF['MoSold'] = tmpDF['MoSold'].astype(str)
    tmpDF['SaleType'] = tmpDF['SaleType'].astype(str)
    tmpDF['SaleCondition'] = tmpDF['SaleCondition'].astype(str)
    
    return tmpDF

def DF_k_features(df, k):
    retDF=df.copy()
    
    listOfFeat=featureArray_Sorted[:k]
    retDF=retDF[listOfFeat]
    return retDF


trainingData = pd.read_csv('C:/Users/Yotti/Desktop/homePrice/train.csv')
testingData = pd.read_csv('C:/Users/Yotti/Desktop/homePrice/test.csv')

#print(trainingData.dtypes)
#print(train['SalePrice'].describe())
#sns.distplot(trainingData['SalePrice'])
#print(train.head())

#sns.relplot(x='GrLivArea', y='SalePrice', data=trainingData)
#sns.relplot(x='TotalBsmtSF', y='SalePrice', data=trainingData)

#print(trainingData.isnull().sum().sort_values(ascending=False))


#missing data
total = trainingData.isnull().sum().sort_values(ascending=False)
percent = (trainingData.isnull().sum()/1460).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(25))

##drop some column
#dealing with missing data
trainingData = trainingData.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu','LotFrontage','GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrArea','MasVnrType',], axis=1)
trainingData = trainingData.drop(trainingData.loc[trainingData['Electrical'].isnull()].index)

testingData=testingData.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu','LotFrontage','GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrArea','MasVnrType',], axis=1)

#deleting points
trainingData = trainingData.drop(trainingData[trainingData['Id'] == 1299].index)
trainingData = trainingData.drop(trainingData[trainingData['Id'] == 524].index)


#Add some combine attributes
trainingData['BsmtHalfBath'].fillna(0.0, inplace=True)
trainingData['BsmtFullBath'].fillna(0.0, inplace=True)
trainingData['TotalSF'] = trainingData['TotalBsmtSF'] + trainingData['1stFlrSF'] + trainingData['2ndFlrSF']
trainingData['TotalBath'] = trainingData['BsmtFullBath'] + trainingData['BsmtHalfBath'] + trainingData['FullBath'] + trainingData['HalfBath']

testingData['BsmtHalfBath'].fillna(0.0, inplace=True)
testingData['BsmtFullBath'].fillna(0.0, inplace=True)
testingData['TotalBsmtSF'].fillna(0.0, inplace=True)
testingData['TotalSF'] = testingData['TotalBsmtSF'] + testingData['1stFlrSF'] + testingData['2ndFlrSF']
testingData['TotalBath'] = testingData['BsmtFullBath'] + testingData['BsmtHalfBath'] + testingData['FullBath'] + testingData['HalfBath']


#drop some attributes
trainingData.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1, inplace=True)
testingData.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1, inplace=True)

trainingData['ExterQual'].fillna(trainingData['ExterQual'].mode()[0], inplace=True)
trainingData['ExterCond'].fillna(trainingData['ExterCond'].mode()[0], inplace=True)
trainingData['HeatingQC'].fillna(trainingData['HeatingQC'].mode()[0], inplace=True)
trainingData['KitchenQual'].fillna(trainingData['KitchenQual'].mode()[0], inplace=True)

testingData['ExterQual'].fillna(trainingData['ExterQual'].mode()[0], inplace=True)
testingData['ExterCond'].fillna(trainingData['ExterCond'].mode()[0], inplace=True)
testingData['HeatingQC'].fillna(trainingData['HeatingQC'].mode()[0], inplace=True)
testingData['KitchenQual'].fillna(trainingData['KitchenQual'].mode()[0], inplace=True)
    
trainingData["ExterQual"] = trainingData["ExterQual"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
trainingData["ExterCond"] = trainingData["ExterCond"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
trainingData["HeatingQC"] = trainingData["HeatingQC"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
trainingData["KitchenQual"] = trainingData["KitchenQual"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})

testingData["ExterQual"] = testingData["ExterQual"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
testingData["ExterCond"] = testingData["ExterCond"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
testingData["HeatingQC"] = testingData["HeatingQC"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})
testingData["KitchenQual"] = testingData["KitchenQual"].map({"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1})

#fillna change dtypes and normalize
tmpTrainingData=changeDtypes(trainingData)
tmpTestingData=changeDtypes(testingData)


new_train_Y=tmpTrainingData[['SalePrice']]

#log transform
new_train_Y=np.log(new_train_Y)


#choose features
numOfFeatures=70
new_train_X=DF_k_features(tmpTrainingData, numOfFeatures)
new_test=DF_k_features(tmpTestingData, numOfFeatures)

#prosess categorical features
new_train_X=pd.get_dummies(new_train_X)
new_test=pd.get_dummies(new_test)


# Splitting
x_train, x_test, y_train, y_test = train_test_split(new_train_X, new_train_Y, test_size=0.1, random_state=1)

GBR = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)


#print(GBR.score(x_test, y_test))

y_pred = GBR.predict(x_test)

##transform 
#y_pred=np.exp(y_pred)
#y_test=np.exp(y_test)
#
##evaluate kaggle score
#y_test['log_value'] = np.log(y_test['SalePrice'])
#new_y_test=y_test[['log_value']]
#
#y_pred=np.log(y_pred)

from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))
print(rms)


#lf there are missing column, add them
add_Col=[]
for f in new_train_X.columns:
    if f not in new_test.columns:
        add_Col.append(f)
for col in add_Col:
    new_test[col] = 0
add_Col=[]   
for f in new_test.columns:
    if f not in new_train_X.columns:
        add_Col.append(f)
for col in add_Col:
    new_train_X[col] = 0
    
#retraining with whole training data
GBR = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(new_train_X, new_train_Y)


# Getting our SalePrice estimation
result = GBR.predict(new_test)
#transform 
result=np.exp(result)

## Saving to CSV
pd.DataFrame({'Id': testingData.Id, 'SalePrice': result}).to_csv('C:/Users/Yotti/Desktop/homePrice/2019-0303xxx.csv', index =False)    




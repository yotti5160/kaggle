Kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
Given a set of labeled data of house prices, build a gradient boosting regressor to predict the house prices of another set of data.   

First clean data, drop some columns with too many missing data and fill missing data according to data description.   
Adding some combined attributes, then use correlation of each column and saleprice, to sort the importance of each feature.   
Finally, use cross validation to choose attributes needed, and train with whole training data.   

Score of final model: 0.140

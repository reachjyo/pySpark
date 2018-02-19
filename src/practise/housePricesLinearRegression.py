#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 20:14:06 2018

@author: jyothsnap
"""

import pandas as pd 
from sklearn import linear_model 

housePricesTrDf = pd.read_csv('/users/jyothsnap/.spyder-py3/pythonpractise/src/pythontraining/train.csv')
housePricesTrDf.shape
housePricesTrDf.info()
cols =  ['GrLivArea','ExterQual','YearBuilt','TotalBsmtSF','FullBath']#['LotArea' , 'YearBuilt',  'FullBath' ,'HalfBath', 'YrSold' ]
# y = mx+c
# we have x as the lotArea from train.csv
xTrain = housePricesTrDf[cols]
# we have y as the salesPrice from  train.csv
yTrain = housePricesTrDf['SalePrice']
# we give X and y value , apply linear model to find out coefficient and slope
lnReg = linear_model.LinearRegression()
# here we learn the model from train data and we obtain coefficent and slope
lnReg.fit(xTrain, yTrain)
lnReg.coef_
lnReg.intercept_

# house prices from testData where in house prices are yet to be predicted
housePricesTestDf = pd.read_csv('/users/jyothsnap/.spyder-py3/pythonpractise/src/pythontraining/test.csv')
housePricesTestDf.shape
housePricesTestDf.info()
xTest = housePricesTestDf[cols]

housePricesTestDf['SalePrice'] = lnReg.predict(xTest)
housePricesTestDf.to_csv('/users/jyothsnap/.spyder-py3/pythonpractise/src/pythontraining/predictionsJan04-2.csv')

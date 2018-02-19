#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:14:45 2018

@author: jyothsnap
"""
import pandas as pd
import seaborn as sns

# read data from csv files to dataframe
titanicTrain = pd.read_csv('/users/jyothsnap/Kaggle/titanic/train.csv')
#titanicTrain.count()

#df2.printSchema()

#df2.select('Fare','Sex','Age').describe().show()
#df2.groupby('Embarked').count().show()
#df2.groupby('Sex','Pclass').count().show()
#df2.groupBy('Sex').count().show()

#titanicTrain = df2.toPandas() # will give in simple Pandas Df , the architecture will only run in 1 machine
# for any missing value  in category we consider the highest occurrences 


#categorical columns : statistical EDA
pd.crosstab(index=titanicTrain['Survived'], columns="count")
pd.crosstab(index=titanicTrain['Pclass'], columns="count")
pd.crosstab(index=titanicTrain['Sex'], columns="count")



#categorical columns : visual EDA
sns.countplot(x='Sex',data=titanicTrain)
sns.countplot(x='Pclass',data=titanicTrain)

#continuous features : visual EDA
titanicTrain.info()
titanicTrain.describe()
titanicTrain['Fare'].describe()
titanicTrain['Age'].describe()
titanicTrain[['Age']].describe()

#continuous features: visual EDA
sns.boxplot(x='Fare',data=titanicTrain)
sns.distplot(titanicTrain['Fare']) #is positively skewed
sns.distplot(titanicTrain['Fare'],kde=False)
sns.distplot(titanicTrain['Fare'], bins = 20, rug=False ,kde=False)
sns.distplot(titanicTrain['Fare'], bins=100, kde=False)
sns.distplot(titanicTrain['Age'] ,bins=10)

#bivariate relationships(c-c): statistical EDA 
pd.crosstab(index=titanicTrain['Survived'], columns=titanicTrain['Sex'])
pd.crosstab(index=titanicTrain['Survived'], columns=titanicTrain['Pclass'], margins=True)

sns.factorplot(x="Sex", hue="Survived", data=titanicTrain, kind="count", size=6)
sns.factorplot(x="Pclass", hue="Survived", data=titanicTrain, kind="count", size=6)
sns.factorplot(x="Embarked", hue="Survived", data=titanicTrain, kind="count", size=6)

#bivariate relationships(n-c): visual EDA 
sns.FacetGrid(titanicTrain, row="Survived",size=8).map(sns.kdeplot, "Fare").add_legend()
sns.FacetGrid(titanicTrain, row="Survived",size=10).map(sns.distplot, "Fare").add_legend()
# the attribute row is more of a visual one  , meaning if its row it will display graph per category in row wise
# the attrobute col means the chart or graphs it will display one be after the other  per category
sns.FacetGrid(titanicTrain, row="Survived",size=8).map(sns.boxplot, "Fare").add_legend()

titanicTrain.loc[titanicTrain['Age'].isnull() == True, 'Age'] = titanicTrain['Age'].mean()

sns.FacetGrid(titanicTrain, row="Survived",size=8).map(sns.kdeplot, "Age").add_legend()


# Multi variate 
# FacetGrid creates a grid for rows and columns
sns.FacetGrid(titanicTrain, row="Sex", col="Pclass").map(sns.countplot, "Survived")
sns.FacetGrid(titanicTrain, row="Survived", col="Sex").map(sns.distplot, "Fare")
sns.FacetGrid(titanicTrain, row="Survived", col="Sex").map(sns.distplot, "Age")
sns.FacetGrid(titanicTrain, row="Pclass", col="Sex", hue="Survived").map(sns.kdeplot, "Age").add_legend()








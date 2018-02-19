#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 21:00:49 2018

@author: jyothsnap
"""
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('test app').config('local','test').getOrCreate()

df2 = spark.read.csv('/users/jyothsnap/Kaggle/titanic/train.csv',header=True)
df2.count()

# ---------------------------------------

df3 = df2.select('Sex','Pclass','Survived','Embarked')
df3.show()
df3.printSchema()

from pyspark.ml.feature import StringIndexer
df3 = StringIndexer(inputCol='Sex',outputCol='Gender').fit(df3).transform(df3)
df3.groupby(df3.Embarked,'Embarked').agg({'Embarked':'count'}).show()
df3 = StringIndexer(inputCol='Embarked',outputCol='Embarked_Transformed').fit(df3).transform(df3)
#df3.groupby(df3.Embarked,'Embarked').agg({'Embarked':'count'}).show()
df3.show()
df3.printSchema()

df3 = df3.select(df3.Pclass.cast('double'),df3.SibSp.cast('double'),df3.Survived.cast('double'),df3.Fare.cast('double'))
df3.show()
df3.printSchema()

# Vector assembler

from pyspark.ml.feature import VectorAssembler
df3 = VectorAssembler(inputCols=['Pclass','SibSp','Fare'],outputCol='Features').transform(df3)

df3.show()
#
# 1 choose approach
from pyspark.ml.classification import DecisionTreeClassifier
dt1 = DecisionTreeClassifier(featuresCol='Features',labelCol='Survived',maxDepth=10,impurity='entropy')

# 2 learning process - created a model
model1 = dt1.fit(df3)
model1.depth

# 3 get predictions


df5 = spark.read.csv('E:/kaggle/titanic/test.csv',header=True).select('PassengerId','Pclass','SibSp')
df5
df5 = df5.select(df5.Pclass.cast('double'),df5.SibSp.cast('double'),df5.PassengerId)
df5 = VectorAssembler(inputCols=['Pclass','SibSp'],outputCol='Features').transform(df5)
df20 = model1.transform(df5)


df20.show()

df20.select('PassengerId','prediction').write.csv('c:/test3.csv')

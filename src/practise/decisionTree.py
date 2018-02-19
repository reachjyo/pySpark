#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 06:50:08 2018

@author: jyothsnap
"""

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# read data from csv files to dataframe
df2 = spark.read.csv('/users/jyothsnap/Kaggle/titanic/train.csv',header=True)
df2.count()
df2.describe().show()
# ---------------------------------------

df2.printSchema()
df3 = df2.select('Sex','Pclass','Survived','Embarked')
df3.show()
df3.printSchema()
# -------------------------------------------
from pyspark.ml.feature import StringIndexer, OneHotEncoder
df3 = StringIndexer(inputCol='Embarked',outputCol='Embarked1').fit(df3).transform(df3)
df3.show()

df3 = OneHotEncoder(inputCol='Embarked1',outputCol='Embarked2',dropLast=False).transform(df3)
df3.show(truncate=False)

# --------------------------------------------

df3 = StringIndexer(inputCol='Sex',outputCol='Gender').fit(df3).transform(df3)


#df3.groupBy(df3.Embarked,'Embarked').agg({'Embarked':'count','Embarked1':'sum'}).show()
#df3.show(5)
#
#
#df3.show(5)
#df3.show(10)
#df3.schema
#df3.printSchema()
## --------------------------------------------
#
#
##df4.show()
##df4.printSchema()
#
##fit(si1)
#male   = 0
#female = 1
#
##transform


df3 = df3.select(df3.Pclass.cast('double'),df3.Gender,df3.Survived.cast('double'))
df3.printSchema()

# Vector assembler

from pyspark.ml.feature import VectorAssembler
df3 = VectorAssembler(inputCols=['Pclass'],outputCol='Features').transform(df3)
df3.show()
#
# 1 choose approach
from pyspark.ml.classification import DecisionTreeClassifier
dt1 = DecisionTreeClassifier(featuresCol='Features',labelCol='Survived')

# 2 learning process - created a model
model2 = dt1.fit(df3)
model2.depth
model2.numFeatures

# 3 get predictions

df5 = spark.read.csv('/users/jyothsnap/Kaggle/titanic/test.csv',header=True).select('PassengerId','Pclass','SibSp')
df5 = df5.select(df5.Pclass.cast('double'),df5.SibSp.cast('double'),df5.PassengerId)
df5 = VectorAssembler(inputCols=['Pclass'],outputCol='Features').transform(df5)
df20 = model2.transform(df5)
df20.show()

df20.select('PassengerId','prediction').coalesce(1).write.csv('/users/jyothsnap/Kaggle/titanic/output.csv')

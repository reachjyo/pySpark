#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 18:44:16 2018

@author: jyothsnap
"""

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

df2 = spark.read.csv('/users/jyothsnap/Kaggle/titanic/train.csv',header=True)
df2.count()

df2.printSchema()
df2.describe().show()
# here we are grouping by Embarked Column ,getting the count and then getting the first row for 'S'
embarkedMode=df2.groupby('Embarked').count().sort('count',ascending=False).first()[0]
df2=df2.na.fill({'Embarked':embarkedMode})
df2.show()
# after this line now we see that the count of rows having Embarked 'S'is 646
df3 = df2.select('Sex',df2.Pclass.cast('double'),df2.Survived.cast('double'),'Embarked',df2.Fare.cast('double'))

df3 = VectorAssembler(inputCols=['Pclass','Gender1','Embarked2','Fare1','Age1'],outputCol=)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:34:15 2018

@author: jyothsnap
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:00:31 2018

@author: jyothsnap
"""
#https://github.com/studyml-lab/Python-Basics
# the following 2 lines are needed to initiate spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('test app').config('local','test').getOrCreate()

data=[
      ('John','Smith',47),
      ('James','Smith',27),
      ('Keith', 'Maren', 38)
      ]
header1 = ['fname','lname','age']
df1 = spark.createDataFrame(data,header1)
df1.show(n=2)
df1.first()
df1.count()
df1.printSchema()
df1.show()
df1.schema

# read data from csv files to dataframe
#df2 = spark.read.csv('E:/kaggle/titanic/train_kaggle.csv',header=True)
df2 = spark.read.csv('/users/jyothsnap/Kaggle/titanic/train.csv',header=True)
df2.count()
df2.show()
df2.show(n=30)
df2.cache()
df2.count()
#df1.write.csv('/users/jyothsnap/Kaggle/titanic/sparkWritingTest1')

# ---------------------------------------

#df3 = df2.select('Pclass','SibSp','Survived','Fare')
#df3.show()

# --------------------------------------------

#df4 = df2.filter(df2.Age > 40).select('Pclass','SibSp','Survived')
#df4.show()
#df4.count()
#df4.describe().show()
#df5 = df2.drop('SibSP')
#df5.describe()





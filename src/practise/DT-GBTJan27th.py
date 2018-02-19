#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 19:22:04 2018

@author: jyothsnap
"""

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# read data from csv files to dataframe
df2 = spark.read.csv('/users/jyothsnap/Kaggle/titanic/train.csv',header=True)
df2.count()


embarked_mode = df2.groupby('Embarked').count().sort('count',ascending=False).first()[0]
df2 = df2.na.fill({'Embarked':embarked_mode})
df3 = df2.select('Sex',df2.Pclass.cast('double'),df2.Survived.cast('double'),'Embarked',df2.Fare.cast('double'),df2.Age.cast('double'))

from pyspark.ml.feature import Imputer
df3 = Imputer(inputCols=['Age','Fare'], outputCols=['Age1','Fare1']).fit(df3).transform(df3)
df3.show(3)

from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler
df3 = StringIndexer(inputCol='Embarked',outputCol='Embarked1').fit(df3).transform(df3)
df3.show()
df3 = OneHotEncoder(inputCol='Embarked1',outputCol='Embarked2',dropLast=False).transform(df3)
df3.show()
df3 = StringIndexer(inputCol='Sex',outputCol='Gender').fit(df3).transform(df3)
df3 = OneHotEncoder(inputCol='Gender',outputCol='Gender1',dropLast=False).transform(df3)
df3.show(5)

df3 = VectorAssembler(inputCols=['Pclass','Gender1','Embarked2','Fare1','Age1'],outputCol='Features').transform(df3)
df3.show(truncate=False)
training = df3
# cache the training data frame in Ram
training.cache
training1 = df3
training.show(truncate=False,n=5)

from pyspark.ml.classification import DecisionTreeClassifier
dt1 = DecisionTreeClassifier(featuresCol='Features',labelCol='Survived') 
dtmodel1 = dt1.fit(training)
predictions = dtmodel1.transform(training)
predictions.select('Survived','rawPrediction','probability','prediction').show(n=5,truncate=False)

from pyspark.ml.classification import GBTClassifier
gbt1 = GBTClassifier(featuresCol='Features',labelCol='Survived',maxDepth=6,maxIter=20)
gbtmodel1 = gbt1.fit(training)
predictions = gbtmodel1.transform(training)
PredictionsandLabels = predictions.select('prediction','Survived').rdd

from pyspark.mllib.evaluation import MulticlassMetrics

metric1 = MulticlassMetrics(PredictionsandLabels)
metric1.accuracy
print(metric1.confusionMatrix())

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
gbt2 = GBTClassifier(featuresCol='Features',labelCol='Survived',seed=5000)
paramGrid = ParamGridBuilder().addGrid(gbt2.maxDepth,[4,6,8]).addGrid(gbt2.maxIter,[10,20,30]).build()

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator1 = BinaryClassificationEvaluator(labelCol='Survived',rawPredictionCol='prediction')
cv1 = CrossValidator(estimator=gbt2,
               estimatorParamMaps=paramGrid,
               evaluator=evaluator1,
               numFolds=10)

cvmodel1 = cv1.fit(training)

cvmodel1.bestModel
cvmodel1.avgMetrics
cvmodel1.getEstimatorParamMaps



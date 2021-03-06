#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 11:30:26 2018

@author: jyothsnap
"""

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# read data from csv files to dataframe
df1 = spark.read.csv('/users/jyothsnap/Kaggle/ghouls-goblins-ghosts/train.csv',header=True)
df1.count()
df1.printSchema()
df1.describe().show()
df1.groupby('type').count().show()
df1.groupby('color').count().sort('count',ascending=False).show()
# --------------------------------------------------------------------------
# cast to double
df2 = df1.select(df1.id.cast('double'),df1.bone_length.cast('double'),
                 df1.rotting_flesh.cast('double'),
                 df1.hair_length.cast('double'),df1.has_soul.cast('double'),
                 'color','type')

from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler
df3 = StringIndexer(inputCol='color',outputCol='color1').fit(df2).transform(df2)
df3.show()
df3.printSchema()
df3 = OneHotEncoder(inputCol='color1',outputCol='color2',dropLast=False).transform(df3)
df3.printSchema()
df4 = StringIndexer(inputCol='type',outputCol='type1').fit(df2).transform(df3)
df4.show()
df4.printSchema()

# Vector assembler
df5 = VectorAssembler(inputCols=['id','bone_length','rotting_flesh','hair_length','has_soul','color2'],
                      outputCol='Features').transform(df4)
df5.show(truncate=False)
df5.printSchema()
# --------------------------------------------------------------------------

# data processing complete---
# 6 .Model building
training = df5
training.show(truncate=False,n=5)
from pyspark.ml.classification import DecisionTreeClassifier 
dt1 = DecisionTreeClassifier(featuresCol='Features',labelCol='type1', maxDepth=10)
model22 = dt1.fit(training)
model22.depth
#model22.numFeatures
training2 = model22.transform(training)
PredictionsandLabels = training2.select('prediction','type1').rdd
PredictionsandLabels.collect()
# --------------------------------------------------------------
#Resubstitution approach
from pyspark.mllib.evaluation import  MulticlassMetrics
metrics1 = MulticlassMetrics(PredictionsandLabels)
metrics1.accuracy
# --------------------------------------------------------------------------

# data processing complete---
# 6 .Model building
from pyspark.ml.classification import RandomForestClassifier 
df1 = RandomForestClassifier(featuresCol='Features',labelCol='type1', maxDepth=10)
model3 = dt1.fit(training)
#model22.depth
#model22.numFeatures
training3 = model3.transform(training)
PredictionsandLabels = training3.select('prediction','type1').rdd
PredictionsandLabels.collect()
# --------------------------------------------------------------
#Resubstitution approach
from pyspark.mllib.evaluation import  MulticlassMetrics
metrics2= MulticlassMetrics(PredictionsandLabels)
metrics2.accuracy
# --------------------------------------------------------------------------



# 1 step calculate cv score for 1 model

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator2 = BinaryClassificationEvaluator(labelCol='type1',rawPredictionCol='prediction')
paramGrid = ParamGridBuilder().addGrid(dt1.maxDepth,[2,3,4]).build() #,5,6,7,8,10,15,20]).build()
crossval2 = CrossValidator(estimator=dt1,estimatorParamMaps=paramGrid,
                           evaluator=evaluator2,numFolds=10)
model27 = crossval2.fit(training)
model27.bestModel
model27.avgMetrics
training2 = model27.transform(training)

# CV / Parameter Tuning approach ---------------------------------------------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

paramGrid = ParamGridBuilder().addGrid(dt1.impurity,['entropy','gini']).addGrid(dt1.maxDepth,[2,3,4]).build()


evaluator1 = MulticlassClassificationEvaluator(predictionCol='prediction',
                                               labelCol='type1',
                                               metricName='accuracy')

crossVal4 = CrossValidator(estimator=dt1,estimatorParamMaps=paramGrid,
                          evaluator=evaluator1, numFolds=10)


model23 = crossVal4.fit(df5)
model23.avgMetrics
model23.getEstimatorParamMaps
model23.bestModel
model27.bestModel

# --------------------------------------------------------
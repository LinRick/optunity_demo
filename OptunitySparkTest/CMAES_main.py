#!/usr/bin/python
# -*- coding: UTF-8 -*-

from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from numpy import array

from pyspark import SparkConf, SparkContext
import optunity

sc = SparkContext()
data = sc.textFile("/root/datasetSpark/sample_svm_data.txt")

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])


parsedData = data.map(parsePoint).cache()

#print("data.collect=%s" %data.collect()[0])
#print("parsedData.collect=%s" %parsedData.collect()[0])
#print  #1 0 2.52078447201548 0 0 0 2.004684436494304 2.00034729926846.....
#print  #(1.0,[0.0,2.52078447202,0.0,0.0,0.0,2.00468....


# cross-validation using optunity
@optunity.cross_validated(x=parsedData.collect(), num_folds=2, num_iter=1)
def logistic_l2_accuracy(x_train, x_test, regParam):
    # cache data to get reasonable speeds for methods like LogisticRegression and SVM
    xc = sc.parallelize(x_train).cache()
    # training logistic regression with L2 regularization
    model = LogisticRegressionWithSGD.train(xc, regParam=regParam, regType="l2")
    # making prediction on x_test
    yhat  = sc.parallelize(x_test).map(lambda p: (p.label, model.predict(p.features)))
    # returning accuracy on x_test
    return yhat.filter(lambda (v, p): v == p).count() / float(len(x_test))

def linearsvm_accuracy(x_train, x_test, regParam):
    # cache data to get reasonable speeds for methods like LogisticRegression and SVM
    xc = sc.parallelize(x_train).cache()
    # training logistic regression with L2 regularization    
    model = SVMWithSGD.train(parsedData,regParam=regParam)
    # making prediction on x_test
    yhat  = sc.parallelize(x_test).map(lambda p: (p.label, model.predict(p.features)))
    # returning accuracy on x_test
    return yhat.filter(lambda (v, p): v == p).count() / float(len(x_test))



# using default maximize (particle swarm) with 10 evaluations, regularization between 0 and 10
optimal_pars, _, _ = optunity.maximize(logistic_l2_accuracy, num_evals=10,
                                       regParam=[0, 10], solver_name='cma-es')

print("optimal_pars=%s" %optimal_pars)


# training model with all data for the best parameters
model = LogisticRegressionWithSGD.train(parsedData, regType="l2", **optimal_pars)

# prediction (in real application you would use here newData instead of parsedData)
yhat = parsedData.map(lambda p: (p.label, model.predict(p.features)))





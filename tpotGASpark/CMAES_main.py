#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Created on 2018年9月10日

@author: A40404
'''

from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib
from distributed import Client
import distributed.joblib

from sklearn.externals.joblib import parallel_backend

from dask_spark.core import spark_to_dask

from distributed.utils_test import gen_cluster, loop, cluster

#import pyspark
from pyspark import SparkConf, SparkContext



if __name__ == '__main__':
    
    '''
    digits = load_digits()    
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)
    #TPOT Classification
    tpot = TPOTClassifier(generations=1, population_size=2, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))    
    #tpot.export('tpot_mnist_pipelineTest.py')
    '''
    
    # create the estimator normally
    
    #sc = pyspark.SparkContext('local[4]')
    #sc = pyspark.SparkContext('spark://10.236.1.8:7077')
    #sc = SparkContext('spark://10.236.1.8:7077')    
    sc = SparkContext()
    
    #connect to the cluster
    client = spark_to_dask(sc)
    
    #client = Client()
    #print(client)    
    
    #sc = dask_to_spark(client)
    
    
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        train_size=0.75,
        test_size=0.25,
    )
    
    tpot = TPOTClassifier(
        generations=5,
        population_size=4,
        cv=2,
        n_jobs=-1,
        random_state=1,
        verbosity=2,
        use_dask=True        
    )
            
    with parallel_backend('dask.distributed', scheduler_host='ubuntu7:8786'):
        tpot.fit(X_train, y_train)    
    
    print(tpot.score(X_test, y_test))
    
    #client.close(1)
    #sc.stop()
    

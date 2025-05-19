"""
This script is used to process the bank dataset, such that the bank data can be used in our system as well.

import the dataset of bank and perform the following operations:
    1. get all labels and split it into test and train data
    2. train the model, and test the model
    3. get the prediction of this data
    4. get the accuracy and the fair metrics of the prediction
"""
from os import path
from app import APP_STATIC
import sys
sys.path.insert(1, "../")  

import numpy as np
import pandas as pd
np.random.seed(0)

df = pd.read_csv(path.join(APP_STATIC,'uploads/bank-additional-full.csv'), ';')
# remove attrs
df.drop(['campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], 
    axis = 1, inplace=True)
# remove unknown rows
df = df[(df != 'unknown').all(1)]
# remove divoiced ones
df = df[(df != 'divorced').all(1)]

# replace all unmarried
df.loc[df['marital'] == 'divorced', 'marital'] = 'unmarried'
df.loc[df['marital'] == 'single', 'marital'] = 'unmarried'

# drop some rows that y = 'no'
df = df.drop(df[df['y'] == 'no'].sample(frac=.6).index)    
df = df.drop(df.query("marital == 'married' &  y == 'no'").sample(frac=.6).index)     # drop 
# df = df.drop(df[df['marital'] == 'married'].sample(frac=.2).index)
df = df.drop(df.query("marital == 'unmarried' &  y == 'yes'").sample(frac=.3).index)     # drop 


df = df.sample(n = 5000)
print(df.marital.value_counts())
un = df.query("marital == 'unmarried' &  y == 'no'").shape[0]
uy = df.query("marital == 'unmarried' &  y == 'yes'").shape[0]
print('unmarried, no', un)
print('unmarried, yes', uy)
print(uy/(un+uy))

mn = df.query("marital == 'married' &  y == 'no'").shape[0]
my = df.query("marital == 'married' &  y == 'yes'").shape[0]
print('married, no', mn)
print('married, yes', my)
print(my/(mn+my))

df.to_csv(path.join(APP_STATIC,'uploads/bank_5000.csv'), sep=';', index=False)


# df = pd.read_csv('data/bank_5000.csv', ';')
# print(df.columns)
# for col in df.columns:
#     print(col)
#     print(df[col].unique())



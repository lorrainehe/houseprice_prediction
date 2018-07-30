# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 20:32:31 2018

@author: lulu
"""

import pandas as pd
import numpy as np

path1 = 'F:/pythonlearn/train.csv'
path2 = 'F:/pythonlearn/test.csv'

train = pd.read_csv(path1)
test = pd.read_csv(path2)
train.info()   #发现有空值

train.isnull().sum().sort_values(ascending=False)  #找出有空值的列，一一探索

#7.26
##1. 游泳池相关
train[['PoolQC','PoolArea']]    #发现poolqc为空的行，poolarea等于0.就是说这个房子没有游泳池
train[train['PoolQC'].notnull()][['PoolQC','PoolArea']]
##进一步证明发现确实没有游泳池的就没有面积，也就没有质量，因此没有游泳池也是一个很重要的特征，
##决策：用None填充PoolQC，用0填充PoolArea

##2.其他因素
train[train['MiscFeature'].notnull()]['MiscFeature']
#决策：用none进行填充

##3.巷子  用none进行填充
##4.阳台  用none进行填充
##5.壁炉质量
train[train['FireplaceQu'].notnull()][['Fireplaces','FireplaceQu']]
train['FireplaceQu'].notnull().sum()
##none填充


##6.LotFrontage  用0
train['LotFrontage'].unique().sort_values(ascending=False) 
train[train['LotFrontage'].isnull()][['LotFrontage','LotArea']]

##车库  none
## 车库建造的年份  待定

##地下室 none
train[train['BsmtExposure'].notnull()][['BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual']]

# 墙壁表面
train[train['MasVnrArea'].isnull()][['MasVnrArea','MasVnrType']]
##MasVnrArea  用0    MasVnrType   用none

#电梯 none
train['Electrical']

#7.27



#type(train.isnull())     #Series
train.isnull().sum().sort_values(ascending = False)
test.isnull().sum().sort_values(ascending = False)
##找出缺失值出现的原因
train_fillna = train.fillna('None')
test_fillna = test.fillna('None')
#test_fillna.isnull().sum().sort_values(ascending = False)

#X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]

y_train.isnull().sum()

train.info()

#...............数据格式转换（无法处理字符串类型）................#
#对于有监督的模型来说，这一步是必须的
train.dropna(axis=0, subset=['SalePrice'], inplace=True)

#把类目属性转变成矩阵形式
train_dummie = pd.get_dummies(train_fillna)
train_dummie
test_val = pd.get_dummies(test_fillna)
test_val


test_val,X_dummie = test_val.align(train_dummie, join='outer', axis=1)

train_dummie['SalePrice']
X_dummie['SalePrice']
X_dummie.head(5)
test_val.head(5)
###########注意注意注意隐患隐患隐患！！！！！！！############
#train_dummie.fillna(99999,inplace = True)
#test_val.fillna(99999,inplace = True)

processed_train = X_dummie.drop(['Id','SalePrice'],axis = 1)
processed_test = test_val.drop(['Id','SalePrice'],axis = 1)

processed_train
processed_train.head(20)

processed_train.isnull()
processed_test.head()


#...............随机森林搭建......................#
from sklearn.ensemble import RandomForestRegressor

###########注意注意注意隐患隐患隐患！！！参数设置！！！！#########
RR = RandomForestRegressor(n_estimators = 200,max_features = 50)
RR.fit(processed_train,y_train)

processed_train.isnull().sum().sort_values(ascending=False).head(1000)
predictions = RR.predict(processed_test)
predictions.size

Id_test = test_val['Id']

from pandas import Series,DataFrame
submission = DataFrame({'Id':Id_test.as_matrix(),'SalePrice':predictions.astype(np.int32)})
submission.to_csv('F:/pythonlearn/firstresult.csv', index=False)

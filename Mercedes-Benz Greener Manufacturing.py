#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing important libraries.

import pandas as pd
import numpy as np


# In[2]:


# calling data sets.

mbtrain_df = pd.read_csv(r"C:\Users\Shams\Downloads\train.csv")
mbtest_df = pd.read_csv(r"C:\Users\Shams\Downloads\test.csv")


# In[3]:


mbtrain_df.head()


# In[4]:


print(f' {mbtrain_df.shape[0]} train observations \n {mbtrain_df.shape[1]} train columns \n {mbtest_df.shape[0]} test observations \n {mbtest_df.shape[1]} test columns')


# In[5]:


mbtrain_df.columns[mbtrain_df.isnull().any()].tolist()


# In[6]:


# We can also count the null with:

mbtrain_df.isnull().sum().sum()


# In[7]:


#  Variance of each column. (standard deviation ==0 will also do : mbtrain_df.std(axis=0))
mbtrain_df.var(axis=0)


# In[8]:


type(mbtrain_df.var(axis=0))


# In[9]:


to_drop = [ind for ind,val in enumerate(mbtrain_df.var(axis=0)) if val ==0]


# In[10]:


mbtrain_df.iloc[:,to_drop]


# In[11]:


mbtrain_df.iloc[:,to_drop].nunique()


# In[12]:


mbtrain_df.iloc[:,to_drop].sum()


# In[13]:


## Checking for null in test
mbtest_df.columns[mbtest_df.isnull().any()].tolist()


# In[14]:


mbtest_df.isnull().sum().sum()


# In[15]:


mbtest_df.var(axis=0)


# In[16]:


to_drop = [ind for ind,val in enumerate(mbtest_df.var(axis=0)) if val ==0]


# In[17]:


to_drop 


# In[18]:


mbtest_df.iloc[:,to_drop].nunique()


# In[19]:


mbtest_df.iloc[:,to_drop].sum()


# In[20]:


mbtrain_df.shape


# In[21]:


mbtrain_df.drop(['X249', 'X287', 'X288', 'X99', 'X227', 'X260', 'X281', 'X282', 'X289', 'X322', 'X339'], axis=1, inplace=True)
mbtest_df.drop(['X249', 'X287', 'X288', 'X99', 'X227', 'X260', 'X281', 'X282', 'X289', 'X322', 'X339'], axis=1, inplace=True)


# In[22]:


print(f' After dropping zero-variance proedictors \n {mbtrain_df.shape[0]} train observations \n {mbtrain_df.shape[1]} train columns \n {mbtest_df.shape[0]} test observations \n {mbtest_df.shape[1]} test columns')


# In[23]:


# Apply label encoder.

# For testing set
train2encode = mbtrain_df.select_dtypes(include=['object']).columns
# For testing set
test2encode = mbtest_df.select_dtypes(include=['object']).columns


# In[24]:


print(f'Number of unique values in columns to be encoded: \n\nTraining set\n{mbtrain_df[train2encode].nunique()}\n\nTesting set\n{mbtest_df[test2encode].nunique()}' )


# In[25]:


from sklearn.preprocessing import LabelEncoder
labelenc = LabelEncoder()


# In[26]:


# For training set
for encoded in enumerate(train2encode):
    mbtrain_df[encoded[1]]=labelenc.fit_transform(mbtrain_df[encoded[1]].values)


# In[27]:


mbtrain_df.head()


# In[28]:


# For testing set
for encoded in enumerate(test2encode):
    mbtest_df[encoded[1]]=labelenc.fit_transform(mbtest_df[encoded[1]].values)


# In[29]:


mbtest_df.head()


# In[30]:


# Dropping ID
mbtrain_df.drop('ID',axis=1,inplace=True)
mbtest_df.drop('ID',axis=1,inplace=True)


# In[31]:


# duplicates rows in training set?
print(mbtrain_df.duplicated().any())


# In[32]:


# duplicates rows in testing set?
print(mbtest_df.duplicated().any())


# In[33]:


# For training set
mbtrain_df.loc[:,mbtrain_df.columns.str.contains("^X")].describe().T


# In[34]:


# For testing set
mbtest_df.loc[:,mbtest_df.columns.str.contains("^X")].describe().T


# In[35]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer


# In[36]:


mbtrain_df.shape


# In[37]:


mbtrain_df.head()


# In[38]:


# Normalizing and scaling training set (without the target)
train_scaler=Normalizer().fit(mbtrain_df.drop('y',1))
norm_train_df = train_scaler.transform(mbtrain_df.drop('y',1))
norm_train_df.shape


# In[39]:


# Normalizing and scaling testing set
test_scaler=Normalizer().fit(mbtest_df)
norm_test_df = test_scaler.transform(mbtest_df)
norm_test_df.shape


# In[40]:


pca =PCA()
pca.fit(norm_train_df)


# In[41]:


f =np.cumsum(pca.explained_variance_ratio_)
plt.plot(f)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# In[42]:


plt.plot(f)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.grid(True)
plt.xlim(0,50)
plt.ylim(.9,1)
plt.show()


# In[43]:


print(f'The mean explained_variance_ratio is {np.mean(pca.explained_variance_ratio_)}')


# In[44]:


print(f'The number of features whose mean are greater than the mean explained_variance_ratio is {np.sum([pca.explained_variance_ratio_ > np.mean(pca.explained_variance_ratio_)])}')


# In[45]:


pca = PCA(n_components=0.97, whiten=True)
norm_features = pca.fit_transform(norm_train_df)


# In[46]:


print(f'The midpoint method retains {norm_features.shape[1]} candidate features')


# In[47]:


# Predict your test_df values using XGBoost.
# Import necessary modules
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# In[48]:


print(xgb.__version__)


# In[49]:


# Splitting into training and validations  sets
X_train, X_val, y_train, y_val = train_test_split(mbtrain_df.iloc[:,1:], mbtrain_df['y'].values, test_size=0.25, random_state=4321)


# In[50]:


X_train.shape, X_val.shape, y_train.shape, y_val.shape


# In[51]:


train_xgb_reg = xgb.XGBRegressor( objective = 'reg:squarederror', colsample_bytree = 0.1,  learning_rate = 0.2, max_depth = 7, alpha = 10)


# In[52]:


train_xgb_reg.fit(X_train,y_train)
train_valid = train_xgb_reg.predict(X_val)


# In[53]:


training_rmse = np.sqrt(mean_squared_error(y_val, train_valid))
print(f'Training RMSE: {training_rmse}')


# In[54]:


train_xgb_reg.score(X_train,y_train)


# In[55]:


testing_preds = train_xgb_reg.predict(mbtest_df)


# In[56]:


testing_rmse = np.sqrt(mean_squared_error(mbtrain_df['y'], testing_preds))
print(f'Testing RMSE: {testing_rmse}')


# Conclusion
# We can see that we got a larger RMSE for the testing. This suggest the model did not do well on the testing set.
# A better way would be to use the Cross Validation method of XGBoost to help identify the features that will yield a better training RMSE.
# We then use the model with a better RMSE to predict.

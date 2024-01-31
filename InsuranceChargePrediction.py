#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics 


# In[25]:


data = pd.read_csv('/Users/macbook/Desktop/my/myprojectsds/insurance.csv')


# # Top five rows of dataset

# In[26]:


data.head()


# # Last five rows of dataset

# In[82]:


data.tail()


# # Number of rows and columns

# In[28]:


number_of_rows=data.shape[0]
number_of_columns=data.shape[1]
print('Number of rows:',number_of_rows)
print('Number of columns:',number_of_columns)


# # Information about our dataset

# In[29]:


data.info()


# # Checking Null values

# In[30]:


data.isnull().sum()


# # Statistics about the dataset

# In[31]:


data.describe()


# # Converting categorical values to numerical values

# In[32]:


data['sex'].unique()


# In[34]:


data['sex']=data['sex'].map({'male':0,'female':1})
data['smoker']=data['smoker'].map({'no':0,'yes':1})
data.head()


# In[35]:


data['region'].unique()


# In[37]:


data['region']=data['region'].map({'southwest':1,
                    'southeast':2,
                    'northwest':3,
                    'northeast':4})


# In[38]:


data


# # Train Test Split

# In[39]:


data.columns


# In[40]:


X = data.drop(['charges'],axis=1)


# In[41]:


X


# In[42]:


y = data['charges']


# In[43]:


y


# In[49]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[50]:


X_train


# In[52]:


X_test


# # Model Training

# In[54]:


lr = LinearRegression()
lr.fit(X_train,y_train)
svm = SVR()
svm.fit(X_train,y_train)
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)


# # Prediction on test data

# In[56]:


y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rfr.predict(X_test)
y_pred4 = gbr.predict(X_test)

df1 = pd.DataFrame({'Real':y_test,
                   'LinearRegression':y_pred1,
                   'SVM':y_pred2,
                   'RandomForest':y_pred3,
                   'GradientBoosting':y_pred4}
                  )


# In[57]:


df1


# # Visual comparison

# In[86]:


plt.subplot(221)
plt.plot(df1['Real'].iloc[0:11],label='real')
plt.plot(df1['LinearRegression'].iloc[0:11],label='LR')
plt.legend()
plt.subplot(222)
plt.plot(df1['Real'].iloc[0:11],label='real')
plt.plot(df1['SVM'].iloc[0:11],label='SVM')
plt.legend()
plt.subplot(223)
plt.plot(df1['Real'].iloc[0:11],label='real')
plt.plot(df1['RandomForest'].iloc[0:11],label='RF')
plt.legend()
plt.subplot(224)
plt.plot(df1['Real'].iloc[0:11],label='real')
plt.plot(df1['GradientBoosting'].iloc[0:11],label='GB')

plt.legend()


# # Evaluating the Algorithm

# ## R squared (higher is better)

# In[68]:


score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)


# In[73]:


print(score1)
print(score2)
print(score3)
print(score4)


# ## MAE (lower is better)

# In[75]:


score_LR = metrics.mean_absolute_error(y_test,y_pred1)
score_SVM = metrics.mean_absolute_error(y_test,y_pred2)
score_RF = metrics.mean_absolute_error(y_test,y_pred3)
score_GB = metrics.mean_absolute_error(y_test,y_pred4)


# In[77]:


print(score_LR)
print(score_SVM)
print(score_RF)
print(score_GB)


# # Predict  charges for new customer

# In[89]:


data = {'age' : 20,
       'sex' : 1,
       'bmi':39.1,
       'children':0,
       'smoker' : 0,
       'region' : 1}


# In[91]:


new_df = pd.DataFrame(data,index=[0])
new_df


# In[92]:


gbr.predict(new_df)


# # Saving the model

# In[94]:


gbr = GradientBoostingRegressor()
gbr.fit(X,y)


# In[95]:


import joblib


# In[96]:


joblib.dump(gbr,'model_gbr')


# In[97]:


model = joblib.load('model_gbr')


# In[107]:


new=model.predict(new_df)
new


# In[109]:


print('Predicted charge for the new customer is :', round(new[0],5))


# In[ ]:





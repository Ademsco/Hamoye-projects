#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[26]:


df= pd.read_csv('energydata_complete.csv')


# In[27]:


df


# In[28]:


slr_df=df[['T2','T6']].sample(15, random_state=2)


# In[29]:


sns.regplot(x="T2", y="T6",
            data=slr_df)


# In[47]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[48]:


from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()


# In[49]:


model=linear_model.fit(df[['T2']], df.T6)


# In[51]:


from sklearn.model_selection import train_test_split


# model

# In[55]:


predicted_values = linear_model.predict(df[['T2']])


# In[56]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(df[['T2']], predicted_values)
round(mae, 3)


# In[59]:


from sklearn.metrics import r2_score
r2_score = r2_score(df[['T2']], predicted_values)
round(r2_score, 3)


# In[61]:


import numpy as np
rss = np.sum(np.square(df['T2'] - predicted_values))
round(rss, 3)


# In[62]:


from sklearn.metrics import  mean_squared_error
rmse = np.sqrt(mean_squared_error(df['T2'], predicted_values))
round(rmse, 3)


# In[63]:


predicted_values.coef_


# In[ ]:





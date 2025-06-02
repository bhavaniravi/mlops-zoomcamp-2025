#!/usr/bin/env python
# coding: utf-8

# In[35]:


# In[4]:


import pickle
import pandas as pd
import sys

try:
    month = int(sys.argv[1])
    if month < 1 and month > 12:
        raise ValueError
except: 
    print ("expected month argument python script.py <month> <year>")
    sys.exit()


try:
    year = int(sys.argv[2])
    if year > 2025:
        raise ValueError
except: 
    print ("expected year argument python script.py <month> <year>")
    sys.exit()


# In[5]:




# In[36]:

import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
model_path = current_file_path + '/model.bin'

with open(model_path, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[37]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    print (filename)
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


# In[38]:


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


# In[39]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# ### What's the standard deviation of the predicted duration for this dataset?

# In[40]:


print ("std deviation=", y_pred.std())
print ("mean = ", y_pred.mean())


# In[41]:


# ### 2. Preparing the output


# In[43]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[44]:


df_result = df[['ride_id']].assign(y_pred=y_pred)


# In[45]:


df_result


# In[46]:


output_file = 'week4_result_file.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[ ]:





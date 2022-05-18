#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels as sm
import time


# In[13]:


df_crop_UK = pd.read_csv("FAOSTAT UK crop data.csv")


# In[14]:


df_crop_UK


# In[15]:


print(df_crop_UK.columns)


# In[16]:


df_crop_UK = df_crop_UK.drop(['Domain Code', 'Domain', 'Area Code (FAO)', 'Area', 'Element Code',
       'Element', 'Item Code (FAO)','Year Code','Unit',
       'Flag', 'Flag Description'], axis = 1)


# In[65]:


df_crop_UK.rename(columns = {'Value':'Tonnes_Yield'}, inplace = True)


# In[118]:


df_crop_UK.rename(columns = {'Year':'year'}, inplace = True)


# In[114]:


df_crop_UK


# In[113]:


df_crop_UK = df_crop_UK.dropna()


# In[95]:


df_weather_UK = pd.read_csv("MET Office Weather Data.csv")


# In[96]:


df_weather_UK


# In[97]:


df_weather_UK.info()


# In[98]:


print(df_weather_UK.columns)


# In[99]:


df_weather_UK= df_weather_UK.drop([ 'sun','af', 'station','month'], axis = 1)


# In[104]:


df_weather_UK


# In[101]:


df_weather_UK = df_weather_UK.dropna()


# In[102]:


for col in df_weather_UK:
    print(df_weather_UK["year"].unique())


# In[103]:


df_weather_UK.year = df_weather_UK.year.astype(int)


# In[105]:


df_weather_UK['mean_temp'] = df_weather_UK[['tmax', 'tmin']].mean(axis=1)


# In[106]:


for col in df_weather_UK:
    print(df_weather_UK["year"].unique())


# In[109]:


df_grp_rain_UK = df_weather_UK
#df_grp_rain_UK = df_weather_UK.groupby(['year'])['rain'].sum().reset_index()
#df_grp_rain_UK = df_weather_UK.groupby(['year'])["rain"].apply(lambda x : x.astype(int).sum())
df_grp_rain_UK = df_grp_rain_UK.groupby(['year'])['rain'].sum().reset_index()


# In[110]:


df_grp_rain_UK


# In[111]:


df_grp_temp_UK = df_weather_UK
df_grp_temp_UK = df_grp_temp_UK.groupby(['year'])['mean_temp'].mean().reset_index()
df_grp_temp_UK


# In[112]:


df_grp_temp_UK


# In[115]:


df_rain_temp_UK=df_grp_temp_UK.merge(df_grp_rain_UK, on='year', how='left') 


# In[117]:


df_rain_temp_UK


# In[119]:


df_ML_UK = df_crop_UK.merge(df_rain_temp_UK, on='year', how='left')


# In[122]:


df_ML_UK


# In[121]:


df_ML_UK.rename(columns = {'Item':'Crop_Type'}, inplace = True)


# In[ ]:


############spain data############


# In[2]:


df_crop_Spain = pd.read_csv("FAOSTAT Spain Crop data.csv")


# In[3]:


df_crop_Spain


# In[5]:


df_crop_Spain = df_crop_Spain.drop(['Domain Code', 'Domain', 'Area Code (FAO)', 'Area', 'Element Code',
       'Element', 'Item Code (FAO)','Year Code','Unit',
       'Flag', 'Flag Description'], axis = 1)


# In[6]:


df_crop_Spain.rename(columns = {'Value':'Tonnes_Yield'}, inplace = True)
df_crop_Spain.rename(columns = {'Year':'year'}, inplace = True)


# In[7]:


df_crop_Spain


# In[8]:


df_crop_Spain = df_crop_Spain.dropna()


# In[11]:


df_weather_Spain = pd.read_csv("Spain weather data.csv")


# In[16]:


df_weather_Spain


# In[13]:


df_weather_Spain.rename(columns = {'Annual Mean':'mean_temp'}, inplace = True)
df_weather_Spain.rename(columns = {'Category':'year'}, inplace = True)


# In[15]:


df_weather_Spain = df_weather_Spain.drop(['5-yr smooth'], axis = 1)


# In[22]:


df_rain_Spain = pd.read_csv("Spain rain data.csv")


# In[23]:


df_rain_Spain 


# In[24]:


df_rain_Spain.rename(columns = {'Annual Mean':'mean_rain'}, inplace = True)
df_rain_Spain.rename(columns = {'Category':'year'}, inplace = True)
df_rain_Spain = df_rain_Spain.drop(['5-yr smooth'], axis = 1)


# In[25]:


df_rain_Spain


# In[26]:


df_rain_temp_Spain=df_rain_Spain.merge(df_weather_Spain, on='year', how='left') 


# In[27]:


df_rain_temp_Spain


# In[31]:


df_ML_Spain =df_crop_Spain.merge(df_rain_temp_Spain, on='year', how='left') 


# In[37]:


df_ML_Spain


# In[33]:


df_ML_Spain.info()


# In[34]:


df_ML_Spain.rename(columns = {'Item':'Crop_Type'}, inplace = True)


# In[36]:


df_ML_Spain = df_ML_Spain.dropna()


# In[ ]:





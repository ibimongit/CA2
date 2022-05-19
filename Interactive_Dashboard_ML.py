#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

data = pd.read_csv('Xdata.csv')
    
X = data[['Yield_000_Tonnes', 'mean_temp', 'total_rain', 'Beans_and_peas',
       'Fodder_beet', 'Kale_and_field_cabbage', 'Oilseed_rape', 'Potatoes',
       'Spring_barley', 'Spring_oats', 'Spring_wheat', 'Sugar_beet',
       'Total_barley', 'Total_oats', 'Total_wheat',
       'Total_wheat,_oats_and_barley', 'Turnips', 'Winter_barley',
       'Winter_oats', 'Winter_wheat']]
y = data['Crop_Cat_Id']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

model = LogisticRegression()
model.fit(X_train, y_train)

explainer = ClassifierExplainer(model, X_test, y_test)
ExplainerDashboard(explainer,mode='inline').run()


# In[ ]:





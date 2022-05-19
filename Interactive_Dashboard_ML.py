#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

data = pd.read_csv('Xdata.csv')

feature_descriptions = {
    'Yield_000_Tonnes':'total yield', 
    'mean_temp':'mean temperature', 
    'total_rain':'total rain', 
    'Beans_and_peas':'beans and peas',
    'Fodder_beet':'Fodder beet', 
    'Kale_and_field_cabbage':'Kale and cabbage', 
    'Oilseed_rape':'Oilseed rape', 
    'Potatoes':'Potatoes',
    'Spring_barley':'Spring barley', 
    'Spring_oats':'Spring oats', 
    'Spring_wheat':'Spring wheat', 
    'Sugar_beet':'Sugar beet',
    'Total_barley':'Total barley', 
    'Total_oats':'Total oats', 
    'Total_wheat':'Total wheat',
    'Turnips':'Turnips', 
    'Winter_barley':'Winter barley',
    'Winter_oats':'Winter oats', 
    'Winter_wheat':'Winter wheat',
    'Crop_Cat_Id':'Crop ID'
}
    
X = data[['Yield_000_Tonnes', 'mean_temp', 'total_rain', 'Beans_and_peas',
       'Fodder_beet', 'Kale_and_field_cabbage', 'Oilseed_rape', 'Potatoes',
       'Spring_barley', 'Spring_oats', 'Spring_wheat', 'Sugar_beet',
       'Total_barley', 'Total_oats', 'Total_wheat',
       'Turnips', 'Winter_barley',
       'Winter_oats', 'Winter_wheat']]
y = data['Crop_Cat_Id']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

model = LogisticRegression()
model.fit(X_train, y_train)

explainer = ClassifierExplainer(model, X_test, y_test,descriptions=feature_descriptions)
ExplainerDashboard(explainer,mode='inline').run()


# In[ ]:





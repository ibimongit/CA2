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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# Importing classifiers


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder, MinMaxScaler
#importing for preprocessing methods used for scaler as well as training


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
#importing cross validation methods



from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')


# <b>Data Prerp for Ireland</b>

# In[2]:


df_crop = pd.read_csv("Crop Yield.csv")


# In[3]:


df_crop


# In[4]:


df_crop.info()


# In[5]:


#remove area under crop
for col in df_crop:
    print(df_crop["Statistic"].unique())


# In[6]:


####remove 000 hecatres and remove tonnes
for col in df_crop:
    print(df_crop["UNIT"].unique())


# In[7]:


df_crop["Type of Crop"].unique()


# In[8]:


df_crop.rename(columns = {'Type of Crop':'Type_of_Crop'}, inplace = True)


# In[9]:


df_crop = df_crop[~df_crop.Type_of_Crop.str.contains("Total wheat, oats and barley")]


# In[10]:


df_crop['Type_of_Crop'].nunique()


# In[11]:


df_crop = df_crop[~df_crop.Statistic.str.contains("Area under Crops")]
df_crop = df_crop[~df_crop.Statistic.str.contains("Crop Yield per Hectare")]
df_crop = df_crop[~df_crop.Statistic.str.contains("Unit")]


# In[12]:


df_crop.rename(columns = {'VALUE':'Yield_000_Tonnes'}, inplace = True)


# In[13]:


df_crop = df_crop.dropna()


# In[14]:


df_temp = pd.read_csv("Temperature Data.csv")


# In[15]:


df_temp


# In[16]:


for col in df_temp:
    print(df_temp["Statistic"].unique())


# In[17]:


df_temp = df_temp[~df_temp.Statistic.str.contains("Average Maximum Temperature")]
df_temp = df_temp[~df_temp.Statistic.str.contains("Average Minimum Temperature")]
df_temp = df_temp[~df_temp.Statistic.str.contains("Highest Temperature")]
df_temp = df_temp[~df_temp.Statistic.str.contains("Lowest Temperature")]


# In[18]:


df_temp = df_temp.dropna()


# In[19]:


df_temp['Month'] = df_temp['Month'].str.replace('M','/')
df_temp['Month'] = pd.to_datetime(df_temp['Month'])
df_temp['Month'] = df_temp['Month'].dt.strftime('%Y')


# In[20]:


df_grp_temp = df_temp.groupby(['Month'])['VALUE'].mean().reset_index()


# In[21]:


df_grp_temp


# In[22]:


df_grp_temp.rename(columns = {'Month':'Year','VALUE':'mean_temp'}, inplace = True)


# In[23]:


df_grp_temp['Year'] = df_grp_temp['Year'].astype(np.int64)


# In[24]:


##merging
#df_crop_temp = pd.merge(df_crop,df_grp_temp,by="Year",all=TRUE)
df_crop_temp=df_crop.merge(df_grp_temp, on='Year', how='left') 


# In[25]:


df_crop_temp


# In[26]:


df_crop_temp.info()


# In[27]:


df_rain = pd.read_csv("Rain Data.csv")


# In[28]:


df_rain


# In[29]:


for col in df_rain:
    print(df_rain["Statistic"].unique())


# In[30]:


df_rain = df_rain[~df_rain.Statistic.str.contains("Most Rainfall in a Day")]
df_rain = df_rain[~df_rain.Statistic.str.contains("Raindays (0.2mm or More)")]


# In[31]:


df_rain['Month'] = df_rain['Month'].str.replace('M','/')
df_rain['Month'] = pd.to_datetime(df_rain['Month'])
df_rain['Month'] = df_rain['Month'].dt.strftime('%Y')


# In[32]:


df_grp_rain = df_rain.groupby(['Month'])['VALUE'].sum().reset_index()


# In[33]:


df_grp_rain.rename(columns = {'Month':'Year','VALUE':'total_rain'}, inplace = True)##rename to tatal rain


# In[34]:


df_grp_rain['Year'] = df_grp_rain['Year'].astype(np.int64)


# In[35]:


df_grp_rain


# In[36]:


df_crop_temp_rain=df_crop_temp.merge(df_grp_rain, on='Year', how='left')


# In[37]:


df_crop_temp_rain


# In[38]:


df_crop_temp_rain.describe()


# In[39]:


df_ML1 = df_crop_temp_rain.copy()


# <b>Data prerp UK</b>

# In[40]:


df_crop_UK = pd.read_csv("FAOSTAT UK crop data.csv")
df_crop_UK


# In[41]:


print(df_crop_UK.columns)


# In[42]:


df_crop_UK = df_crop_UK.drop(['Domain Code', 'Domain', 'Area Code (FAO)', 'Area', 'Element Code',
       'Element', 'Item Code (FAO)','Year Code','Unit',
       'Flag', 'Flag Description'], axis = 1)


# In[43]:


df_crop_UK.rename(columns = {'Value':'Tonnes_Yield'}, inplace = True)
df_crop_UK.rename(columns = {'Year':'year'}, inplace = True)
df_crop_UK


# In[44]:


df_crop_UK = df_crop_UK.dropna()


# In[45]:


df_weather_UK = pd.read_csv("MET Office Weather Data.csv")
df_weather_UK


# In[46]:


df_weather_UK.info()


# In[47]:


print(df_weather_UK.columns)


# In[48]:


df_weather_UK= df_weather_UK.drop([ 'sun','af', 'station','month'], axis = 1)
df_weather_UK


# In[49]:


df_weather_UK = df_weather_UK.dropna()


# In[50]:


for col in df_weather_UK:
    print(df_weather_UK["year"].unique())


# In[51]:


df_weather_UK.year = df_weather_UK.year.astype(int)


# In[52]:


df_weather_UK['mean_temp'] = df_weather_UK[['tmax', 'tmin']].mean(axis=1)


# In[53]:


df_grp_rain_UK = df_weather_UK
df_grp_rain_UK = df_grp_rain_UK.groupby(['year'])['rain'].sum().reset_index()
df_grp_rain_UK


# In[54]:


df_grp_temp_UK = df_weather_UK
df_grp_temp_UK = df_grp_temp_UK.groupby(['year'])['mean_temp'].mean().reset_index()
df_grp_temp_UK


# In[55]:


df_rain_temp_UK=df_grp_temp_UK.merge(df_grp_rain_UK, on='year', how='left') 
df_rain_temp_UK


# In[56]:


df_ML_UK = df_crop_UK.merge(df_rain_temp_UK, on='year', how='left')
df_ML_UK


# In[57]:


df_ML_UK.rename(columns = {'Item':'Crop_Type'}, inplace = True)


# <b>Data prerp Spain</b>

# In[58]:


df_crop_Spain = pd.read_csv("FAOSTAT Spain Crop data.csv")
df_crop_Spain


# In[59]:


df_crop_Spain = df_crop_Spain.drop(['Domain Code', 'Domain', 'Area Code (FAO)', 'Area', 'Element Code',
       'Element', 'Item Code (FAO)','Year Code','Unit',
       'Flag', 'Flag Description'], axis = 1)
df_crop_Spain = df_crop_Spain.dropna()


# In[60]:


df_crop_Spain.rename(columns = {'Value':'Tonnes_Yield'}, inplace = True)
df_crop_Spain.rename(columns = {'Year':'year'}, inplace = True)


# In[61]:


df_weather_Spain = pd.read_csv("Spain weather data.csv")
df_weather_Spain


# In[62]:


df_weather_Spain.rename(columns = {'Annual Mean':'mean_temp'}, inplace = True)
df_weather_Spain.rename(columns = {'Category':'year'}, inplace = True)
df_weather_Spain = df_weather_Spain.drop(['5-yr smooth'], axis = 1)


# In[63]:


df_rain_Spain = pd.read_csv("Spain rain data.csv")
df_rain_Spain


# In[64]:


df_rain_Spain.rename(columns = {'Annual Mean':'mean_rain'}, inplace = True)
df_rain_Spain.rename(columns = {'Category':'year'}, inplace = True)
df_rain_Spain = df_rain_Spain.drop(['5-yr smooth'], axis = 1)


# In[65]:


df_rain_temp_Spain=df_rain_Spain.merge(df_weather_Spain, on='year', how='left') 


# In[66]:


df_ML_Spain =df_crop_Spain.merge(df_rain_temp_Spain, on='year', how='left') 
df_ML_Spain


# In[67]:


df_ML_Spain.info()


# In[68]:


df_ML_Spain.rename(columns = {'Item':'Crop_Type'}, inplace = True)


# In[152]:


df_ML_Spain['Crop_Type'].nunique()


# In[69]:


for col in df_ML_Spain:
    print(df_ML_Spain["year"].unique())


# <b>Statistics</b>

# In[70]:


##Ireland decribe
df_ML1.describe()


# In[71]:


df_ML_Spain.describe()


# Shapiro-Wilk Tests

# In[72]:


#Shapiro-Wilk Test for Ireland Yield
from scipy.stats import shapiro
data = df_ML1['Yield_000_Tonnes']
stat, p = shapiro(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')


# In[73]:


#Shapiro-Wilk Test for Ireland temp
data = df_ML1['mean_temp']
stat, p = shapiro(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')


# In[74]:


#Shapiro-Wilk Test for Spain Yield
data = df_ML_Spain['Tonnes_Yield']
stat, p = shapiro(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')


# In[75]:


#Shapiro-Wilk Test for Spain temp
data = df_ML_Spain['mean_temp']
stat, p = shapiro(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')


# Spearman’s Rank Correlation

# In[76]:


#Spearman’s Rank Correlation for Ireland Yield vs Rain
from scipy.stats import spearmanr
data1 = df_ML1['Yield_000_Tonnes']
data2 = df_ML1['total_rain']
stat, p = spearmanr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')


# In[77]:


#Spearman’s Rank Correlation for Ireland Yield vs temp
from scipy.stats import spearmanr
data1 = df_ML1['Yield_000_Tonnes']
data2 = df_ML1['mean_temp']
stat, p = spearmanr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')


# In[78]:


#Spearman’s Rank Correlation for Spain Yield vs Rain
from scipy.stats import spearmanr
data1 = df_ML_Spain['Tonnes_Yield']
data2 = df_ML_Spain['mean_rain']
stat, p = spearmanr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')


# In[79]:


#Spearman’s Rank Correlation for Ireland Yield vs temp
from scipy.stats import spearmanr
data1 = df_ML_Spain['Tonnes_Yield']
data2 = df_ML_Spain['mean_temp']
stat, p = spearmanr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')


# Chi-square 

# In[80]:


df_stats_Ireland = df_ML1.copy() #must use .copy to avoid parsing changes to original df


# In[83]:


df_stats_Ireland.drop(['Type_of_Crop','Statistic', 'Year','UNIT'], axis=1, inplace=True)


# In[84]:


df_stats_Ireland


# In[85]:


# Chi-Squared Test Ireland
from scipy.stats import chi2_contingency
table = df_stats_Ireland
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')


# In[86]:


df_stats_Spain = df_ML_Spain.copy() 


# In[87]:


df_stats_Spain.drop(['Crop_Type','year'], axis=1, inplace=True)


# In[88]:


# Chi-Squared Test Spain
from scipy.stats import chi2_contingency
table = df_stats_Spain
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')


# Mann-Whitney U Test

# In[89]:


#Mann-Whitney U Test yield for Ireland vs Spain
from scipy.stats import mannwhitneyu
data1 = df_ML_Spain['Tonnes_Yield']
data2 = df_ML1['Yield_000_Tonnes']
stat, p = mannwhitneyu(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')


# In[90]:


#Mann-Whitney U Test temp for Ireland vs Spain
from scipy.stats import mannwhitneyu
data1 = df_ML_Spain['mean_temp']
data2 = df_ML1['mean_temp']
stat, p = mannwhitneyu(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')


# Kruskal-Wallis H Test

# In[91]:


# Kruskal-Wallis H Test yield Ireland vs Spain
from scipy.stats import kruskal
data1 = df_ML_Spain['Tonnes_Yield']
data2 = df_ML1['Yield_000_Tonnes']
stat, p = kruskal(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')


# In[92]:


# Kruskal-Wallis H Test temp Ireland vs Spain
from scipy.stats import kruskal
data1 = df_ML_Spain['mean_temp']
data2 = df_ML1['mean_temp']
stat, p = kruskal(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')


# <b>Visualzation</b>

# In[93]:


df_vis_IRL = df_ML1.copy()
df_vis_SPA = df_ML_Spain.copy()
df_vis_UK = df_ML_UK.copy()


# In[94]:


df_vis_IRL


# In[95]:


df_vis_IRL.rename(columns = {'Yield_000_Tonnes':'Tonnes_Yield'}, inplace = True)
df_vis_IRL.rename(columns = {'Type_of_Crop':'Crop_Type'}, inplace = True)


# In[96]:


df_vis_IRL.insert(0,'Country','Ireland')


# In[97]:


df_vis_IRL['Tonnes_Yield'] = df_vis_IRL['Tonnes_Yield'].apply(lambda x: x*1000)


# In[98]:


df_vis_IRL.drop(['Statistic','UNIT'], axis=1, inplace=True)


# In[99]:


#df_vis_IRL['total_rain'] = df_vis_IRL['total_rain'].apply(lambda x:/22) #dividing by 22 to get mean (max year 2007- min year1985)
df_vis_IRL['total_rain'] = df_vis_IRL['total_rain'].div(22).round(2)


# In[100]:


df_vis_IRL.rename(columns = {'total_rain':'mean_rain'}, inplace = True)


# In[101]:


df_vis_IRL.rename(columns = {'Year':'year'}, inplace = True)


# In[102]:


df_vis_SPA


# In[103]:


df_vis_SPA.insert(0,'Country','Spain')


# In[104]:


cols = list(df_vis_SPA.columns.values)
cols


# In[105]:


df_vis_SPA = df_vis_SPA[['Country','year', 'Crop_Type', 'Tonnes_Yield','mean_temp', 'mean_rain']]


# In[106]:


df_vis_UK 


# In[107]:


df_vis_UK.describe()


# In[108]:


df_vis_UK['rain'] = df_vis_UK['rain'].div(59).round(2) #finding mean same as above


# In[109]:


df_vis_UK.rename(columns = {'rain':'mean_rain'}, inplace = True)


# In[110]:


df_vis_UK.insert(0,'Country','United Kingdom')


# In[111]:


df_vis_UK = df_vis_UK[['Country','year', 'Crop_Type', 'Tonnes_Yield','mean_temp', 'mean_rain']]


# In[112]:


###merging###
frames = [df_vis_IRL, df_vis_UK, df_vis_SPA]
df_visualization = pd.concat(frames)


# In[113]:


df_visualization


# In[114]:


df_visualization.info()


# In[115]:


df_vis_year=df_visualization.groupby(['Country','year']).agg({'Tonnes_Yield':'sum', 'mean_temp':'mean','mean_rain':'mean'}).round(2).reset_index()


# In[116]:


#removing any year bellow 1985 and above 2007
df_vis_year_2007 = df_vis_year.copy()
df_vis_year_2007 = df_vis_year_2007.drop(df_vis_year_2007[df_vis_year_2007.year > 2007].index)
df_vis_year_2007 = df_vis_year_2007.drop(df_vis_year_2007[df_vis_year_2007.year < 1985].index)


# Comparison

# In[117]:


#comparing crop yields
import plotly.express as px
fig = px.line(df_vis_year_2007, x="year", y="Tonnes_Yield", color ="Country", title='Yield compare')
fig.update_traces(mode="markers+lines")
fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()


# In[118]:


#comparing data up to intereactive
df = px.data.gapminder()
fig = px.bar(df_vis_year_2007, x="Country", y="Tonnes_Yield", animation_frame="year", animation_group="Country",
            color="Country", hover_name="Country"
           )

fig["layout"].pop("updatemenus") # optional, drop animation buttons
fig.show()


# In[119]:


#comparing mean temp
fig = px.line(df_vis_year_2007, x="year", y="mean_temp", color ="Country", title='Temperature compare')
fig.update_traces(mode="markers+lines")
fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()


# In[120]:


#Rain comparison
fig = px.line(df_vis_year_2007, x="year", y="mean_rain", color ="Country", title='Rain compare')
fig.update_traces(mode="markers+lines")
fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()


# In[121]:


#cmparing yield+drill down into type of crop
fig = px.bar(df_visualization, x="Country", y="Tonnes_Yield", color="Crop_Type", title="comparison with crop type")
fig.show()


# In[122]:


#histfunc='avg',
fig = px.histogram(df_vis_year_2007, x="year", y="Tonnes_Yield",
             color='Country', barmode='group',
             
             height=400)
fig.show()


# <h1>Machine Learning</h1>

# In[123]:


df_ML1 = df_crop_temp_rain.copy()


# In[124]:


df_ML1.drop(['Statistic', 'Year','UNIT'], axis=1, inplace=True)


# In[125]:


df_ML1 


# In[126]:


df_ML1.info()


# In[127]:


df_dum = pd.get_dummies(df_ML1 ['Type_of_Crop'])
X = df_ML1.drop('Type_of_Crop', axis = 1) # setting X as independent features appart from Zone which will be our dependant features
y = df_ML1['Type_of_Crop'] # setting y as dependent feature Zone
X = pd.concat([X,df_dum], axis = 1)
X = X.select_dtypes(exclude=['object'])
X['Type_of_Crop'] = y
X


# In[128]:


#seperating and preparing the dataset for ML modeling
#as outlined above our X are the independent features and y is our dependent feature
X_new = X.drop('Type_of_Crop', axis = 1)
y = X['Type_of_Crop']

# 70/30 split as per texbooks
X_train, X_test, y_train, y_test = train_test_split(X_new,y, test_size = 0.3, random_state = 42)##removed stratisfy
#ref for random state
#https://stackoverflow.com/questions/42191717/scikit-learn-random-state-in-splitting-dataset


# In[129]:


# function for evaluation metrics precision, recall, f1 etc
#this way we save time printing results,tables and graphs at the end of each ML model
#simillar functions exist or can be written for data visualization
#concepts taken from bellow+Udemy course mentioned above
#ref https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
#ref https://stackoverflow.com/questions/40249943/adding-percentage-labels-to-a-bar-chart-in-ggplot2
def modelEvaluation(model,predictions, y_test_set, model_name):
    #defining modelEvaluation with existing methods/functions from ML libraries
    kfold = KFold(n_splits=10)
    #kfold cross validation set to 10 folds
    # Print model evaluation to predicted result everytime we will call modelEvaluation  
    start_time = time.time()
    print("==========",model_name,"==========")
    print ("\nAccuracy on validation set: {:.4f}".format(accuracy_score(y_test_set, predictions)))
    print ("Precision on validation set: {:.4f}".format(precision_score(y_test_set, predictions, average='macro')))    
    print ("Recall on validation set: {:.4f}".format(recall_score(y_test_set, predictions, average='macro')))
    print ("F1_Score on validation set: {:.4f}".format(f1_score(y_test_set, predictions, average='macro')))

    scores = cross_validate(model, X_test, y_test, cv=kfold, scoring='accuracy')
    #snipped of code taken from the Udemy lecture notes
    print ("KFold Cross Validation on validation set: {:.4f}".format(scores['test_score'].mean()))
    
    
    print ("\nClassification report : \n", classification_report(y_test_set, predictions))
    #printing y_test scores
    print("="*30)
    print ("\tConfusion Matrix",)
    print("="*30)
    #printing a seperator
    
    #defining and making conusion matrix
    cm = confusion_matrix(y_test_set, predictions)
    print(cm)
    #pritnting confusion matrix
    
    #plotting the confusion matrix and adding additional % lables
    #simillar functions exist and can be written for data visualization to add % or data on top of bar plots etc.
    #concepts taken form bellow+Udemy course mentioned above
    ##ref https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
    #ref https://stackoverflow.com/questions/40249943/adding-percentage-labels-to-a-bar-chart-in-ggplot2
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum()) for item in cm.flatten()]

        ]
    ).reshape(16, 16)

    plt.figure(figsize=(16, 16))

    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    end = round(time.time()-start_time,3)
    results = [accuracy_score(y_test_set, predictions),precision_score(y_test_set, predictions, average='macro'),
              recall_score(y_test_set, predictions, average='macro'),f1_score(y_test_set, predictions, average='macro'),scores['test_score'].mean(),end]
    return results


# In[130]:


###Tried to apply gridsearchCV to get params, to stop random forest overfitting, but all params that I have tried overfit
#from sklearn.model_selection import GridSearchCV

#rfc = RandomForestClassifier (random_state = 42)

#param_grid = {'n_estimators' :[200,500],
             #'max_features':['auto','sqr','log2'],
             #'max_depth':[4,5,6,7,8],
             #'criterion':['gini','entropy']}
#CV_rfc = GridSearchCV(estimator =rfc,param_grid=param_grid,cv=5)

#CV_rfc.fit(X_train,y_train)
#print (CV_rfc.best_params_)


#import sklearn.metrics as metrics
#y_pred = CV_rfc.predict(X_test)
#X_pred = CV_rfc.predict(X_train)
#print('train accuracy: ',metrics.accuracy_score(y_test,y_pred))
#print('test accuracy: ',metrics.accuracy_score(y_train,X_pred))


# In[131]:


#same as above,tried to apply gridsearchCV to get params, to stop random forest overfitting, but all params that I have tried overfit
#from sklearn.model_selection import GridSearchCV
#from sklearn.datasets import make_classification
#from sklearn.ensemble import RandomForestClassifier
# Build a classification task using 3 informative features
#X, y = make_classification(n_samples=1000,
                           #n_features=10,
                           #n_informative=3,
                           #n_redundant=0,
                           #n_repeated=0,
                           #n_classes=2,
                           #random_state=0,
                           #shuffle=False)


#rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

#param_grid = { 
    #'n_estimators': [200, 700],
    #'max_features': ['auto', 'sqrt', 'log2']
#}

#CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
#CV_rfc.fit(X, y)
#print (CV_rfc.best_params_)


# In[132]:


#Logistic Regression model
#using above function with ML library we can start building our ML models
#please note that you need to run these models in the order as in this notebook and all the cells as well
#if one was to jump from one cell and skip to another, you will then need to reset the kernel
#as the krnel stores executions,data and results
lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
results_lr = modelEvaluation(lr,predictions, y_test, "logistic Regression")


# In[133]:


kfold = KFold(n_splits=10)

scores = cross_validate(lr, X_test, y_test, cv=kfold, scoring='accuracy',return_train_score=True)
print(f"Average Testing Accuracy : {scores['test_score'].mean()}")
print(f"Testing Accuracy : {scores['test_score']}")
#using snippet from the function above
#plotting our results of test vs train
plt.plot(scores['train_score'] ,label ='training score', marker='o')
plt.plot(scores['test_score'] ,label ='testing score',marker='o')
plt.title("Logistic Regression Cross Validation Results")
plt.xlabel("Cross Validation")
plt.ylabel("Score")
plt.legend()
plt.show()


# In[134]:


#Decision Tree Model
dt = DecisionTreeClassifier(criterion='gini',
                            max_depth=5 ,
                             min_samples_split=10,
                             min_samples_leaf=22,
                             )
                             
dt.fit(X_train,y_train)
prediction = dt.predict(X_test)
results_mp = modelEvaluation(dt,prediction, y_test,"DecisionTreeClassifier")


# In[135]:


from sklearn.model_selection import cross_validate

scores = cross_validate(dt, X_test, y_test, cv=kfold, scoring='accuracy',return_train_score=True)
print(f"Average Testing Accuracy : {scores['test_score'].mean()}")
print(f"Testing Accuracy : {scores['test_score']}")

plt.plot(scores['train_score'] ,label ='training score', marker='o')
plt.plot(scores['test_score'] ,label ='testing score',marker='o')
plt.title("Decision Tree Cross Validation Results")
plt.xlabel("Cross Validation")
plt.ylabel("Score")
plt.legend()
plt.show()


# In[136]:


rf = RandomForestClassifier(criterion='gini',
                            max_depth=16 , #was 8 testing 16
                            n_estimators = 200,
                         max_features ='log2'
                            
                             )
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
results_rf = modelEvaluation(rf,predictions, y_test, "RandomForestClassifier")


# In[137]:


from sklearn.model_selection import cross_validate

scores = cross_validate(rf, X_test, y_test, cv=kfold, scoring='accuracy',return_train_score=True)
print(f"Average Testing Accuracy : {scores['test_score'].mean()}")
print(f"Testing Accuracy : {scores['test_score']}")

plt.plot(scores['train_score'] ,label ='training score', marker='o')
plt.plot(scores['test_score'] ,label ='testing score',marker='o')
plt.title("Random Forest Cross Validation Results")
plt.xlabel("Cross Validation")
plt.ylabel("Score")
plt.legend()
plt.show()


# Machine Learning Spain

# In[145]:


df_dum = pd.get_dummies(df_ML_Spain ['Crop_Type'])
X = df_ML_Spain.drop('Crop_Type', axis = 1) # setting X as independent features appart from Zone which will be our dependant features
y = df_ML_Spain['Crop_Type'] # setting y as dependent feature Zone
X = pd.concat([X,df_dum], axis = 1)
X = X.select_dtypes(exclude=['object'])
X['Crop_Type'] = y
X


# In[147]:


#seperating and preparing the dataset for ML modeling
#as outlined above our X are the independent features and y is our dependent feature
X_new = X.drop('Crop_Type', axis = 1)
y = X['Crop_Type']

# 70/30 split as per texbooks
X_train, X_test, y_train, y_test = train_test_split(X_new,y, test_size = 0.3, random_state = 42)##removed stratisfy
#ref for random state
#https://stackoverflow.com/questions/42191717/scikit-learn-random-state-in-splitting-dataset


# In[155]:


##running model evaluation defined above for ML1 part
def modelEvaluation(model,predictions, y_test_set, model_name):
    #defining modelEvaluation with existing methods/functions from ML libraries
    kfold = KFold(n_splits=10)
    #kfold cross validation set to 10 folds
    # Print model evaluation to predicted result everytime we will call modelEvaluation  
    start_time = time.time()
    print("==========",model_name,"==========")
    print ("\nAccuracy on validation set: {:.4f}".format(accuracy_score(y_test_set, predictions)))
    print ("Precision on validation set: {:.4f}".format(precision_score(y_test_set, predictions, average='macro')))    
    print ("Recall on validation set: {:.4f}".format(recall_score(y_test_set, predictions, average='macro')))
    print ("F1_Score on validation set: {:.4f}".format(f1_score(y_test_set, predictions, average='macro')))

    scores = cross_validate(model, X_test, y_test, cv=kfold, scoring='accuracy')
    #snipped of code taken from the Udemy lecture notes
    print ("KFold Cross Validation on validation set: {:.4f}".format(scores['test_score'].mean()))
    
    
    print ("\nClassification report : \n", classification_report(y_test_set, predictions))
    #printing y_test scores
    print("="*30)
    print ("\tConfusion Matrix",)
    print("="*30)
    #printing a seperator
    
    #defining and making conusion matrix
    cm = confusion_matrix(y_test_set, predictions)
    print(cm)
    #pritnting confusion matrix
    
    #plotting the confusion matrix and adding additional % lables
    #simillar functions exist and can be written for data visualization to add % or data on top of bar plots etc.
    #concepts taken form bellow+Udemy course mentioned above
    ##ref https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
    #ref https://stackoverflow.com/questions/40249943/adding-percentage-labels-to-a-bar-chart-in-ggplot2
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum()) for item in cm.flatten()]

        ]
    ).reshape(103, 103)

    plt.figure(figsize=(16, 16))

    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    end = round(time.time()-start_time,3)
    results = [accuracy_score(y_test_set, predictions),precision_score(y_test_set, predictions, average='macro'),
              recall_score(y_test_set, predictions, average='macro'),f1_score(y_test_set, predictions, average='macro'),scores['test_score'].mean(),end]
    return results


# In[156]:


lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
results_lr = modelEvaluation(lr,predictions, y_test, "logistic Regression")


# In[157]:


kfold = KFold(n_splits=10)

scores = cross_validate(lr, X_test, y_test, cv=kfold, scoring='accuracy',return_train_score=True)
print(f"Average Testing Accuracy : {scores['test_score'].mean()}")
print(f"Testing Accuracy : {scores['test_score']}")
#using snippet from the function above
#plotting our results of test vs train
plt.plot(scores['train_score'] ,label ='training score', marker='o')
plt.plot(scores['test_score'] ,label ='testing score',marker='o')
plt.title("Logistic Regression Cross Validation Results")
plt.xlabel("Cross Validation")
plt.ylabel("Score")
plt.legend()
plt.show()


# In[158]:


#Decision Tree Model
dt = DecisionTreeClassifier(criterion='gini',
                            max_depth=5 ,
                             min_samples_split=10,
                             min_samples_leaf=22,
                             )
                             
dt.fit(X_train,y_train)
prediction = dt.predict(X_test)
results_mp = modelEvaluation(dt,prediction, y_test,"DecisionTreeClassifier")


# In[159]:


from sklearn.model_selection import cross_validate

scores = cross_validate(dt, X_test, y_test, cv=kfold, scoring='accuracy',return_train_score=True)
print(f"Average Testing Accuracy : {scores['test_score'].mean()}")
print(f"Testing Accuracy : {scores['test_score']}")

plt.plot(scores['train_score'] ,label ='training score', marker='o')
plt.plot(scores['test_score'] ,label ='testing score',marker='o')
plt.title("Decision Tree Cross Validation Results")
plt.xlabel("Cross Validation")
plt.ylabel("Score")
plt.legend()
plt.show()


# In[160]:


rf = RandomForestClassifier(criterion='gini',
                            max_depth=16 , #was 8 testing 16
                            n_estimators = 200,
                         max_features ='log2'
                            
                             )
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
results_rf = modelEvaluation(rf,predictions, y_test, "RandomForestClassifier")


# In[161]:


from sklearn.model_selection import cross_validate

scores = cross_validate(rf, X_test, y_test, cv=kfold, scoring='accuracy',return_train_score=True)
print(f"Average Testing Accuracy : {scores['test_score'].mean()}")
print(f"Testing Accuracy : {scores['test_score']}")

plt.plot(scores['train_score'] ,label ='training score', marker='o')
plt.plot(scores['test_score'] ,label ='testing score',marker='o')
plt.title("Random Forest Cross Validation Results")
plt.xlabel("Cross Validation")
plt.ylabel("Score")
plt.legend()
plt.show()


# In[ ]:


######interactive dashboard####


# In[226]:


######This dataframe is prepped from Ireland X df and exported to csv so it can be used for interactive ML dashboard, which is attached as seperate workbook##
Xdata = X.copy()
Xdata .columns = Xdata .columns.str.replace(' ','_')
Xdata .Type_of_Crop = pd.Categorical(Xdata .Type_of_Crop)
Xdata ['Crop_Cat_Id'] = Xdata.Type_of_Crop.cat.codes
Xdata  = Xdata.drop(['Type_of_Crop'], axis = 1)
Xdata 


# In[42]:


print(Xdata.columns)


# In[228]:


Xdata.to_csv('Xdata.csv')


# In[ ]:





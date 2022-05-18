#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install selenium


# In[1]:


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

#import os
#import selenium
#import time
#from PIL import Image
#import io
#import requests
#from selenium.common.exceptions import ElementClickInterceptedException


# In[2]:


driver = webdriver.Chrome(executable_path= r'C:\Users\niyaz\Downloads\chromedriver_win32 (1)\chromedriver.exe')


# In[3]:


urlQuery = "https://www.farmersjournal.ie/archive.php#stq=crop%20yield&stp=1"
driver.get(urlQuery)
elems = driver.find_elements_by_css_selector(".st-result [href]")
links = [elem.get_attribute('href') for elem in elems]


# In[4]:


for link in links:
    print(link)


# In[5]:


driver.get(urlQuery)
elems = driver.find_elements_by_css_selector(".st-page [href]")
pages = [elem.get_attribute('href') for elem in elems]


# In[6]:


for page in pages:
    print(page)


# In[8]:


driver.get(links[0])
page = driver.find_element_by_xpath("/html/body")
print('URL Title: \n {}'.format(page.find_element(By.CLASS_NAME,'article-title').text)) ##title
#print('URL Date: \n {}'.format(page.find_element(By.CLASS_NAME,'gray').text)) ##date
print('URL Title: \n {}'.format(page.find_element(By.CLASS_NAME,'article-body').text))###contnet


# In[28]:


links=[]
the_links=driver.find_elements_by_class_name("table-list-item")
for link in the_links:
    links.append(link.get_attribute('href'))


# In[33]:


for link in links:
    driver.get(link)
    print("do something on this link")


# In[12]:


for link in links:
    driver.get(link)
    page = driver.find_element_by_xpath("/html/body")
    print('URL Title: \n {}'.format(page.find_element(By.CLASS_NAME,'article-title').text)) ##title
#print('URL Date: \n {}'.format(page.find_element(By.CLASS_NAME,'gray').text)) ##date
    print('URL body: \n {}'.format(page.find_element(By.CLASS_NAME,'article-body').text))###contnet


# In[15]:


for link in links:
    driver.get(link)
    page = driver.find_element_by_xpath("/html/body")
    
    print('URL Title: \n {}'.format(page.find_element(By.CLASS_NAME,'article-title').text)) ##title
#print('URL Date: \n {}'.format(page.find_element(By.CLASS_NAME,'gray').text)) ##date
    print('URL body: \n {}'.format(page.find_element(By.CLASS_NAME,'article-body').text))###contnet


# In[16]:


f = open('workfile', 'w')
for link in links:
    driver.get(link)
    page = driver.find_element_by_xpath("/html/body")
    
    f.write('URL Title: \n {}'.format(page.find_element(By.CLASS_NAME,'article-title').text)) ##title
#print('URL Date: \n {}'.format(page.find_element(By.CLASS_NAME,'gray').text)) ##date
    f.write('URL body: \n {}'.format(page.find_element(By.CLASS_NAME,'article-body').text))###contnet
    
f.close()


# In[ ]:





# In[26]:


import csv
with open('sentimentCSV.csv' 'w') as file:
    writer = csv.writer(file)
    for link in links(lenght):
        driver.get(link)
        page = driver.find_element_by_xpath("/html/body")
        Title = "ti".format(page.find_element(By.CLASS_NAME,'article-title').text)
        content = "co".format(page.find_element(By.CLASS_NAME,'article-body').text
        writer.writerow([Title, Content])


# In[ ]:


#####write and add comma!!!


# In[ ]:


with open ('/home/shayez/Desktop/karim.csv','wt') as csvfile:
    writer = csv.writer(csvfile, delimiter ="\t" )
    writer.writerow(header)

    for result in records :
        PMID = result['PMID']
        Abstract = result['AB']
        title = result['TI']
        Date = result['DP']

        print (PMID,Date,title,Abstract)

        fields = [PMID, title,Date,Abstract]
        rows = [PMID,Date,title,Abstract]

        writer.writerow(rows)


# In[29]:


f = open('SentimentTextFile.txt', 'r')
file_contents = f.read()
text = (file_contents)


# In[30]:


print(text)


# In[32]:


from textblob import TextBlob
blob = TextBlob(text)

sentiment = blob.sentiment.polarity
print(sentiment)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Nearest Earth Objects Portfolio Project

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc


# In[2]:


df = pd.read_csv('neo_v2.csv')


# In[3]:


# how many records?
len(df)


# In[4]:


# how many attributes?
df.columns


# In[5]:


# peek at the data
df.head(10)


# ### Attributes
# - id: identifier (same ones are used multiple times)
# - name: given by NASA
# - est_diameter_min: minimum estimated diameter in kilometers
# - est_diameter_max: maximum estimated diameter in kilometers
# - relative_velocity: velocity relative to earth
# - miss_distance: distance in kilometers missed
# - orbiting_body: planet that the asteroid orbits
# - sentry_object: Included in sentry - an automated collision monitoring system
# - absolute_magnitude: intrinsic luminosity
# - hazardous: whether the asteriod is harmful or not

# In[6]:


print(df.nunique())


# Delete orbiting_body and sentry_object

# In[7]:


df.drop(["orbiting_body", "sentry_object"], axis = 1, inplace = True)


# In[8]:


df.head()


# See why same id is recorded several times

# In[9]:


df[df['id'] == 2512244]


# See if categories are disproportionate  
# about 10% of asteroids are hazardous

# In[10]:


print(df["hazardous"].value_counts() / len(df))


# In[11]:


sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# Delete the diameter vars because they are perfectly negatively correlated with absolute magnitude

# In[12]:


df.drop(["est_diameter_min"], axis = 1, inplace = True)


# In[13]:


# Extracting year from name to see later on whether we could have a pattern
df[['drop','work']]=df.name.str.split('(',expand=True)


# In[14]:


df.drop(columns='drop',inplace=True)

def year_extract(x):
    return x.strip()[0:x.strip().index(' ')]
df['year']=df['work'].apply(year_extract)


# In[15]:


df.drop(columns='work', inplace=True)


# In[16]:


df['year'].unique()


# In[17]:


# [A911,6743,A898,4788,6344,A924,A/2019] --> Years to be changed 

df.loc[df.year=='A911','year']='1911' 
df.loc[df.year=='6743','year']='1960'
df.loc[df.year=='A898','year']='1898'
df.loc[df.year=='6344','year']='1960'
df.loc[df.year=='A924','year']='1924'
df.loc[df.year=='A/2019','year']='2019'
df.loc[df.year=='4788','year']='1960'

df.year.unique()


# In[42]:


#sns.boxplot(x='hazardous',y='year',data=df)
#plt.show()


# Inference from the boxplot
# 
#     Hazardous objects were mostly discovered between around 2002 to before 2020
#     There were many non-hazardous objects discovered pre-1980s (is this because hazardous objects tend to be farther away so harder to detect with older equipment, also non-hazardous objects tend to have higher magnitude so easier to spot)

# ## Univariate and Bivariate analysis

# In[20]:


num_cols = ["est_diameter_max", "relative_velocity", "miss_distance", "absolute_magnitude"]


# In[21]:


rows=2
cols=2
count=1
plt.rcParams['figure.figsize']=[10,10]
for i in num_cols:
    plt.subplot(rows,cols,count)
    sns.distplot(df[i], color='c')
    count+=1
plt.tight_layout()
plt.show()


# Inferences From the subplots
# 
#     Estimated minimum Diameter and estimated maximum Diameter are highly positively skewed with sharp spike at on end, indicating the presence of outliers
#     Relative Velocity is mildly positive skewed
#     miss_distance is uniform over the data
#     Absolute magnitude is the closest one to a normal distribution curve

# In[23]:


sns.pairplot(df[num_cols+['hazardous','year']],hue = 'hazardous')


# In[24]:


# Making subplots for multiple boxplots and comparing with our target variable

rows=3
cols=2
counter=1
plt.rcParams['figure.figsize']=[10,8]
for i in num_cols:
    plt.subplot(rows,cols,counter)
    sns.boxplot(x='hazardous',y=i,data=df)
    counter+=1
    
plt.tight_layout()
plt.show()


# Inferences from the boxplots:
# 
#     Non-Hazardous objects have a higher estimated minimum and maximum diameter
#     Our Hypothesis is that Hazardous objects have a higher relative_velocity
#     It seems that hazardous objects have a slightly higher miss_distance
#     The graphs indicate that Non-Hazardous objects have considerably higher absolute_magnitude.

# # Classifier Models

# In[25]:


X = df.drop(["id","name","hazardous"], axis=1)
y = df.hazardous.astype(int)


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0, stratify=y)


# In[27]:


sc=StandardScaler() # Machine Instance
X_train_scaled=pd.DataFrame(sc.fit_transform(X_train)) # Scaling the train set
X_test_scaled=pd.DataFrame(sc.transform(X_test)) # Scaling the test set


# In[28]:


def roc_curve_plot(y_test, y_scores, method):
    fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend()
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve of ' + method)
    plt.show()
    return roc_auc


# ## Decision Tree

# In[29]:


# DecisionTree Model

DT = DecisionTreeClassifier()
DT.fit(X_train_scaled, y_train)
DT_pred = DT.predict(X_test_scaled)
Acc_DT = round(accuracy_score(DT_pred, y_test), 4)
xgprec_DT, xgrec_DT, xgf_DT, support_DT = score(y_test, DT_pred)
precision_DT, recall_DT, f1_DT = round(xgprec_DT[0], 4), round(xgrec_DT[0],4), round(xgf_DT[0],4)
scores_DT = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Score': [Acc_DT, precision_DT, recall_DT, f1_DT]})
scores_DT


# In[30]:


y_scores_DT = DT.predict_proba(X_test_scaled)
auc_DT = roc_curve_plot(y_test, y_scores_DT, 'Decision Tree')


# ## K Nearest Neighbors

# In[31]:


KNN = KNeighborsClassifier(n_neighbors = 4)
KNN.fit(X_train_scaled, y_train)
KNN_pred = KNN.predict(X_test_scaled)
Acc_KNN = round(accuracy_score(KNN_pred, y_test), 4)
xgprec_KNN, xgrec_KNN, xgf_KNN, support_KNN = score(y_test, KNN_pred)
precision_KNN, recall_KNN, f1_KNN = round(xgprec_KNN[0], 4), round(xgrec_KNN[0],4), round(xgf_KNN[0],4)
scores_KNN = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Score': [Acc_KNN, precision_KNN, recall_KNN, f1_KNN]})
scores_KNN


# In[32]:


y_scores_KNN = KNN.predict_proba(X_test_scaled)
auc_KNN = roc_curve_plot(y_test, y_scores_KNN, 'kNN')


# ## Random Forest

# In[33]:


# Random Forest Model
RF = RandomForestClassifier()
RF.fit(X_train_scaled, y_train)
RF_pred = RF.predict(X_test_scaled)
Acc_RF = round(accuracy_score(RF_pred, y_test), 4)
xgprec_RF, xgrec_RF, xgf_RF, support_RF = score(y_test, RF_pred)
precision_RF, recall_RF, f1_RF = round(xgprec_RF[0], 4), round(xgrec_RF[0],4), round(xgf_RF[0],4)
scores_RF = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Score': [Acc_RF, precision_RF, recall_RF, f1_RF]})
scores_RF


# In[34]:


y_scores_RF = RF.predict_proba(X_test_scaled)
auc_RF = roc_curve_plot(y_test, y_scores_RF, 'Random Forest')


# ## Gradient Boosted Decision Tree

# In[35]:


# Gradient Boosted Decision Tree Model
XGB = XGBClassifier()
XGB.fit(X_train_scaled, y_train)
XGB_pred = XGB.predict(X_test_scaled)
Acc_XGB = round(accuracy_score(XGB_pred, y_test),4)
xgprec_XGB, xgrec_XGB, xgf_XGB, support_XGB = score(y_test, XGB_pred)
precision_XGB, recall_XGB, f1_XGB = round(xgprec_XGB[0], 4), round(xgrec_XGB[0],4), round(xgf_XGB[0],4)
scores_XGB = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Score': [Acc_XGB, precision_XGB, recall_XGB, f1_XGB]})
scores_XGB


# In[36]:


y_scores_XGB = XGB.predict_proba(X_test_scaled)
auc_XGB = roc_curve_plot(y_test, y_scores_XGB, 'Gradient Boosted Decision Tree')


# # Compare Models

# In[37]:


accuracy_table = pd.DataFrame({
    'Model': ['Random Forest', 'XG Boost','K Nearest Neighbors', 'Decision tree'],
    'Accuracy': [Acc_RF, Acc_XGB, Acc_KNN, Acc_DT]})
accuracy_table.sort_values(by='Accuracy', ascending=False)


# In[38]:


precision_table = pd.DataFrame({
    'Model': ['Random Forest', 'XG Boost','K Nearest Neighbors', 'Decision tree'],
    'Precision': [precision_RF, precision_XGB, precision_KNN, precision_DT]})
precision_table.sort_values(by='Precision', ascending=False)


# In[39]:


recall_table = pd.DataFrame({
    'Model': ['Random Forest', 'XG Boost','K Nearest Neighbors', 'Decision tree'],
    'Recall': [recall_RF, recall_XGB, recall_KNN, recall_DT]})
recall_table.sort_values(by='Recall', ascending=False)


# In[40]:


f1_table = pd.DataFrame({
    'Model': ['Random Forest', 'XG Boost','K Nearest Neighbors', 'Decision tree'],
    'F1': [f1_RF, f1_XGB, f1_KNN, f1_DT]})
f1_table.sort_values(by='F1', ascending=False)


# In[41]:


auc_table = pd.DataFrame({
    'Model': ['Random Forest', 'XG Boost','K Nearest Neighbors', 'Decision tree'],
    'AUC': [auc_RF, auc_XGB, auc_KNN, auc_DT]})
auc_table.sort_values(by='AUC', ascending=False)


# Overall, it is clear that the random forest model performed the best  
# Random forest was not the best in recall and precision  
# Gradient boosted decision tree was a close second  
# If recall is most important, may be better to use XGB

# In[ ]:





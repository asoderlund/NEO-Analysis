
# # Nearest Earth Objects Portfolio Project: Alyssa Soderlund

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc


df = pd.read_csv('neo_v2.csv')

# how many records?
len(df)

# how many attributes?
df.columns

# peek at the data
df.head(10)

# How many unique values for each attribute?
print(df.nunique())


# Delete orbiting_body and sentry_object
df.drop(["orbiting_body", "sentry_object"], axis = 1, inplace = True)

# Look at just one object- what changes for each observation?
df[df['id'] == 2512244]

# Is there a class imbalance for hazardous?
print(df["hazardous"].value_counts() / len(df))

sns.countplot(x='hazardous',data=df)
plt.suptitle('Number of Non-Hazardous and Hazardous Objects')
plt.rcParams['figure.figsize']=[4,4]
plt.show()

# Look at bivariate correlations
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.suptitle('Bivariate Correlations')
plt.show()

# Drop one diameter variable since they are perfectly correlated
df.drop(["est_diameter_min"], axis = 1, inplace = True)

# Extracting year from name to see later on whether we could have a pattern
df[['drop','temp']]=df.name.str.split('(',expand=True)

df.drop(columns='drop',inplace=True)

def get_year(x):
    return x.strip()[0:x.strip().index(' ')]
df['year']=df['temp'].apply(get_year)
df.drop(columns='temp', inplace=True)

# Check year labels
df['year'].unique()

# Change abnormal years to the correct year

df.loc[df.year=='A911','year']='1911' 
df.loc[df.year=='6743','year']='1960'
df.loc[df.year=='A898','year']='1898'
df.loc[df.year=='6344','year']='1960'
df.loc[df.year=='A924','year']='1924'
df.loc[df.year=='A/2019','year']='2019'
df.loc[df.year=='4788','year']='1960'

df.year.unique()

# Change years to ints
df.year=df.year.astype(int)

# Boxplot for year and target variable
sns.boxplot(x='hazardous',y='year',data=df)
plt.suptitle('Box Plots for Hazardous Objects by Year')
plt.show()




# ## Univariate and Bivariate analysis
num_cols = ["est_diameter_max", "relative_velocity", "miss_distance", "absolute_magnitude","year"]

# Univariate distributions for numerical variables
rows=2
cols=3
count=1
plt.rcParams['figure.figsize']=[15,9]
for i in num_cols:
    plt.subplot(rows,cols,count)
    sns.distplot(df[i], color='c')
    count+=1
plt.suptitle('Distributions of Numerical Variables')
plt.show()

# Pairs plots, color coded by hazardous classification
sns.pairplot(df[num_cols+['hazardous']],hue = 'hazardous')




# # Classifier Models

# Create dataframes for predictor and target variables
X = df.drop(["id","name","hazardous"], axis=1)
y = df.hazardous.astype(int)

# Test train split 80/20, stratify y due to class imbalance
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0, stratify=y)

# Scale x for K nearest neighbors
sc=StandardScaler()
X_train_scaled=pd.DataFrame(sc.fit_transform(X_train))
X_test_scaled=pd.DataFrame(sc.transform(X_test))

# Define functions for plotting ROC curves
def roc_curve_plot(y_test, y_scores, method):
    fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend()
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve of ' + method)
    plt.rcParams['figure.figsize']=[6,5]
    plt.show()
    return roc_auc


# ## Decision Tree

# Create decision tree model
DT = DecisionTreeClassifier()
tree = DT.fit(X_train_scaled, y_train)
DT_pred = DT.predict(X_test_scaled)

# Create metrics table
Acc_DT = round(accuracy_score(DT_pred, y_test), 4)
xgprec_DT, xgrec_DT, xgf_DT, support_DT = score(y_test, DT_pred)
precision_DT, recall_DT, f1_DT = round(xgprec_DT[0], 4), round(xgrec_DT[0],4), round(xgf_DT[0],4)
scores_DT = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Score': [Acc_DT, precision_DT, recall_DT, f1_DT]})
scores_DT

# Plot ROC curve
y_scores_DT = DT.predict_proba(X_test_scaled)
auc_DT = roc_curve_plot(y_test, y_scores_DT, 'Decision Tree')

# Variable importance plot
feat_importances = pd.Series(DT.feature_importances_, index=X.columns)
feat_importances.plot(kind='barh', title='Variable Importance for Decision Tree',figsize=[5,3])


# ## K Nearest Neighbors

# Hyperparameter tuning for k
error_rates = []
for i in np.arange(1, 40):
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    error_rates.append(np.mean(predictions != y_test))

plt.rcParams['figure.figsize']=[6,4]
plt.suptitle('Error Rates for k from 1 to 40')
plt.plot(error_rates)


# Create KNN model
KNN = KNeighborsClassifier(n_neighbors = 15)
KNN.fit(X_train_scaled, y_train)
KNN_pred = KNN.predict(X_test_scaled)

# Create metrics table
Acc_KNN = round(accuracy_score(KNN_pred, y_test), 4)
xgprec_KNN, xgrec_KNN, xgf_KNN, support_KNN = score(y_test, KNN_pred)
precision_KNN, recall_KNN, f1_KNN = round(xgprec_KNN[0], 4), round(xgrec_KNN[0],4), round(xgf_KNN[0],4)
scores_KNN = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Score': [Acc_KNN, precision_KNN, recall_KNN, f1_KNN]})
scores_KNN


# Plot ROC curve
y_scores_KNN = KNN.predict_proba(X_test_scaled)
auc_KNN = roc_curve_plot(y_test, y_scores_KNN, 'kNN')


# ## Random Forest

# Hyperparameter tuning
max_depth=[2, 8, 16]
n_estimators = [64, 128, 256]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

dfrst = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
grid = GridSearchCV(estimator=dfrst, param_grid=param_grid, cv = 5)
grid_results = grid.fit(X_train_scaled, y_train)

print("Best: {0}, using {1}".format(grid_results.cv_results_['mean_test_score'], grid_results.best_params_))


# Create model
RF = RandomForestClassifier()
RF.fit(X_train_scaled, y_train)
RF_pred = RF.predict(X_test_scaled)

# Create metrics table
Acc_RF = round(accuracy_score(RF_pred, y_test), 4)
xgprec_RF, xgrec_RF, xgf_RF, support_RF = score(y_test, RF_pred)
precision_RF, recall_RF, f1_RF = round(xgprec_RF[0], 4), round(xgrec_RF[0],4), round(xgf_RF[0],4)
scores_RF = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Score': [Acc_RF, precision_RF, recall_RF, f1_RF]})
scores_RF

# Plot ROC curve
y_scores_RF = RF.predict_proba(X_test_scaled)
auc_RF = roc_curve_plot(y_test, y_scores_RF, 'Random Forest')

# Variable importance plot
feat_importances_RF = pd.Series(RF.feature_importances_, index=X.columns)
feat_importances_RF.nlargest(8).plot(kind='barh', title = 'Variable Importance for Random Forest', figsize=[5,3])


# ## Gradient Boosted Decision Tree

# Create model
XGB = XGBClassifier()
XGB.fit(X_train_scaled, y_train)
XGB_pred = XGB.predict(X_test_scaled)

# Create metrics table
Acc_XGB = round(accuracy_score(XGB_pred, y_test),4)
xgprec_XGB, xgrec_XGB, xgf_XGB, support_XGB = score(y_test, XGB_pred)
precision_XGB, recall_XGB, f1_XGB = round(xgprec_XGB[0], 4), round(xgrec_XGB[0],4), round(xgf_XGB[0],4)
scores_XGB = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Score': [Acc_XGB, precision_XGB, recall_XGB, f1_XGB]})
scores_XGB

# Plot ROC Cruve
y_scores_XGB = XGB.predict_proba(X_test_scaled)
auc_XGB = roc_curve_plot(y_test, y_scores_XGB, 'Gradient Boosted Decision Tree')

# Variable importance plot
feat_importances_XGB = pd.Series(XGB.feature_importances_, index=X.columns)
feat_importances_XGB.plot(kind='barh', title = 'Variable Importance for Gradient Boosted Tree', figsize=[5,3])


# # Compare Models table

total_table = pd.DataFrame({
    'Model': ['Random Forest', 'XG Boost','K Nearest Neighbors', 'Decision tree'],
    'AUC': [auc_RF, auc_XGB, auc_KNN, auc_DT],
    'Accuracy': [Acc_RF, Acc_XGB, Acc_KNN, Acc_DT],
    'Precision': [precision_RF, precision_XGB, precision_KNN, precision_DT],
    'Recall': [recall_RF, recall_XGB, recall_KNN, recall_DT],
    'F1': [f1_RF, f1_XGB, f1_KNN, f1_DT]})
total_table.sort_values(by='AUC', ascending=False)


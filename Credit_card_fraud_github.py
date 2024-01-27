#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection
# 
# 
# Credit Card Fraud Detection is a major problem that financial world is facing now a days.
# Even greater problem is to detect the fraudsters and fraud transactions. To tackle this we need AI to automate the process, because we, a humans, its merely impossible to search the fraud transactions in a short period of time. 
# 
# Now, we use Machine learning algorithms here to find the the number of fraud transactions. It's number is much lesser than the number of legitimate transaction for any bank. Most approaches involve building model on such imbalanced data, and thus fails to produce results on real-time new data because of overfitting on training data which is bias towards the class of legitimate transactions (major class). 
# Thus, we can see this as an anomaly detection problem.
# 
# ## Here, I found out the answer to 2 of the major problems in anomaly.
# 
# #### Question.1:  At what time does the Credit Card Frauds takes place usually?
# #### Question.2:  On legitimate transactions, how do we balance the data and dont let it overfit?

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from mlxtend.plotting import plot_learning_curves
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold


# In[3]:


df = pd.read_csv('credit.csv')
df.head()


# # About Data
# 
# The Data has 32 features which are unknown for Time, Amount and Class(From V1-V28).
# 
# ##### Input features are V1-V28, Time and Amount 
# ##### Target variable is Class
# 
# 

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


# Check for missing values
print(df.isnull().sum())



# # About Data
# 
# #### The Data doesn't have any missing values, so, no need not be handled.
# The Data has all numerical features except Target Variable Class which is a categorical feature.
# 
# Class 0: Legitimate Transaction
# Class 1: Fraud Transaction

# In[7]:


df.describe()


# # Data Declaration
# 
# 
# - Mean and StdDev of Amount of Data is shown above.
# - In this case, I will not delete or drop any data.

# In[13]:


def countplot_data(data, feature):
    plt.figure(figsize=(6,4))
    sns.countplot(x=feature, data=data)
    plt.show()

def pairplot_data_grid(data, feature1, feature2, target):
    sns.FacetGrid(data, hue=target, size=6).map(plt.scatter, feature1, feature2).add_legend()
    plt.show()
        


# In[14]:


countplot_data(df, df.Class)


# # Insights:
# 
# - The target variable is Class and rest are are input features.
# - 0 = Legitimate Transactions
# - 1 = Fraud Transactions
# 
# As we can see Dataset is highly imbalanced 
# Major class label = 0 and minor class label 1.
# Now, we will perform Synthetic Minority Oversampling on the data to balance it out
# 
# What is relationship of fraud transactions with amount of money?
# Let us try to determine the nature of transactions which are fraud and obtain a relevant set of the same with respect to their amount.
# 
# #### Results: All fraud transactions occur for an amount less than 2500.

# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

def pairplot_data_grid(data, feature1, feature2, target):
    sns.FacetGrid(data, hue=target, height=6).map(plt.scatter, feature1, feature2).add_legend()
    plt.show()

# Example usage
pairplot_data_grid(df, "Time", "Amount", "Class")


# ## Insights:
# 
# - The fraud transactions are evenly distributed about time.
# - The fraud transactions are generally not above an amount of 2500.
# 

# In[16]:


df_refine = pd.DataFrame(df)
amount_more = 0
amount_less = 0
for i in range(df_refine.shape[0]):
    if(df_refine.iloc[i]["Amount"] < 2500):
        amount_less += 1
    else:
        amount_more += 1
print(amount_more)
print(amount_less)
    


# In[17]:


percentage_less = (amount_less/df.shape[0])*100
percentage_less


# Hence, we observe that the 99.85% of transactions are amount to less than 2500. Let us see how many of these are fraud and others legitimate.
# 
# 

# In[18]:


fraud = 0
legitimate = 1
for i in range(df_refine.shape[0]):
    if(df_refine.iloc[i]["Amount"]<2500):
        if(df_refine.iloc[i]["Class"] == 0):
            legitimate += 1
        else:
            fraud+=1
print(fraud)
print(legitimate)


# In[19]:


df_refine = df[["Time", "Amount", "Class"]]
sns.pairplot(df_refine, hue= "Class", size=6)
plt.show()


# In[20]:


sns.FacetGrid(df_refine, hue="Class").map(sns.distplot,"Time").add_legend()
plt.show()


# # From the above distribution plot, it is clear that the fraudulent transactions are spread throughout the time period
# 
# - Modelling
# - Study the Feature Correlations of the given data
# - Plot a Heatmap
# - Run GridSearch on the Data
# - Fine Tune the Classifiers
# - Create Pipelines for evaluation

# In[21]:


plt.figure(figsize=(10,10))
df_corr = df.corr()
sns.heatmap(df_corr)


# In[22]:


# Create Train and Test Data in ratio 70:30
X = df.drop(labels='Class', axis=1) # Features
y = df.loc[:,'Class']               # Target Variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


# In[23]:


# Use Synthetic Minority Oversampling
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[24]:


from sklearn.feature_selection import mutual_info_classif
mutual_infos = pd.Series(data=mutual_info_classif(X_res, y_res, discrete_features=False, random_state=1), index=X_train.columns)


# In[25]:


mutual_infos.sort_values(ascending=False)


# In[30]:


sns.countplot(y_res)


# Hence, we can say that the most correlated features after resolving class imbalance using Synthetic Minority Oversampling are V14, V10, V4, V12 and V17.
# 
# # Evaluation
# We make use of AUC-ROC Score, Classification Report, Accuracy and F1-Score to evaluate the performance of the classifiers
# 
# ##### Method to compute the following:
# 1. Classification Report
# 2. F1-score
# 3. AUC-ROC score
# 4. Accuracy
# ##### Parameters:
# - y_test: The target variable test set
# - grid_clf: Grid classifier selected
# - X_test: Input Feature Test Set

# In[31]:


# Evaluation of Classifiers
def grid_eval(grid_clf):
    print("Best Score", grid_clf.best_score_)
    print("Best Parameter", grid_clf.best_params_)
    
def evaluation(y_test, grid_clf, X_test):
    y_pred = grid_clf.predict(X_test)
    print('CLASSIFICATION REPORT')
    print(classification_report(y_test, y_pred))
    
    print('AUC-ROC')
    print(roc_auc_score(y_test, y_pred))
      
    print('F1-Score')
    print(f1_score(y_test, y_pred))
    
    print('Accuracy')
    print(accuracy_score(y_test, y_pred))


# In[32]:


# The parameters of each classifier are different
# Hence, we do not make use of a single method and this is not to violate DRY Principles
# We set pipelines for each classifier unique with parameters
param_grid_sgd = [{
    'model__loss': ['log'],
    'model__penalty': ['l1', 'l2'],
    'model__alpha': np.logspace(start=-3, stop=3, num=20)
}, {
    'model__loss': ['hinge'],
    'model__alpha': np.logspace(start=-3, stop=3, num=20),
    'model__class_weight': [None, 'balanced']
}]

pipeline_sgd = Pipeline([
    ('scaler', StandardScaler(copy=False)),
    ('model', SGDClassifier(max_iter=1000, tol=1e-3, random_state=1, warm_start=True))
])

MCC_scorer = make_scorer(matthews_corrcoef)
grid_sgd = GridSearchCV(estimator=pipeline_sgd, param_grid=param_grid_sgd, scoring=MCC_scorer, n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)


grid_sgd.fit(X_res, y_res)


# In[33]:


grid_eval(grid_sgd)


# In[34]:


evaluation(y_test, grid_sgd, X_test)


# In[ ]:


pipeline_rf = Pipeline([
    ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
])
param_grid_rf = {'model__n_estimators': [75]}
grid_rf = GridSearchCV(estimator=pipeline_rf, param_grid=param_grid_rf, scoring=MCC_scorer, n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)
grid_rf.fit(X_res, y_res)


# In[ ]:


grid_eval(grid_rf)


# In[ ]:


evaluation(y_test, grid_rf, X_test)


# In[ ]:


pipeline_knn = Pipeline([
    ('model', KNeighborsClassifier(n_neighbors=5))
])
param_grid_knn = {'model__p': [2]}
grid_knn = GridSearchCV(estimator=pipeline_knn, param_grid=param_grid_knn, scoring=MCC_scorer, n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)
grid_knn.fit(X_res, y_res)


# In[ ]:


grid_eval(grid_knn)


# In[ ]:


evaluation(y_test, grid_knn, X_test)


# # Conclusion
# The K-Nearest Neighbors Classifier tuned with Grid Search with the best parameter being the Euclidean Distance (p=2) outperforms its counterparts to give a test accuracy of nearly 99.8% and a perfect F1-Score with minimal overfitting
# SMOTE overcomes overfitting by synthetically oversampling minority class labels and is successful to a great degree
# # Summary
# All Fraud Transactions occur for an amount below 2500. Thus, the bank can infer clearly that the fraud committers try to commit frauds of smaller amounts to avoid suspicion.
# The fraud transactions are equitable distributed throughout time and there is no clear relationship of time with commiting of fraud.
# The number of fraud transactions are very few comparted to legitimate transactions and it has to be balanced in order for a fair comparison to prevent the model from overfitting.

# In[ ]:





# In[ ]:





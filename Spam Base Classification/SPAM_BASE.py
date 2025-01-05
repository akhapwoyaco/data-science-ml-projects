#!/usr/bin/env python
# coding: utf-8
---
title: 'SPAM BASE CLASSIFICATION'
format: 
    docx: 
        toc: true
        section-numbers: true
        fig-width: 10
        fig-height: 9
execute: 
    echo: false
    warning: false
---
# # Introduction

# Email as a means of digital communication has been in use since the internet boom, both formally and informally. Companies use email as a means to provide further updates to their customers concerning products. However, emails have been misused, spam emails have been flooding into email users inbox with fake advertisements leading to phony sites, with links containing phishing links and pyramid schemes.
# 
# The need to sieve genuine from spam email guides our analysis, with the use of datasets on frequently occuring wordings from both spam and non spam email to build a spam classification model. The data is pre-processed, and used to train classification models that in turn are used on new data for classification purposes. The efficiency of the model is based upon the accuracy of correct classification of observations into either of the spam or non spam category.

# In[ ]:





# # Corpus Preparation

# In[ ]:





# ## Data

# The [Spambase Data Set](https://archive.ics.uci.edu/ml/datasets/Spambase) was obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/). The data is characterised by: 
# 
# 1. Number of observations: 4601
# 2. Number of features (attributes): 57

# In[1]:


import pandas as pd # data manipulation
import numpy as np # data manipulation
import re # clean the attribute names
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# ### Data Import

# The data was loaded as **spam_base_data** into the workspace from the **spambase.data** file, and attribute names assigned from the **spambase.names** data file, loaded as **attribute_names**.  

# In[ ]:





# In[2]:


# read spam data attribute names into local workspace
with open('spambase.names') as f:
 text = f.read()
attribute_names = re.findall(r'\n(\w*_?\W?):', text)

# read in data, with no headers and column names as previousy imported column names 
spam_base_data = pd.read_csv('spambase.data', header=None, names = attribute_names +['spam'])


# In[3]:


spam_base_data.head()


# In[ ]:





# In[4]:


print('The Spam Base data contains %d observations and %d columns, inclusive of 57 attributes and the column attribute that \n denotes whether the e-mail was considered spam (1) or not (0).' % spam_base_data.shape)


# In[ ]:





# ### Check Duplicated Observations

# In[5]:


#checking whether our data has duplicated values
count_of_duplicated = spam_base_data.duplicated().sum()
count_of_duplicated


# In[6]:


print("There are %d count of duplicated observations, of the %d observations." % (count_of_duplicated, spam_base_data.shape[0]))


# Duplicated observations will be dropped from the dataset.

# In[7]:


#dropping duplicated values
spam_base = spam_base_data.drop_duplicates(subset=None, keep = 'first', inplace = False)


# In[8]:


print('The Spam Base data after dropping duplicates contains %d observations and %d columns, inclusive of 57 attributes and the column attribute that \n denotes whether the e-mail was considered spam (1) or not (0).' % spam_base.shape)


# In[ ]:





# ### Check Missing Data Points

# In[9]:


#checking for null values
spam_base.isna().any().any()


# There are no missing data points in the data.

# In[ ]:





# ### Convert spam column to categorical

# The final column **spam**, that is nominal, taking up binary values is converted to categorical as it denotes whether the e-mail was considered spam (1) or not (0).

# In[10]:


# convert cst column to categorical
spam_base.spam = pd.Categorical(
       spam_base.spam, categories = [0, 1], ordered = True)
print(spam_base['spam'].unique())
print(spam_base['spam'].value_counts())


# In[ ]:





# ### Describe Data

# In[11]:


#checking the description our the dataset we have for analysis
spam_base.describe()


# In[ ]:





# ### EDA

# #### Correlation Plot

# Correlation analysis provides the linear relation between attributes. Majority of the variables are weakly correlated.

# In[12]:


#checking between our datasets
correlation_mat = spam_base.corr()


# In[13]:


# get the highy correlated variables
corr_mat1 = correlation_mat.stack().reset_index() #convert to long format
corr_mat1.columns = ['Var1', 'Var2', 'Cor'] # set column names
corr_mat1[corr_mat1.Cor != 1].drop_duplicates(
    # remove duplicated corr values
    subset = ['Cor'], keep = 'first', inplace = False).sort_values(
    # sort by corr
    by = 'Cor', ascending = False).head(20) # get top 20 highly correlated


# The frequency across words 85, 67, 650, 857, 415, direct, telnet and technology may be attributed to presence of phone numbers or addresses associated to given labs providing certain technological products, as such given the lucrativeness of blue chip companies, theor associated key words may have been frequently used in spam or no spam emails.

# In[ ]:





# In[14]:


# corr plot
plt.figure(figsize=(16, 12))
ax = sns.heatmap(
    correlation_mat, xticklabels = correlation_mat.columns, 
    yticklabels = correlation_mat.columns, 
    cmap = sns.diverging_palette(220, 10, as_cmap = True), 
    square=True, linewidths=.1)
ax.set(title="Correlation heatmap");


# In[ ]:





# #### Average Word, Char and Capital Run Distributions

# Word Frequency

# In[15]:


fig, axes = plt.subplots(1, 1)
fig.set_figheight(40)
fig.set_figwidth(10)
average_word_char_cap = pd.melt(
    spam_base[spam_base.columns[spam_base.columns.to_series().str.contains(r'word|spam')]], 
    id_vars = ['spam'])
sns.boxplot(y = 'variable', x = 'value', hue = 'spam', data = average_word_char_cap, 
            orient = 'h')
plt.show()


# Character Frequency

# In[16]:


char_spam_data = spam_base[spam_base.columns[spam_base.columns.to_series().str.contains(r'char|spam')]]
fig, axes = plt.subplots(6, 1)
fig.set_figheight(20)
fig.set_figwidth(10)
sns.boxplot(x = 'char_freq_;', hue = 'spam', 
                 y = 'spam', data = char_spam_data, orient = 'h',
                 ax = axes[0]).set(title = 'char_freq_;', xlabel = '')
sns.boxplot(x = 'char_freq_(', hue = 'spam',
                 y = 'spam', data = char_spam_data, orient = 'h', 
                 ax = axes[1]).set(title = 'char_freq_(', xlabel = '')
sns.boxplot(x = 'char_freq_[', hue = 'spam',
                 y = 'spam', data = char_spam_data, orient = 'h', 
                 ax = axes[2]).set(xlabel = '', title = 'char_freq_[')
sns.boxplot(x = 'char_freq_!', hue = 'spam',
                 y = 'spam', data = char_spam_data, orient = 'h', 
                 ax = axes[3]).set(title = 'char_freq_!', xlabel = '')
sns.boxplot(x = 'char_freq_$', hue = 'spam',
                 y = 'spam', data = char_spam_data, orient = 'h', 
                 ax = axes[4]).set(xlabel = '', title = 'char_freq_$')
sns.boxplot(x = 'char_freq_#', hue = 'spam',
                 y = 'spam', data = char_spam_data, orient = 'h', 
                 ax = axes[5]).set(title = 'char_freq_#', xlabel = '')
plt.show()


# Capita Run Length

# In[17]:


capital_spam_data = spam_base[spam_base.columns[spam_base.columns.to_series().str.contains(r'cap|spam')]]
fig, axes = plt.subplots(3, 1)
fig.set_figheight(10)
fig.set_figwidth(10)
sns.boxplot(x = 'capital_run_length_average', hue = 'spam', 
                 y = 'spam', data = capital_spam_data, orient = 'h',
                 ax = axes[0]).set(title = 'capital_run_length_average', xlabel = '')
sns.boxplot(x = 'capital_run_length_longest', hue = 'spam',
                 y = 'spam', data = capital_spam_data, orient = 'h', 
                 ax = axes[1]).set(title = 'capital_run_length_longest', xlabel = '')
sns.boxplot(x = 'capital_run_length_total', hue = 'spam',
                 y = 'spam', data = capital_spam_data, orient = 'h', 
                 ax = axes[2]).set(xlabel = '', title = 'capital_run_length_total')
plt.show()


# There appears to be extreme values within our data variables, majorly among the spma = 1 categories.

# In[ ]:





# # Solution Methodology

# Data modelling phase will involve:
#     
# 1. Extended corpus preparation through pre-processing of the independent attribute variables: The process will involve the scaling of all of numerical variables with the sole aim of standardizing the variables that vary with regard to units of measurement.
# 2. The data is split into train (80%) and test (20%), as a percentage of whole data, with indices of observation in either of the data sets selected randomly. The train data (80%) is used for training the model while 20% to test model perfomance through evaluation metrics, in our case accuracy.
# 3. The binary nature of the response variable guides the selection of the models of interest: K Nearest Neighbors (K-nn) and Decision Trees.
# 4. Evaluation criteria and model evaluation encompasses the training of models using hypertuned parameters on training data sets and comparing the perfomance of the models on new dataset, the test data.

# In[ ]:





# ## Data Preprocessing

# In[18]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# In[19]:


# Standardardize the data
spam_base_model_dataset = spam_base.copy()
spam_base_model_dataset.iloc[:, 0:57] = MinMaxScaler().fit_transform(spam_base_model_dataset.iloc[:, 0:57])


# ## Data Split

# In[20]:


# set random seed for reproducibility
np.random.seed(12345)
train_index = np.random.rand(len(spam_base_model_dataset)) < 0.8
# split the dataset into train and test sets
train_data = spam_base_model_dataset.iloc[train_index,:]
test_data = spam_base_model_dataset.iloc[~train_index,:]
# get predictors of train and test
X_train = train_data.copy()
X_test = test_data.copy()
# get response train and test
y_train = train_data.copy()
y_test = test_data.copy()
# drop response from predictor variables
X_train = X_train.drop(['spam'],axis=1)
X_test = X_test.drop(['spam'],axis=1)
# get responce train and test
y_train = y_train['spam']
y_test = y_test['spam']
print('Training Data Set Predictor shape: ', X_train.shape, 'Response Variable:', y_train.shape)
print('Testing Data Set Predictor shape: ', X_test.shape, 'Response Variable:' , y_test.shape)


# In[ ]:





# ## Modelling

# In[21]:


from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix


# In[ ]:





# ### K Neares Neighbors (knn)

# In[22]:


# Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# model initialisation
knn = KNeighborsClassifier()


# The KNN model is initialised, by the _KNeighborsClassifier()_ method of scikit learn module. Hyperparameter tuning or optimization is undertaken to choosing a set of optimal hyperparameters for the KNN learning on parameters: 
# 
# 1. The number of neighbors to use: Selected as 2 given the binary nature of outcome spam.
# 2. The weight function to be used in prediction, with selection from the 'uniform', for equal weighing of points within a neighborhood and 'distance', for weighing based on inverse of distance within neighborhood.
# 3. The metric to be used for computation of the distance: 'minkowski', 'euclidean','manhattan'.
# 
# Other values passed to the other parameters are learned over the course of training the model. A KNN model is then fit, with the best parameters.

# In[23]:


# Hypertuning parameters
grid_params_knn = { 'n_neighbors' : list(range(2, 20)),
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
gscv_knn = GridSearchCV(KNeighborsClassifier(), grid_params_knn, verbose = 1, cv=4, n_jobs = -1)


# In[24]:


# fit knn model on train data
gsearchcv_knn = gscv_knn.fit(X_train, y_train)
# hyperparameters from best score
gsearchcv_knn.best_params_


# #### Evaluation Criteria

# The KNN classification model is fit, with the best parameters.

# In[25]:


# best hyperparameters suppied as fitting parameters
knncv_knn = KNeighborsClassifier(**gsearchcv_knn.best_params_)
knncv_knn.fit(X_train, y_train)


# #### Model Evaluation

# In[ ]:





# In[26]:


# prediction
y_hat_knn = knncv_knn.predict(X_train)


# In[27]:


accuracy_knn_train = accuracy_score(y_train, y_hat_knn)
print('knn: Training Data Set Accuracy: ', accuracy_knn_train)


# In[28]:


# Confusion Matrix train data
cm_knn = confusion_matrix(y_train, y_hat_knn)
print(cm_knn)


# In[29]:


from sklearn.metrics import classification_report
print(classification_report(y_train, y_hat_knn, target_names = ['0', '1']))


# In[30]:


y_knn = knncv_knn.predict(X_test)
accuracy_knn_test = accuracy_score(y_test, y_knn)
print('knn: Testing set accuracy: ', accuracy_knn_test)


# In[31]:


print(classification_report(y_test, y_knn, target_names = ['0', '1']))


# ### Decision Tree

# In[32]:


from sklearn.tree import DecisionTreeClassifier


# The Decision Tree model is initialised, by the _DecisionTreeClassifier()_ method of scikit learn module. Hyperparameter tuning or optimization is undertaken to choosing a set of optimal hyperparameters for the Decision Tree learning on parameters: 
# 
# 1. The criterio argument supplied with "gini" or "entropy", that measures the quality of splits.
# 2. The maximum depth of the decision tree.
# 3. The maximum number of leaf nodes to grow a tree.
# 
# Other values passed to the other parameters are learned over the course of training the model. A Decision Tree Classifier model is then fit, with the best parameters.

# In[33]:


# Hyperparameter tuning
param_decision_tree = [{'criterion':["gini", "entropy"],
          'max_depth': list(range(2, 20)),'max_leaf_nodes': list(range(2, 50))}]
grid_search_decision_tree = GridSearchCV(
    DecisionTreeClassifier(random_state=12345), param_decision_tree, verbose=1, cv=4, 
    n_jobs=-1, scoring = "accuracy")


# #### Evaluation Criteria

# In[34]:


# Decision Trees fitting and prediction
grid_search_decision_tree.fit(X_train, y_train)
# best hyperparameter
grid_search_decision_tree.best_estimator_


# The Decision Tree classification model is fit, with the best parameters.

# #### Model Evaluation

# In[35]:


y_pred_dec_tree = grid_search_decision_tree.predict(X_train)
accuracy_dec_tree_train = accuracy_score(y_train, y_pred_dec_tree)
print('Decision Tree: Training set accuracy: ', accuracy_dec_tree_train)


# In[36]:


cm_dt = confusion_matrix(y_train, y_pred_dec_tree)
print(cm_dt)


# In[37]:


print(classification_report(y_train, y_pred_dec_tree, target_names = ['0','1']))


# In[38]:


y_dec_tree = grid_search_decision_tree.predict(X_test)
accuracy_dec_tree_test = accuracy_score(y_test, y_dec_tree)
print('Decision Tree: Test set accuracy: ', accuracy_dec_tree_test)


# In[39]:


print(classification_report(y_test, y_dec_tree, target_names = ['0','1']))


# In[ ]:





# # Experimental Results and Model Comparison

# Model comparison are based on the accuracy, the rate at which a model correctly classifies the spam email as spam and non-spam as non-spam. 
# 
# 1. On the train data, the KNN model performs better as compared to the Decision Tree classifier, with both performing fairly better at 99% and 94.5% rates respectively.
# 2. On the test data, the Decision Tree Classifier model performs slightly better compared to the KNN Classifier, with both performing fairly better at 93% and 91% rates respectively.
# 3. Missclassification rates are at a low at 8.6% and 6.9% for KNN and Decision Tree Classifier respectively on the test data.
# 4. Missclassification rates are at a low at 0% and 0.6% for KNN and Decision Tree Classifier respectively on the train data.
# 5. The ROC AUC measures the separation existing between classes of the binary classifier, with AUC values on test data for KNN and Decision Tree classifiers at 0.96 and 0.97 respectively.
# 
# 
# The KNN model outperforms the Decision Tree for train data set, but slightly lower on test data set. The AUC value of Decision Classifier is higher, but both models are better of to distinguish the positive and negative classes and as such is classifying the email data as spam or not spam.

# In[40]:


from sklearn.metrics import roc_auc_score

y_knn_proba = knncv_knn.predict_proba(X_test)[:,1]
y_dec_tree_proba = grid_search_decision_tree.predict_proba(X_test)[:,1]


# In[41]:


from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
def plot_roc_curve(y_real, y_pred, title = ''):
    fpr, tpr, _ = roc_curve(y_real, y_pred)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    roc_display.figure_.set_size_inches(5,5)
    plt.plot([0, 1], [0, 1], color = 'g')
    plt.title(title)
# Plots the ROC curve using the sklearn methods - Good plot
plot_roc_curve(y_test, y_knn_proba, title = 'KNN')
plot_roc_curve(y_test, y_dec_tree_proba, title = 'Decision Tree')


# In[42]:


pd.DataFrame(
    [
        ['kNN', 
         accuracy_knn_train, accuracy_knn_test, 
         1-accuracy_knn_train, 1-accuracy_knn_test,  
         roc_auc_score(y_test, y_knn_proba)],
        ['Decision Tree', 
         accuracy_dec_tree_train, accuracy_dec_tree_test, 
         1-accuracy_dec_tree_train, 1-accuracy_dec_tree_test, 
         roc_auc_score(y_test, y_dec_tree_proba)]
    ], columns = ['Model',
                  'Train Set: Accuracy','Test Set: Accuracy', 
                  'Train Set: misclassification_error',
                  'Test Set: misclassification_error',
                 'Test Data: Area Under the Curve (auc_score)'])


# In[ ]:





# # Limitations and Possible Enhancements

# Limitations
# 
# 1. The data is based on frequency of words, characters and capitalization, with certain words being more prone to good non-spam emails being mimicked within the spam emails to escape the classification model scrutiny in real case scenarios, which in effect will have a impact in the evaluation of the model.
# 
# 
# Possible Enhancements
# 
# 1. The analysis is more of inclined towards ensuring that more and more good emails are flagged as good, while spam emails are classified as spam. However, over time, semantics change and spam email words and frequencies evolve, as such it is imperative when dealing with spam classification to take into consideration changes in semantics and their inclusion into spam classification models.
# 2. The hypertuning parameters used within our analysis are basic, as such extension to more data centric parameters would improve perfomance interms of better classification.

# In[ ]:





# In[ ]:





# In[ ]:





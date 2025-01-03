#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Predictive analytics is undertaken on the data provided after undertaking a series of data understanding, pre-processing and model selection. Data understanding is undertaken under data preparation to get a clear picture with regard to statistical data summaries, missing entries in the data, duplicate row data. The imperfections noted are handled through the pre-processing of the data and put in consideration in the assessment of variables viable for predictive task. The main predictive task to be undertaken is classification, with the dependent variable a multi-label. Three classification models, the k-NN, Naïve Bayes, and the Decision Trees are used to train the data, fine-tuned to attain the best parameters to attain models better suited at prediction tasks on the train data. The two best models based on their performances are used to predict the test data provided. All steps are backed by concrete explanations with regard to the choice.

# In[ ]:





# # Methodology

# In[ ]:





# The data preparation section handles the identifying and removal of irrelevant attributes, detecting and handling of missing entries, duplicates in the instances and attributes, the selection of suitable data types for attributes and data transformation whenever required.

# In[ ]:





# ## Data

# ### Data Description

# The data stored within a SQLite file is imported into the workspace and the first few rows explored.

# In[1]:


# Import Libraries


# In[2]:


import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# In[3]:


import sqlite3
import pandas as pd
# Create your connection.
cnx = sqlite3.connect("Assignment2021.sqlite")
dataset = pd.read_sql_query("SELECT * from data", cnx)
cnx.close()


# In[4]:


# First 5 rows
dataset.head()


# In[ ]:





# The dataset comprises of 1200 observations with 32 attributes. The Table 1 Data Types and Summary Statistics provides the summary statistics of numeric data columns, for which we can note the diverse ranges on the data variables and the count of non-null values within all columns.

# A better understanding of the data structure, description with regard to the variable types and summary statistics are undertaken.The data comprices of numeric and categorical data types.

# 

# In[5]:


dataset.shape


# In[6]:


dataset.info()


# Missing data in the class column is due to the said observations being part of the test data used in the predictive tasks. The Att00 has 9 missing data points and Att09 has 581 missing data points. The missing data points will be handled in the pre-processing phase. A check on the duplicated observations within the data shows that there are no duplicated observations in the data.

# In[7]:


dataset.describe()


# The count of missing data per feature are as below.

# In[8]:


#Missing Data Points
dataset.isnull().sum()


# Unique values per feature

# In[9]:


dataset.nunique()


# Check duplicated observations within the data. There are no duplicated observations in the data.

# In[10]:


duplicate_values = dataset.duplicated()
print(dataset[duplicate_values])


# Categorical column count

# In[11]:


dataset['Att01'].value_counts()


# In[12]:


dataset['Att08'].value_counts()


# In[13]:


dataset['Att29'].value_counts()


# In[14]:


dataset['class'].value_counts()


# In[ ]:





# ### Correlation

# In[ ]:





# Check the correlation between features.

# Correlation analysis is conducted on the data to ascertain the linear relation between the attributes and is depicted in the Figure Correlation Plot where there was a fairly a strong correlation between the variables:

# In[15]:


dataset_1 = dataset.copy()


# In[16]:


# numeric columns
columns_num = dataset_1.select_dtypes(include=np.number).columns.tolist()[:-1]
new_data = dataset_1[columns_num[1:]].copy()


# In[17]:


from sklearn import preprocessing

scaler_var = preprocessing.MinMaxScaler()
new_data_norm = scaler_var.fit_transform(new_data)
new_data_norm = pd.DataFrame(new_data_norm)
new_data_norm.columns = new_data.columns


# In[18]:


pd.DataFrame(new_data_norm).corr().style.background_gradient(cmap='coolwarm')


# In[19]:


from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# Compute the correlation matrix
corr = pd.DataFrame(new_data_norm).corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 11))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns_plot_corr = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.7, center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .99})
ax.figure.savefig("attribute_corr.png")
sns_plot_corr


# In[ ]:





# In[20]:


# get the distribution of the target variable
# Set up the matplotlib figure
f, ax = plt.subplots()
sns_plot_class = sns.countplot(x="class", data = dataset_1)
ax.figure.savefig("sns_plot_class.png")
sns_plot_class


# In[ ]:





# ### Missing Data and Data Transformation

# Data transformation in preparation for the classification tasks involved creating of dummy variables for the categorical variables Att01, Att08 and Att29. The creating of dummy variables was based on the fact that categorical data requires transformation to numeric for ease of handling by the classification models. The columns whose missing values percentage are way above 40% are excluded from the data, for which drops feature Att09. Similarly, missing data points in Att00 are filled with mean of the column due to the negligible missing 9 data points as opposed to Att09’s 581 missing data points. 
# Transformation of the data through scaling was undertaken for the k-NN and Decision Tree models, while the Naïve Bayes model was carried out with the non-transformed data. Naïve Bayes classifier transformation of data was not carried out due to the need for both positive and negative values for the model to be able to perform better on the training data.

# The data was scaled, given the varying ranges and skewness observed in the data description section, and also based on the assumption that the attributes of the data may be in units of varying scales. The normalization of each of the numeric attribute with the exception of the binary attribute was undertaken with the help of the min-max scaler whose formula is as shown below:

# X_std  =((X-X.min⁡(axis=0) ))/((X.max⁡(axis=0)-X.min⁡(axis=0) ) )

# X_scaled=X_std × (max-min)+min

# Where the min, max = feature_range.

# ## Data Classification

# ### Train-Test Split

# The data was split into train and test data, with the train data covering the first 1000 instance and the test data covering the last 200 instances. The split was conducted manually through indexing. 

# Classification was conducted after the selection of the train and test data sets. The models were fine-tuned to increase model accuracy. Classifiers used in the model are the k-NN, Naive Bayes, and Decision Trees. Cross validation was incorporated into the hyper parameter tuning and modelling to improve on the training accuracy of the models.

# ### KNN

# In[21]:


from sklearn import tree
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
####
from sklearn.model_selection import cross_val_predict
import sklearn.model_selection as model_selection


# The k-nearest neighbors (KNN) classifier model takes a positive integer K, for which it is to identify K points closest to a test observation x_0 after which it estimates the conditional probability of a given class within the list of classes based on the frequency of the class and takes advantage of Bayes rule to classify the x_0 observation to a class that attains the highest conditional probability.
# Pr⁡〖(Y=j| X=x_0 )= 1/K _(i ∈N_0 ) I (y_i=j)〗
# Where N_0 are K points closest to x_0 and j is a class within the dependent variables.
# 

# ##### Hyperparameter Tuning

# The Exhaustive Grid Search technique for hyperparameter optimization was used. The exhaustive grid search takes in various hyperparameters that are desired and tries every single possible combination within the hyperparameters provided while also undertaking cross-validations on the data. The output of the exhaustive grid search is the best hyperparameter values to input into the model. However, in the interest of time, the most basic input are included in the hyperparameters of KNN.
# 
# Parameter used herein for the KNN are:
# 1.	n_neighbors: The best k based on the computed values.
# 2.	weights: The parameter checks whether the addition of weights to the instances improves model accuracy or not, with the 'uniform' assigning no weights, while 'distance' weighs points based on the inverse of the distance between them which implies that the nearer points will have more weight compared to those points farther.
# 3.	metric: is the metric used in calculating the similarity.
# 

# In[22]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Feature Selection
dataset_2 = pd.get_dummies(dataset,columns=["Att01","Att08","Att29"])
dataset_2.shape
dataset_2 = dataset_2.loc[:, dataset_2.isnull().mean() < .4]
#dataset_2.shape
# Fill na with mean Attr00
dataset_2['Att00'] = dataset_2['Att00'].fillna(dataset_2['Att00'].mean())
# Standardize
dataset_2.iloc[:, 1:27] = MinMaxScaler().fit_transform(dataset_2.iloc[:, 1:27])


# In[23]:


# split the dataset into train and test sets
train_data = dataset_2.iloc[0:1000,:]
test_data = dataset_2.iloc[1000:1200,:]
train_data = train_data.drop(['index'], axis=1)
test_data = test_data.drop(['index'], axis=1)
#
X_train = train_data.copy()
X_test = test_data.copy()
#
y_train = train_data.copy()
y_test = test_data.copy()
#
X_train = X_train.drop(['class'],axis=1)
X_test = X_test.drop(['class'],axis=1)

y_train = y_train['class']
y_test = y_test['class']
#
print(train_data.shape)
print(test_data.shape)
print('Training set shape: ', X_train.shape, y_train.shape)
print('Testing set shape: ', X_test.shape, y_test.shape)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()


# Hyperparameter Tuning

# In[25]:


grid_params_knn = { 'n_neighbors' : [2,3,5,7,9],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}


# In[26]:


gs_knn = GridSearchCV(KNeighborsClassifier(), grid_params_knn, verbose = 1, cv=3, n_jobs = -1)


# In[27]:


# fit the model on our train set
g_res_knn = gs_knn.fit(X_train, y_train)


# In[28]:


# get the hyperparameters with the best score
g_res_knn.best_params_


# The hyper-parameters that provide the best models were 5 n_neighbors, manhattan metric and distance weights. The model provided an accuracy of 100% on the training data set.

# In[ ]:





# In[29]:


# use the best hyperparameters
knn_knn = KNeighborsClassifier(**g_res_knn.best_params_)
#n_neighbors = 5, weights = 'distance',algorithm = 'brute',metric = 'manhattan')
knn_knn.fit(X_train, y_train)


# In[30]:


# get a prediction
y_hat_knn = knn_knn.predict(X_train)


# In[ ]:





# In[31]:


accuracy_knn_1 = metrics.accuracy_score(y_train, y_hat_knn)
print('knn: Training set accuracy: ', accuracy_knn_1)


# In[32]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_train, y_hat_knn)
print(cm_knn)


# In[33]:


#finding accuracy from the confusion matrix.
a_knn = cm_knn.shape
corrPred = 0
falsePred = 0

for row in range(a_knn[0]):
    for c in range(a_knn[1]):
        if row == c:
            corrPred +=cm_knn[row,c]
        else:
            falsePred += cm_knn[row,c]
print('Correct predictions: ', corrPred)
print('False predictions', falsePred)
print ('\n\nAccuracy of the knn Clasification is: ', corrPred/(cm_knn.sum()))


# In[34]:


from sklearn.metrics import classification_report
print(classification_report(y_train, y_hat_knn, target_names = ['0','1','2']))


# In[ ]:





# Prediction based on KNN

# In[35]:


y_knn = knn_knn.predict(X_test)


# In[ ]:





# In[ ]:





# ### Decision Trees

# Decision trees predict that each instance of test data falls into the most commonly occurring class of the training instances in the region to which it belongs. Either of the Gini index or the entropy, both of which are sensitive to the purity of the node, simply referred to as the classification error rate can be used to offer evaluation of a splits within the decision trees. For the purposes of prediction accuracy, the classification rate is used in the pruning of the classification trees.

# In[ ]:





# In[36]:


from sklearn import tree
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import sklearn.model_selection as model_selection


# In[37]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
params_dt = [{'criterion':["gini", "entropy"],
          'max_depth': list(range(2, 20)),'max_leaf_nodes': list(range(2, 50))}]
grid_search_cv_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42), params_dt, verbose=1, cv=3, 
    n_jobs=-1, scoring = "accuracy")
grid_search_cv_dt.fit(X_train, y_train)


# In[38]:


grid_search_cv_dt.best_estimator_


# The best hyper parameters for the Decision Tree model were at max_depth of 9, max_leaf of 22 and random state of 42. 

# In[39]:


#Decision Trees
dtree = tree.DecisionTreeClassifier(max_depth=9, max_leaf_nodes=22, random_state=42)
dtree.fit(X_train, y_train)
Y_pred_dt = cross_val_predict(dtree, X_train, y_train, cv = 3)


# In[40]:


accur_dt = metrics.accuracy_score(y_train, Y_pred_dt)
print('Decision Tree: Training set accuracy: ', accur_dt)


# The model provided an accuracy of 70.9% on the training data set.

# In[41]:


cm_dt = confusion_matrix(y_train, Y_pred_dt)
print(cm_dt)


# In[42]:


#finding accuracy from the confusion matrix.
a_dt = cm_dt.shape
corrPred = 0
falsePred = 0

for row in range(a_dt[0]):
    for c in range(a_dt[1]):
        if row == c:
            corrPred +=cm_dt[row,c]
        else:
            falsePred += cm_dt[row,c]
print('Correct predictions: ', corrPred)
print('False predictions', falsePred)
print ('\n\nAccuracy of the Decision Tree Clasification is: ', corrPred/(cm_dt.sum()))


# In[43]:


from sklearn.metrics import classification_report, f1_score
print(classification_report(y_train, Y_pred_dt, target_names = ['0','1','2']))


# In[ ]:





# Prediction based on Decision tree on the test data

# In[44]:


y_dtree = dtree.predict(X_test)


# In[ ]:





# In[ ]:





# ### NAIVE

# The naive Bayes classifier works by finding instances of similar attribute class, learns of the classes in the dependent categorical variable for each of the instances and builds upon the most frequent to predict class for new test data. The classifier gives the probability of belonging to any of the available classes after which cutoff probability is used in the assigning of observations to new test data instances. There isn't a hyper-parameter to tune with the Naive Bayes as such, no grid search over was undertaken.

# In[ ]:





# In[45]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Feature Selection
dataset_2 = pd.get_dummies(dataset,columns=["Att01","Att08","Att29"])
dataset_2.shape
dataset_2 = dataset_2.loc[:, dataset_2.isnull().mean() < .4]
dataset_2.shape
# Fill na with mean
dataset_2['Att00'] = dataset_2['Att00'].fillna(dataset_2['Att00'].mean())
# No Standardization
#dataset_2.iloc[:, 1:27] = MinMaxScaler().fit_transform(dataset_2.iloc[:, 1:27])


# In[46]:


# split the dataset into train and test sets
train_data = dataset_2.iloc[0:1000,:]
test_data = dataset_2.iloc[1000:1200,:]
train_data = train_data.drop(['index'], axis=1)
test_data = test_data.drop(['index'], axis=1)
#
X_train = train_data.copy()
X_test = test_data.copy()
#
y_train = train_data.copy()
y_test = test_data.copy()
#
X_train = X_train.drop(['class'],axis=1)
X_test = X_test.drop(['class'],axis=1)

y_train = y_train['class']
y_test = y_test['class']
#
print(train_data.shape)
print(test_data.shape)
print('Training set shape: ', X_train.shape, y_train.shape)
print('Testing set shape: ', X_test.shape, y_test.shape)


# There isn't a hyper-parameter to tune with the naive bayes as such, no grid search over was undertaken.

# In[47]:


# Fitting Naive Bayes Classification to the Training set with linear kernel
from sklearn.naive_bayes import GaussianNB
nvclassifier = GaussianNB()
nvclassifier.fit(X_train, y_train)


# In[48]:


# Predicting the Test set results
y_pred_nv = nvclassifier.predict(X_train)


# In[49]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nv = confusion_matrix(y_train, y_pred_nv)
print(cm_nv)


# In[50]:


#finding accuracy from the confusion matrix.
a_nv = cm_nv.shape
corrPred = 0
falsePred = 0

for row in range(a_nv[0]):
    for c in range(a_nv[1]):
        if row == c:
            corrPred +=cm_nv[row,c]
        else:
            falsePred += cm_nv[row,c]
print('Correct predictions: ', corrPred)
print('False predictions', falsePred)
print ('\n\nAccuracy of the Naive Bayes Clasification is: ', corrPred/(cm_nv.sum()))


# The model provided an accuracy of 70.4% on the training data set.

# In[51]:


accur_nv =  metrics.accuracy_score(y_train, y_pred_nv)
print('NB: Training set accuracy: ',accur_nv)


# In[52]:


from sklearn.metrics import classification_report
print(classification_report(y_train, y_pred_nv, target_names = ['0','1','2']))


# In[ ]:





# Prediction based on Naive Bayes on the test data

# In[53]:


y_nvclassifier = nvclassifier.predict(X_test)


# In[ ]:





# # Results and Conclusion

# ## Model Comparison and Prediction

# Model comparisons were made based on the accuracy, for which the best 2 models were selected for the purpose of predicting class labels for the 200 instances in the test data. The knn training set accuracy was at 100%, Decision Tree Classification at 70.9% and Naïve Bayes training set accuracy at 70.4%.

# ## Prediction

# The models selected for the prediction task were the knn and Decision Tree Classification. The predictions were derived from the models attained in the classification section applied on the test data created in the data preparation stage. Predictions were made on the previously partitioned test data of the dataset based on models used to fit the train data with the tuned parameters.

# The predicted class labels from the best 2 models, the KNN and the Decision Tree that were selected solely based on the accuracy score from fitting on the training data. The predicted variables were converted into type integer, concatenated into a data frame. The results saved in the Answers.sqlite file with the three columns, ID, Predict1 of the KNN model and Predict2 of the Decision Tree model.

# Further assessment can be made to include more parameters in the tuning phase as well as extend comparisons of model effectiveness beyond only accuracy measures.

# In[54]:


# Based on accuracy
# k-NN
# Decision Tree


# In[55]:


all_predictions = pd.DataFrame({'pred1':y_knn,'pred2':y_nvclassifier,'pred3':y_dtree})


# In[56]:


index = dataset_2.iloc[1000:1200,0]
Answers = pd.DataFrame({'ID': index, 'Predict1':y_knn.astype(int), 'Predict2': y_dtree.astype(int)})
Answers.to_csv('Answers.csv', index=False)


# In[57]:


Answers.head(10)


# In[58]:


# save to Answer.sqlite
from sqlalchemy import create_engine
disk_engine = create_engine('sqlite:///Answers.sqlite')
Answers.to_sql('Answers', disk_engine, if_exists='fail')


# In[ ]:





# # References

# 1. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2017). An introduction to statistical learning: With applications in R. Springer.
# 2. Sklearn.preprocessing.MinMaxScaler. scikit. (n.d.). Retrieved October 16, 2021, from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html.

# In[ ]:





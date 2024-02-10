#!/usr/bin/env python
# coding: utf-8

# # PRICE PREDICTION FOR GROCERY

# ![image.png](attachment:image.png)

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv("aisles.csv")
data1=pd.read_csv("departments.csv")
data2=pd.read_csv("order_products__prior.csv")
data3=pd.read_csv("orders.csv")
data4=pd.read_csv("products.csv")
data5=pd.read_csv("order_products__train.csv")


# In[3]:


x=data.join(data1)


# In[4]:


data


# In[5]:


data1


# In[6]:


data2


# In[7]:


data3


# In[8]:


data4


# In[9]:


data5


# # EDA

# In[10]:


df1 = pd.merge(data, data4, how='inner', on='aisle_id')
df2 = pd.merge(df1, data1, how='inner', on='department_id')
df3 = pd.merge(data5, df2, how='inner', on='product_id')
df4 = pd.merge(data3, df3, how='inner', on='order_id')


# In[11]:


df4


# In[12]:


df = df4.drop(['aisle_id','department_id','eval_set','order_number'],axis=1)
display(df.head())
print('\n\033[1mInference:\033[0m The Datset consists of {} features & {} samples.'.format(df.shape[1], df.shape[0]))


# In[13]:


pd.DataFrame(df)


# In[14]:


df.info()


# In[15]:


df.columns


# In[16]:


df.shape


# In[17]:


df.nunique().sort_values()


# In[18]:


df.describe()


# In[19]:


df.shape


# ## VISUALIZATION

# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[21]:


type(df)


# In[22]:


print(type(df['department']))


# In[23]:


df['department'] = pd.Series(df['department'])


# In[24]:


print(df['department'].unique())


# In[25]:


import matplotlib.pyplot as plt
df['department'].value_counts().plot(kind='bar')
plt.show()


# In[26]:


g=df['aisle'].loc[:20]
pd.DataFrame(g)


# In[27]:


import matplotlib.pyplot as plt
g.value_counts().plot(kind='bar')
plt.show()



# In[28]:


prod=df['product_name'].loc[:20]
pd.DataFrame(prod)


# In[29]:


import matplotlib.pyplot as plt
prod.value_counts().plot(kind='bar')
plt.show()



# In[30]:


# sns.countplot(df.order_hour_of_day)#, order=df.aisle.value_counts().index[:20])
# plt.title('Online Shopping Horly-Frequency')
# plt.xticks(rotation=90)
# plt.show()


# In[31]:


import matplotlib.pyplot as plt
df['order_hour_of_day'].value_counts().plot(kind='bar')
plt.show()



# In[32]:


df.isnull().sum()


# In[37]:


df.duplicated().sum()


# In[33]:


dummies_df = pd.get_dummies(data=df, prefix=['Day','Hour'], columns=['order_dow','order_hour_of_day'], drop_first=True)
dummies_df.head()


# In[ ]:


# #Final Dataset size after performing Preprocessing

# plt.title('Final Dataset Samples')
# plt.pie([df1.shape[0], df1.shape[0]-df.shape[0]], radius = 1, shadow=True,
#         labels=['Retained','Dropped'], counterclock=False, autopct='%1.1f%%', pctdistance=0.9, explode=[0,0])
# plt.pie([df.shape[0]], labels=['100%'], labeldistance=-0, radius=0.78, shadow=True, colors=['powderblue'])
# plt.show()

# print('\n\033[1mInference:\033[0mThe final dataset after cleanup has {} samples & {} rows.'.format(df.shape[0], df.shape[1]))


# In[34]:


user_prod_df = dummies_df.groupby(['user_id','product_id']).agg({'order_id':'nunique',
                                                                 'days_since_prior_order':'mean',
                                                                 'reordered':'max',
                                                                 'Day_1':'sum',
                                                                 'Day_2':'sum',
                                                                 'Day_3':'sum',
                                                                 'Day_4':'sum',
                                                                 'Day_5':'sum',
                                                                 'Day_6':'sum',
                                                                 'Hour_1':'sum',
                                                                 'Hour_2':'sum',
                                                                 'Hour_3':'sum',
                                                                 'Hour_4':'sum',
                                                                 'Hour_5':'sum',
                                                                 'Hour_6':'sum',
                                                                 'Hour_7':'sum',
                                                                 'Hour_8':'sum',
                                                                 'Hour_9':'sum',
                                                                 'Hour_10':'sum',
                                                                 'Hour_11':'sum',
                                                                 'Hour_12':'sum',
                                                                 'Hour_13':'sum',
                                                                 'Hour_14':'sum',
                                                                 'Hour_15':'sum',
                                                                 'Hour_16':'sum',
                                                                 'Hour_17':'sum',
                                                                 'Hour_18':'sum',
                                                                 'Hour_19':'sum',
                                                                 'Hour_20':'sum',
                                                                 'Hour_21':'sum',
                                                                 'Hour_22':'sum',
                                                                 'Hour_23':'sum'
                                                                }).reset_index()
user_prod_df.head()


# In[35]:


user_purchase_df = dummies_df.groupby(['user_id']).agg({         'order_id':'nunique',
                                                                 'product_id': 'nunique',
                                                                 'days_since_prior_order':'mean',
                                                                 'reordered':'sum',
                                                                 'Day_1':'sum',
                                                                 'Day_2':'sum',
                                                                 'Day_3':'sum',
                                                                 'Day_4':'sum',
                                                                 'Day_5':'sum',
                                                                 'Day_6':'sum',
                                                                 'Hour_1':'sum',
                                                                 'Hour_2':'sum',
                                                                 'Hour_3':'sum',
                                                                 'Hour_4':'sum',
                                                                 'Hour_5':'sum',
                                                                 'Hour_6':'sum',
                                                                 'Hour_7':'sum',
                                                                 'Hour_8':'sum',
                                                                 'Hour_9':'sum',
                                                                 'Hour_10':'sum',
                                                                 'Hour_11':'sum',
                                                                 'Hour_12':'sum',
                                                                 'Hour_13':'sum',
                                                                 'Hour_14':'sum',
                                                                 'Hour_15':'sum',
                                                                 'Hour_16':'sum',
                                                                 'Hour_17':'sum',
                                                                 'Hour_18':'sum',
                                                                 'Hour_19':'sum',
                                                                 'Hour_20':'sum',
                                                                 'Hour_21':'sum',
                                                                 'Hour_22':'sum',
                                                                 'Hour_23':'sum'
                                                                }).reset_index()
user_purchase_df.head()


# In[36]:


product_purchase_df = dummies_df.groupby(['product_id']).agg({   'order_id':'nunique',
                                                                 'user_id': 'nunique',
                                                                 'days_since_prior_order':'mean',
                                                                 'reordered':'sum',
                                                                 'Day_1':'sum',
                                                                 'Day_2':'sum',
                                                                 'Day_3':'sum',
                                                                 'Day_4':'sum',
                                                                 'Day_5':'sum',
                                                                 'Day_6':'sum',
                                                                 'Hour_1':'sum',
                                                                 'Hour_2':'sum',
                                                                 'Hour_3':'sum',
                                                                 'Hour_4':'sum',
                                                                 'Hour_5':'sum',
                                                                 'Hour_6':'sum',
                                                                 'Hour_7':'sum',
                                                                 'Hour_8':'sum',
                                                                 'Hour_9':'sum',
                                                                 'Hour_10':'sum',
                                                                 'Hour_11':'sum',
                                                                 'Hour_12':'sum',
                                                                 'Hour_13':'sum',
                                                                 'Hour_14':'sum',
                                                                 'Hour_15':'sum',
                                                                 'Hour_16':'sum',
                                                                 'Hour_17':'sum',
                                                                 'Hour_18':'sum',
                                                                 'Hour_19':'sum',
                                                                 'Hour_20':'sum',
                                                                 'Hour_21':'sum',
                                                                 'Hour_22':'sum',
                                                                 'Hour_23':'sum'
                                                                }).reset_index()
product_purchase_df.head()


# In[37]:


temp = pd.merge(left=user_prod_df,  right=user_purchase_df, on='user_id', suffixes=('','_user'))
temp.head(10)


# In[38]:


features_df = pd.merge(left=temp,  right=product_purchase_df, on='product_id', suffixes=('','_prod'))
features_df.head(10)


# In[39]:


def my_reset(varnames):
    """
    varnames are what you want to keep
    """
    globals_ = globals()
    to_save = {v: globals_[v] for v in varnames}
    to_save['my_reset'] = my_reset  # lets keep this function by default
    del globals_
    get_ipython().magic("reset")
    globals().update(to_save)
    
variables = ['features_df']
my_reset(variables)


# In[41]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        col_type2 = df[col].dtype.name
        
        if ((col_type != object) and (col_type2 != 'category')):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[42]:


import numpy as np
reduce_features_df = reduce_mem_usage(features_df)


# In[43]:


reduce_features_df.isnull().sum().sort_values()


# In[44]:


reduced_feature= reduce_features_df[:1000]
reduced_feature


# In[47]:


from sklearn.model_selection import train_test_split

# Assuming reduced_feature is your feature matrix or dataset
# You have already imported train_test_split from scikit-learn

# Splitting the data into training and testing sets
X_train, X_test = train_test_split(reduced_feature, test_size=0.3, random_state=100)

# Printing the shapes of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)


# In[48]:


# Building Neareset Neighbours Classifier with Cosine distance measure

from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(X_train)


# In[49]:


import numpy as np
query_index = np.random.choice(X_train.shape[0])
distances, indices = model_knn.kneighbors(X_train.iloc[query_index, :].values.reshape((1, -1)), n_neighbors = 6)

j=1
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(X_train.index[indices.flatten()[i]]))
    else:
        try:
            print('{0}: {1}'.format(j, df1[df1['product_id']==X_train.index[indices.flatten()[i]]].product_name.values[0]))
            j+=1
        except:
            pass


# In[ ]:


# 6. Project Outcomes & Conclusions
# Here are some of the key outcomes of the project:
# The Dataset was quiet large with combined data totally around 1.3M.
# There were also few outliers & no duplicates present in the datset, which had to be dropped.
# Visualising the distribution of data & their relationships, helped us to get some insights on the relationship between the featureset.
# Further filtering was done with threshold for the number of user id's & product id's.
# Finally Nearest Neighbours Algorithm was employed to get the similar Groceries Recommendations based on the Cosine Similarity.


# In[51]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Assuming you have target labels for training and testing sets
# Assuming y_train and y_test are your target labels

# Instantiate the model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the testing set:", accuracy)


# In[52]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Assuming reduced_feature contains both features and the target variable
# Assuming the target variable is the last column

# Separate features (X) and the target variable (y)
X = reduced_feature.iloc[:, :-1]  # Assuming the target variable is in the last column
y = reduced_feature.iloc[:, -1]   # Assuming the target variable is in the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Instantiate the model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the testing set:", accuracy)


# In[53]:


from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Instantiate the model
model = DecisionTreeClassifier()

# Define hyperparameters to tune
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found by grid search
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the best model using cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation accuracy:", cv_scores.mean())

# Train the best model on the full training set
best_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = best_model.predict(X_test)

# Evaluate the best model on the testing set
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the testing set:", accuracy)


# In[54]:


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[63]:


from sklearn.metrics import accuracy_score

# Assuming you have trained your model and made predictions
# Let's say your predictions are stored in y_pred and your true labels are in y_test

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model:", accuracy)


# In[ ]:





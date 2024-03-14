#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifier
dt_classifier = DecisionTreeClassifier()

# Define the hyperparameters to tune
params = {
    'criterion': ['gini', 'entropy'],

}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(dt_classifier, params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best hyperparameters to train the classifier
best_dt_classifier = DecisionTreeClassifier(**best_params)
best_dt_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = best_dt_classifier.predict(X_test)

# Evaluate the classifier performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)



# In[2]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifier
nb_classifier = GaussianNB()

# Define the hyperparameters to tune
params = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(nb_classifier, params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best hyperparameters to train the classifier
best_nb_classifier = GaussianNB(**best_params)
best_nb_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = best_nb_classifier.predict(X_test)

# Evaluate the classifier performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[3]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the SVM classifier
svm = SVC()

# Define hyperparameters to tune
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best hyperparameters to train the classifier
best_svm = SVC(**best_params)
best_svm.fit(X_train, y_train)

# Predictions on the test set
y_pred = best_svm.predict(X_test)

# Evaluate the classifier performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# 4.. Explain various types of  Kernels with respect to the formula 
# 
# 

# # Linear Kernel:
# # The linear kernel computes the dot product between two vectors. It is simple and efficient, making it suitable for linearly separable data.
# 
# # Polynomial Kernel:
# # The polynomial kernel maps data into a higher-dimensional space using polynomial functions. It's useful for capturing non-linear relationships in the data. The degree of the polynomial determines the complexity of the decision boundary.
# 
# # Gaussian (RBF) Kernel:
# # The Gaussian kernel, also known as the Radial Basis Function (RBF) kernel, assigns weights to data points based on their distance from a reference point. It maps data into an infinite-dimensional space and is effective for capturing complex, non-linear relationships.
# 
# # Sigmoid Kernel:
# # The sigmoid kernel is inspired by the sigmoid function and maps data into a higher-dimensional space. It is useful for learning non-linear decision boundaries and can be customized using parameters to adjust its behavior.
# 
# # These kernels are commonly used in machine learning algorithms like support vector machines (SVMs) for tasks such as classification, regression, and clustering. Each kernel has its own properties and is suited to different types of data and problem scenarios.

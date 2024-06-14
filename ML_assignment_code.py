#===============================================#
#=======Importing all packages necessary========#
#===============================================#

#Note:Install whatever necessary 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.model_selection import train_test_split, GridSearchCV
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, plot_roc_curve 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import random
import warnings

#Some of the classifiers bring up Future Warnings, silence them: 
warnings.filterwarnings("ignore", category=FutureWarning)

#===============================================#
#===Setting seed to ensure reproducibiity=======#
#===============================================#

np.random.seed(31415)
random.seed (42069)

#===============================================#
#============== Importing data  ================#
#===============================================#

df = pd.read_csv("Newdata-2.csv") #Note the path to the file can be different for each user.

#===============================================#
#================ Data Prepping ================#
#===============================================#

Data_completeness = df.info() # checking if there are nulls: NONE 
print("Information on the data: ", Data_completeness)

#One Hot Encode the Customer_Type variable 
df['New_Customer'] = df['Customer_Type'].apply(lambda x: 1 if x == 'New_Customer' else 0)
df['Ret_Customer'] = df['Customer_Type'].apply(lambda x: 1 if x == 'Returning_Customer' else 0)
df['Other'] = df['Customer_Type'].apply(lambda x: 1 if x == 'Other' else 0)

#Make integer variable (encode) out of boolean variable Weekday
df['Weekday_Clean'] = df['Weekday'].astype(int)

#Scale the Systems columns 
columns_to_scale = ['SystemF1', 'SystemF2', 'SystemF3', 'SystemF4', 'SystemF5']
scaler = MinMaxScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])*10

#===============================================#
#=== Data split and SMOTE: prep the datasets====#
#===============================================#

#Assigning X and y 
y = df['Transaction']

selected_columns = ['SystemF1', 'SystemF2', 'SystemF3', 'SystemF4', 'SystemF5', 'Account_Page', 'Account_Page_Time',
                    'Info_Page', 'Info_Page_Time', 'ProductPage', 'ProductPage_Time', 'Month', 'SpecificHoliday',
                    'GoogleAnalytics_BR', 'GoogleAnalytics_ER', 'GoogleAnalytics_PV', 'Ad_Campaign_1', 'Ad_Campaign2',
                    'Ad_Campaign3', 'New_Customer', 'Ret_Customer', 'Other' , 'Weekday_Clean']
X = df[selected_columns]

#Split into training and testing sets, allocating 20% to test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  #Note that seed is added for reproducibility 

# Apply SMOTE to the training set
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#===============================================#
#================== kNN ========================#
#===============================================#

# Define the parameter grid for grid search
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan']  # Example values, adjust as needed
}

# Create a kNN classifier
clf_knn = KNeighborsClassifier()

# Create GridSearchCV object with kNN classifier and parameter grid
grid_search = GridSearchCV(clf_knn, param_grid, cv=5)  # cv=5 for 5-fold cross-validation

# Train the classifier on the training data using grid search to find the optimal hyperparameters (tuning)
grid_search.fit(X_train_resampled, y_train_resampled)

# Use the optimal HP found with grid search to make predictions on the testing data
best_clf_knn = grid_search.best_estimator_
y_pred_knn = best_clf_knn.predict(X_test)

# Print the best hyperparameters found by grid search, knowing the choosen HPs ensures reproducibility
print("Best hyperparameters kNN:", grid_search.best_params_)

# Calculate the performance metrics for the classifier
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
ROC_AUC_knn = roc_auc_score(y_test, y_pred_knn, average="macro")

#Plot the Confusion Matrix 
skplt.metrics.plot_confusion_matrix(y_test, y_pred_knn)
plt.title('Confusion Matrix KNN', fontsize=12)
plt.show()  #This picture has been attached in the abstract 

#===============================================#
#============== Logistic Regression ============#
#===============================================#

# Create a pipeline with StandardScaler and SGDClassifier, we standarized the variables to improve model performance 
clf_lr = make_pipeline(StandardScaler(), SGDClassifier())

# Define the parameter grid for grid search
param_grid = {
    'sgdclassifier__alpha': [0.0001, 0.001, 0.01],  # Regularization parameter
    'sgdclassifier__eta0': [0.01, 0.1, 1.0],  # Initial learning rate
}

# Create GridSearchCV object with Logistic Regression classifier and parameter grid
grid_search = GridSearchCV(clf_lr, param_grid, cv=5)  # cv=5 for 5-fold cross-validation

# Train the classifier on the training data using grid search to find the optimal hyperparameters (tuning)
grid_search.fit(X_train_resampled, y_train_resampled)

# Use the optimal HP found with grid search to make predictions on the testing data
best_clf_lr = grid_search.best_estimator_
y_pred_lr = best_clf_lr.predict(X_test)

# Calculate the performance metrics for the classifier
accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
ROC_AUC_lr = roc_auc_score(y_test, y_pred_lr, average="macro")

# Print the best hyperparameters found by grid search
print("Best hyperparameters Logistic Regression:", grid_search.best_params_)

#Plot the Confusion Matrix 
skplt.metrics.plot_confusion_matrix(y_test, y_pred_lr)
plt.title('Confusion Matrix Logistic Regression', fontsize=12)
plt.show()   #This picture has been attached in the abstract 

#===============================================#
#================ Random Forest ================#
#===============================================#

# Create a Random Forest classifier
clf_rf = RandomForestClassifier()

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [20,50,100],  # Number of trees in the forest
    'max_depth': [None, 5, 10],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
}


# Create GridSearchCV object with Random Forest classifier and parameter grid
grid_search = GridSearchCV(clf_rf, param_grid, cv=5)  # cv=5 for 5-fold cross-validation

# Train the classifier on the training data using grid search to find the optimal hyperparameters (tuning)
grid_search.fit(X_train, y_train)

# Use the optimal HP found with grid search to make predictions on the testing data
best_clf_rf = grid_search.best_estimator_
y_pred_rf = best_clf_rf.predict(X_test)

# Calculate the performance metrics for the classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
ROC_AUC_rf = roc_auc_score(y_test, y_pred_rf,  average="macro")

# Print the best hyperparameters found by grid search
print("Best hyperparameters for Descision Trees:", grid_search.best_params_)

skplt.metrics.plot_confusion_matrix(y_test, y_pred_rf)
plt.title('Confusion Matrix Random Forest', fontsize=12)
plt.show()  #This picture has been attached in the abstract 

#===============================================#
#================== Results ====================#
#===============================================#
#Result Metrics:
result_metrics = {
    'Model': ['KNN', 'Logistic Regression', 'Random Forest'],
    'Accuracy': [accuracy_knn, accuracy_lr, accuracy_rf],
    'F1 Score': [f1_knn, f1_lr, f1_rf],
    'Precision': [precision_knn, precision_lr, precision_rf],
    'Recall': [recall_knn, recall_lr, recall_rf],
    'ROC AUC': [ROC_AUC_knn, ROC_AUC_lr, ROC_AUC_rf]
}

df_result_metrics = pd.DataFrame(result_metrics)
df_result_metrics.set_index('Model', inplace=True)
df_result_metrics.to_csv("Metrics.csv")  #make it into a csv file to make it easier to copy into the report table.
# Display the DataFrame
print("Result Metrics: ", df_result_metrics)

#ROC Curves to be placed into the appendix: 
# Plot ROC curve for KNN
plt.figure()
plot_roc_curve(best_clf_knn, X_test, y_test, label=f"macro-average ROC curve (AUC = {ROC_AUC_knn:.2f})")
plt.title('KNN ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_knn.png')

# Plot ROC curve for Logistic Regression
plt.figure()
plot_roc_curve(best_clf_lr, X_test, y_test, label=f"macro-average ROC curve (AUC = {ROC_AUC_lr:.2f})")
plt.title('Logistic Regression ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_lr.png')

# Plot ROC curve for Random Forest
plt.figure()
plot_roc_curve(best_clf_rf, X_test, y_test, label=f"macro-average ROC curve (AUC = {ROC_AUC_rf:.2f})")
plt.title('Random Forest ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_rf.png')

# Display the plots
plt.show()

#===============================================#
#=========== Export Results to csv =============#
#===============================================#

df_results = pd.DataFrame({'True Y': y_test, 'kNN Estimate of Y': y_pred_knn, 'Logistic Regression Estimate of Y': y_pred_lr, 'Random Forest Estimate of Y': y_pred_rf})
df_results.to_csv( 'Classification_Results_Group_11.csv', index=False)

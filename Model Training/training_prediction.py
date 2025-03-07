import itertools
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, davies_bouldin_score, \
    silhouette_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, learning_curve
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris


################################ data preparation ####################################
#data set
df = pd.read_csv("dfnew.csv")

# drop the not relevant columns
df = df.drop(columns=['Flight Distance Rank', 'Departure Delay Rank', 'Baggage service'])
df.drop(columns=df.columns[0], axis=1, inplace=True)

df['Gender'] = df['Gender'].replace(
    {'Male': 1, 'Female': 0})

df['Customer Type'] = df['Customer Type'].replace(
    {'Loyal Customer': 1, 'disloyal Customer': 0})

df['Type of Travel'] = df['Type of Travel'].replace(
    {'Personal Travel': 1, 'Business travel': 0})

df['satisfaction'] = df['satisfaction'].replace(
    {'satisfied': 1, 'neutral or dissatisfied': 0})

#dummy variable
def dummy_categories(df, column_name):
    encoded_df = pd.get_dummies(df[column_name], dtype=int)
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(column_name, axis=1, inplace=True)
    return df

#dummy variable df1
df = dummy_categories(df, 'Class')
df = dummy_categories(df, 'Plane colors')

#copy the data set
df2 = df

#Normalizes continuous variables to variables between 0 and 1 - minmax_scaling
def minmax_scaling(df, col):
    Scaler = MinMaxScaler()
    Scaler.fit(df[[col]])
    # minmax scaling using sklearn that normalizes the column and returns it
    df[col] = Scaler.transform(df[[col]])
    return df

df = minmax_scaling(df, 'Age')
df = minmax_scaling(df, 'Flight Distance')
df = minmax_scaling(df, 'Inflight wifi service')
df = minmax_scaling(df, 'Departure/Arrival time convenient')
df = minmax_scaling(df, 'Ease of Online booking')
df = minmax_scaling(df, 'Gate location')
df = minmax_scaling(df, 'Food and drink')
df = minmax_scaling(df, 'Seat comfort')
df = minmax_scaling(df, 'On-board service')
df = minmax_scaling(df, 'Baggage handling')
df = minmax_scaling(df, 'Checkin service')
df = minmax_scaling(df, 'Inflight service')
df = minmax_scaling(df, 'Cleanliness')
df = minmax_scaling(df, 'Departure Delay in Minutes')
df = minmax_scaling(df, 'Service quality')


#Normalizes continuous variables to standati nornaly - StandardScaler()
def standard_scaling(df, col):
    scaler = StandardScaler()
    scaler.fit(df[[col]])
    # standard scaling using sklearn that normalizes the column and returns it
    df[col] = scaler.transform(df[[col]])
    return df

# Assuming df2 is your DataFrame
df2 = standard_scaling(df2, 'Age')
df2 = standard_scaling(df2, 'Flight Distance')
df2 = standard_scaling(df2, 'Inflight wifi service')
df2 = standard_scaling(df2, 'Departure/Arrival time convenient')
df2 = standard_scaling(df2, 'Ease of Online booking')
df2 = standard_scaling(df2, 'Gate location')
df2 = standard_scaling(df2, 'Food and drink')
df2 = standard_scaling(df2, 'Seat comfort')
df2 = standard_scaling(df2, 'On-board service')
df2 = standard_scaling(df2, 'Baggage handling')
df2 = standard_scaling(df2, 'Checkin service')
df2 = standard_scaling(df2, 'Inflight service')
df2 = standard_scaling(df2, 'Cleanliness')
df2 = standard_scaling(df2, 'Departure Delay in Minutes')
df2 = standard_scaling(df2, 'Service quality')


#Training set and validation set df1___
X = df.drop(['satisfaction'], axis=1)
Y = df['satisfaction']

# Convert all column names to strings
X.columns = X.columns.astype(str)

# Split the data into training and holdout sets with a 90-10 ratio, stratified by the target variable
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Join the training and testing set with their Y values respectively
TrainingSet = pd.concat([X_train, y_train], axis=1)
TestSet = pd.concat([X_test, y_test], axis=1)
y_train = y_train.astype('int')
y_test = y_test.astype('int')


#Training set and validation set df2___
X2 = df2.drop(['satisfaction'], axis=1)
Y2 = df2['satisfaction']

# Convert all column names to strings
X2.columns = X2.columns.astype(str)

# Split the data into training and holdout sets with a 90-10 ratio, stratified by the target variable
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size=0.1, stratify=Y, random_state=1)

# Join the training and testing set with their Y values respectively
TrainingSet2 = pd.concat([X2_train, y2_train], axis=1)
TestSet2 = pd.concat([X2_test, y2_test], axis=1)
y2_train = y2_train.astype('int')
y2_test = y2_test.astype('int')
# different names for Kmeans and Decision Tree
X_Training_Selected = TrainingSet.drop(['satisfaction'], axis=1)
y_Training_Selected = TrainingSet['satisfaction']
X_Training_Selected.columns = X_Training_Selected.columns.astype(str)

X_Testing_Selected = TestSet.drop(['satisfaction'], axis=1)
y_Testing_Selected = TestSet['satisfaction']
X_Testing_Selected.columns = X_Testing_Selected.columns.astype(str)


######################################## Decision tree #########################################
'''************************** 1.Decision tree ******************************'''
# Run the default full decision tree (overfitting)
TreeModel = DecisionTreeClassifier(random_state=40, criterion='entropy')
TreeModel.fit(X_Training_Selected, y_Training_Selected)



y_training_Predicitons = TreeModel.predict_proba(X_Training_Selected)[:, 1]
y_training_Predicitons = (y_training_Predicitons >= 0.5).astype(int)
y_Training_Selected = (y_Training_Selected >= 0.5).astype(int)

train_score = f1_score(y_Training_Selected, y_training_Predicitons)

y_Testing_Predicitons = TreeModel.predict_proba(X_Testing_Selected)[:, 1]
y_Testing_Predicitons = (y_Testing_Predicitons >= 0.5).astype(int)
y_Testing_Selected = (y_Testing_Selected >= 0.5).astype(int)
test_score = f1_score(y_Testing_Selected, y_Testing_Predicitons)

print('f1 Training score: {:.4f}'.format(train_score))
print('f1 Testiing score: {:.4f}'.format(test_score))

# Plotting the leaves amount with help of kfold to check accuracy using F1

Min_Sample_Leaves_Amount = np.arange(1, 1350, 50)
TreeModel = DecisionTreeClassifier(random_state=40)
SKFold_CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
ParamatersDict = {'min_samples_leaf': Min_Sample_Leaves_Amount}

# using grid search to find min leaves amount that we can starting our paraments from:
grid = GridSearchCV(estimator=TreeModel, param_grid=ParamatersDict, scoring='f1',
                    cv=SKFold_CV, n_jobs=-1, refit=True, return_train_score=True)
grid.fit(X_Training_Selected, y_Training_Selected)
cv_results = pd.DataFrame(grid.cv_results_)
Selected_Leaves = ['mean_test_score',
                   'mean_train_score', 'param_min_samples_leaf']

DF_Selected_Values = cv_results[Selected_Leaves]
DF_Selected_Values = DF_Selected_Values.sort_values(
    'param_min_samples_leaf', ascending=True)

plt.figure(figsize=(13, 4))
plt.plot(DF_Selected_Values['param_min_samples_leaf'],
         DF_Selected_Values['mean_train_score'], marker='x', markersize=4)
plt.plot(DF_Selected_Values['param_min_samples_leaf'],
         DF_Selected_Values['mean_test_score'], marker='o', markersize=4)
plt.title('min samples leaves Amount F1 score')
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.xlabel('min_samples_leaf')
plt.xticks([int(x) for x in DF_Selected_Values['param_min_samples_leaf']])
plt.ylabel('F1 score')
plt.show()

# pre-pruning with criterion and class weight params:

max_depth_list = np.arange(1, 20)
leaf_samples = np.arange(50, 250, 5)
params_dt = {
    'max_depth': max_depth_list,
    'criterion': ['entropy', 'gini'],
    'class_weight': ['balanced', None],
    'min_samples_leaf': leaf_samples,
}
TreeModel = DecisionTreeClassifier(random_state=40)
SKFold_CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# use grid tree to tune the parameters
Grids_Tree = GridSearchCV(estimator=TreeModel, param_grid=params_dt, scoring='f1',
                          cv=SKFold_CV, n_jobs=-1, refit=True, return_train_score=True)
Grids_Tree.fit(X_Training_Selected, y_Training_Selected)

best_hyperparameters = Grids_Tree.best_params_
best_TreeModel = Grids_Tree.best_estimator_

# Get predictions of the best model:

Tuned_y_Training_Predictions = best_TreeModel.predict(X_Training_Selected)
Tuned_Train_f1_score = f1_score(
    y_Training_Selected, Tuned_y_Training_Predictions)

Tuned_y_Testing_Predictions = best_TreeModel.predict(X_Testing_Selected)

Tuned_Test_f1_score = f1_score(
    y_Testing_Selected, Tuned_y_Testing_Predictions)

print('F1 Training score: {:.4f}'.format(Tuned_Train_f1_score))
print('F1 Testiing score: {:.4f}'.format(Tuned_Test_f1_score))

# create cv data frame with the results
cv_results = pd.DataFrame(Grids_Tree.cv_results_)
Selected_Leaves = ['std_test_score', 'mean_test_score', 'mean_train_score',
                   'param_max_depth', 'param_criterion', 'param_class_weight', 'param_min_samples_leaf']
DF_Selected_Values = cv_results[Selected_Leaves]
DF_Selected_Values = DF_Selected_Values.sort_values(
    'mean_test_score', ascending=False).head(10)
DF_Selected_Values['mean_test_score'] = DF_Selected_Values['mean_test_score'].round(
    4)
DF_Selected_Values['mean_train_score'] = DF_Selected_Values['mean_train_score'].round(
    4)
DF_Selected_Values['std_test_score'] = DF_Selected_Values['std_test_score'].round(
    4)
column_names = {
    'mean_test_score': 'Mean test score',
    'mean_train_score': 'Mean train score',
    'param_max_depth': 'Max depth',
    'param_criterion': 'Criterion',
    'param_class_weight': 'Class weight',
    'param_min_samples_leaf': 'Min samples leaf',
    'std_test_score': 'std test score'
}

# plot the results
DF_Selected_Values = DF_Selected_Values.rename(columns=column_names)
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')
ax.set_title('Grid Search results', y=1.1)
table = ax.table(cellText=DF_Selected_Values.values,
                 colLabels=DF_Selected_Values.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.25, 1.25)
plt.show()

Tuned_y_predicts = best_TreeModel.predict(X_Testing_Selected)

# plot the importance features table
importance = best_TreeModel.feature_importances_
importance = pd.DataFrame({'Feature_name': X_Training_Selected.columns,
                           'Importance': best_TreeModel.feature_importances_.round(4)})

fig, ax = plt.subplots(figsize=(20, 20))
ax.axis('off')
table = ax.table(cellText=importance.values,
                 colLabels=importance.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(0.4, 1.25)
plt.show()

# plot tree:
plt.figure(figsize=(22, 15))
plot_tree(best_TreeModel, filled=True, class_names=[
    'Negative', 'Positive'], max_depth=2)
plt.rcParams.update({'font.size': 30})
plt.show()

'''**optimal model'''
optimal_model = DecisionTreeClassifier(
    min_samples_leaf=85, max_depth=8, class_weight='balanced', random_state=40)
optimal_model.fit(X_Training_Selected, y_Training_Selected)

# get the training and validation set F1 scoring
Tuned_y_Training_Predictions = optimal_model.predict(X_Training_Selected)
Tuned_Train_F1 = f1_score(
    y_Training_Selected, Tuned_y_Training_Predictions)

y_pred_proba = optimal_model.predict(X_Testing_Selected)
Tuned_Test_F1 = f1_score(y_Testing_Selected, y_pred_proba)

print('F1 Training score: {:.4f}'.format(Tuned_Train_F1))
print('F1 Testiing score: {:.4f}'.format(Tuned_Test_F1))





######################################## Neural Networks #########################################

#________ 3.1 Pre-Processing_________
#in data preparation

#________ 3.2 A deductive neural network_________

# Initialize the MLPClassifier model
model = MLPClassifier(random_state=1)

# Fit the model df1
model.fit(X_train, y_train)

y_ANN_pred_train = model.predict(X_train)
f1_score_train = f1_score(y_train, y_ANN_pred_train)

y_ANN_pred_test = model.predict(X_test)
f1_score_test = f1_score(y_test, y_ANN_pred_test)

# Fit the model df2
model.fit(X2_train, y2_train)

y_ANN_pred_train2 = model.predict(X2_train)
f1_score_train2 = f1_score(y2_train, y_ANN_pred_train2)

y_ANN_pred_test2 = model.predict(X2_test)
f1_score_test2 = f1_score(y2_test, y_ANN_pred_test2)

#print Calculate F1 scores for df1
print("defult")
print('f1 Training score: {:.4f}'.format(f1_score_train))
print('f1 Testiing score: {:.4f}'.format(f1_score_test))

#print Calculate F1 scores for df2
print('f1 Training score2: {:.4f}'.format(f1_score_train2))
print('f1 Testiing score2: {:.4f}'.format(f1_score_test2))



#________ 3.3 Hyper parameter tuning_________

#Checking what the parameters of bring about the best result
Model_Tuning = MLPClassifier(random_state=1)
stratified_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
layer_size = range(3, 6)
neuron_Amount = range(2, 50, 5)
# create permutations of possible neurons layers
hidden_layer_combinations = []
for i in layer_size:
    combination = list(itertools.permutations(neuron_Amount, i))
    hidden_layer_combinations.extend(combination)

params = {
    'hidden_layer_sizes': hidden_layer_combinations,
    'batch_size': [32, 64, 128, 256],
    'max_iter': [100],
    'learning_rate_init': [0.0005, 0.001, 0.005, 0.01, 0.05],
    'learning_rate': ['constant', 'adaptive'],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam']
}

# use random search to save runtime to find best hyperparams configuration:
MLPRandomSearch = RandomizedSearchCV(estimator=Model_Tuning, param_distributions=params, scoring='f1',
                                     cv=stratified_cv, n_jobs=-1, refit=True, return_train_score=True,
                                     random_state=1, n_iter=100)

# run the search
MLPRandomSearch.fit(X2_train, y2_train)
# save the best model in a variable
BestModel = MLPRandomSearch.best_estimator_
best_hyperparams = MLPRandomSearch.best_params_
#Print best hyperparams
print(best_hyperparams)

# Calculate F1 score for train and test sets
y_train_pred_labels = BestModel.predict(X2_train)
train_f1_score = f1_score(y2_train, y_train_pred_labels)

y_test_pred_labels = BestModel.predict(X2_test)
test_f1_score = f1_score(y2_test, y_test_pred_labels)


#plots____

#1 plot loss curve:
plt.plot(BestModel.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()


#2 plot cv results- best 10:
cv_results = pd.DataFrame(MLPRandomSearch.cv_results_)
selected_columns = ['std_test_score', 'mean_test_score', 'mean_train_score', 'param_hidden_layer_sizes',
                    'param_batch_size', 'param_learning_rate_init', 'param_learning_rate', 'param_activation',
                    'param_solver']
df_selected = cv_results[selected_columns]
df_selected = df_selected.sort_values(
    'mean_test_score', ascending=False).head(10)
df_selected['mean_test_score'] = df_selected['mean_test_score'].round(4)
df_selected['mean_train_score'] = df_selected['mean_train_score'].round(4)
df_selected['std_test_score'] = df_selected['std_test_score'].round(4)
column_names = {
    'mean_test_score': 'Mean test score',
    'mean_train_score': 'Mean train score',
    'param_hidden_layer_sizes': 'Hidden Layer Size',
    'param_batch_size': 'Batch Size',
    'param_learning_rate_init': 'Learning Rate Init',
    'param_learning_rate': 'Learning Rate',
    'param_activation': 'Activation',
    'param_solver': 'Solver',
    'std_test_score': 'std test score'
}
df_selected = df_selected.rename(columns=column_names)
fig, ax = plt.subplots(figsize=(17, 5))
ax.axis('off')
ax.set_title('Random search results', y=1.1)
table = ax.table(cellText=df_selected.values,
                 colLabels=df_selected.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.25, 1.25)
plt.show()


#3 confusion matrix for best model
# Define the best parameters
best_params = {'hidden_layer_sizes': (7, 12, 47, 2, 22), 'batch_size': 32, 'learning_rate_init': 0.01,
                'learning_rate': 'adaptive', 'activation': 'logistic', 'solver': 'adam', 'random_state': 1,
                'max_iter': 100}

# Initialize the model with the best parameters
BestModel = MLPClassifier(
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    batch_size=best_params['batch_size'],
    learning_rate_init=best_params['learning_rate_init'],
    learning_rate=best_params['learning_rate'],
    activation=best_params['activation'],
    solver=best_params['solver'],
    random_state=best_params['random_state'],
    max_iter=best_params['max_iter']
)

# Train the model
BestModel.fit(X2_train, y2_train)

# Calculate F1 score for train and test sets
y_train_pred_labels = BestModel.predict(X2_train)
train_f1_score = f1_score(y2_train, y_train_pred_labels)
print("Train F1 Score: ", train_f1_score)

y_test_pred_labels = BestModel.predict(X2_test)
test_f1_score = f1_score(y2_test, y_test_pred_labels)
print("Test F1 Score: ", test_f1_score)

# Generate confusion matrix for the test set
test_conf_matrix = confusion_matrix(y2_test, y_test_pred_labels)
print("Test Confusion Matrix:\n", test_conf_matrix)

# Calculate TP, TN, FP, FN for the test set
test_TP = test_conf_matrix[1, 1]
test_TN = test_conf_matrix[0, 0]
test_FP = test_conf_matrix[0, 1]
test_FN = test_conf_matrix[1, 0]

print("Test Set: TP={}, TN={}, FP={}, FN={}".format(test_TP, test_TN, test_FP, test_FN))

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

################################################  Clustering #############################################
'''******************* Clustering **************************'''

# create a plot for the first point of view
trainPlot = df
trainPlot = trainPlot[['Type of Travel','Inflight wifi service', 'Cleanliness', 'satisfaction']]

sns.pairplot(trainPlot, hue='satisfaction')
plt.show()
# PCA

pca = PCA()
# calculating the variation of each feature
pca.fit(trainPlot.drop(columns=['satisfaction']))
pca_data = pca.transform(trainPlot.drop(columns=['satisfaction']))

# calculating the percentage of variation that each feature accounts for (for the whole variables we have)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
# creating the bar plot
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.xticks(rotation=90)
plt.show()
# visualize the data by using PCA
pca = PCA(n_components=2)
pca.fit(trainPlot.drop(columns=['satisfaction']))
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

df_pca = pca.transform(trainPlot.drop(columns=['satisfaction']))
df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
df_pca['satisfaction'] = df['satisfaction']
sns.scatterplot(x='PC1', y='PC2', hue='satisfaction', data=df_pca)
plt.show()


#-----------Kmeans
kmeans = KMeans(n_clusters=2, max_iter=300, n_init=10, random_state=42)
kmeans.fit(trainPlot.drop(columns=['satisfaction']))
kmeans.predict(trainPlot.drop(columns=['satisfaction']))
df_pca['cluster'] = kmeans.predict(trainPlot.drop(columns=['satisfaction']))
sns.scatterplot(x='PC1', y='PC2', hue='satisfaction', data=df_pca, palette='Accent')
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+', s=100 ,color='red')
plt.show()

kmeans = KMeans(n_clusters=2, max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_Training_Selected)
y_pred= kmeans.predict(X_Training_Selected)



# confusion matrix:
def CM_Func(y, Tuned_y_predicts):
    ((tn, fp), (fn, tp)) = metrics.confusion_matrix(y, Tuned_y_predicts)
    ((tnr, fpr), (fnr, tpr)) = metrics.confusion_matrix(
        y, Tuned_y_predicts, normalize='true')
    return pd.DataFrame([[f'TN = {tn} (TNR = {tnr:1.2%})',
                          f'FP = {fp} (FPR = {fpr:1.2%})'],
                         [f'FN = {fn} (FNR = {fnr:1.2%})', f'TP = {tp} (TPR = {tpr:1.2%})']],
                        index=['True 0(neutral or dissatisfied)', 'True 1(satisfied)'],
                        columns=['Pred 0(Accept neutral or dissatisfied)', 'Pred 1(Accept satisfied)'])


ConfusionMatrix = CM_Func(y_Training_Selected, y_pred)
print(ConfusionMatrix)

# plot the matrix:
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_Training_Selected, y_pred), annot=True,
            fmt='d', cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
#-----

iner_list = []
dbi_list = []
sil_list = []

# for k_values range run KMeans for each:
k_values = range(2, 10)
for n_clusters in k_values:
    # Fit the KMeans
    kmeans = KMeans(n_clusters=n_clusters, max_iter=300, n_init=10, random_state=42)
    kmeans.fit(trainPlot.drop(columns=['satisfaction']))
    assignment = kmeans.predict(trainPlot.drop(columns=['satisfaction']))
    # value of measure for each eateration of k
    iner = kmeans.inertia_
    sil = silhouette_score(trainPlot.drop(columns=['satisfaction']), assignment)
    dbi = davies_bouldin_score(trainPlot.drop(columns=['satisfaction']), assignment)
    # lists of scores for each measure
    dbi_list.append(dbi)
    sil_list.append(sil)
    iner_list.append(iner)
# choosing the  k:
# plot inter
plt.plot(range(2, 10, 1), iner_list, marker='o')
plt.title("Inertia")
plt.xlabel("Number of clusters")
plt.show()

# plot davies
plt.plot(range(2, 10, 1), dbi_list, marker='o')
plt.title("Davies-Bouldin")
plt.xlabel("Number of clusters")
plt.show()

# plot silhouette
plt.plot(range(2, 10, 1), sil_list, marker='o')
plt.title("Silhouette")
plt.xlabel("Number of clusters")
plt.show()

# Best K found:
k = 3
kmeans = KMeans(n_clusters=3, max_iter=300, n_init=10, random_state=42)
kmeans.fit(trainPlot.drop(columns=['satisfaction']))
kmeans.predict(trainPlot.drop(columns=['satisfaction']))
df_pca['cluster'] = kmeans.predict(trainPlot.drop(columns=['satisfaction']))
sns.scatterplot(x='PC1', y='PC2', hue='satisfaction', data=df_pca, palette='Accent')
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+', s=100 ,color='red')
plt.show()





################################################  random forest #############################################

#Default random forest_
# Initialize the Random Forest model
modelRF = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
modelRF.fit(X2_train, y2_train)


# Make predictions on the training set
y2_train_pred_RF = modelRF.predict(X2_train)
# Make predictions
y2_test_pred_RF = modelRF.predict(X2_test)


# Evaluate the model using F1 score
f1_train_RF = f1_score(y2_train, y2_train_pred_RF, average='weighted')
f1_test_RF = f1_score(y2_test, y2_test_pred_RF, average='weighted')

print("random forest Default")
print(f'F1 Score on Training Set: {f1_train_RF:.2f}')
print(f'F1 Score on Test Set: {f1_test_RF:.2f}')



#improved random forest model-

# Initialize the Random Forest model
m = RandomForestClassifier(random_state=1)

"""
#_Define the parameter for RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
    'criterion': ['gini', 'entropy']
}
# Use RandomizedSearchCV to find the best parameters
grid_search_RF = RandomizedSearchCV(estimator=m, param_distributions=param_dist, n_iter=100, cv=5, scoring='f1_weighted', n_jobs=-1, random_state=1)
"""

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30,40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}


# Use GridSearchCV to find the best parameters
grid_search_RF = GridSearchCV(estimator=m, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)


# Train the model
grid_search_RF.fit(X2_train, y2_train)

# Get the best model
best_model_RF = grid_search_RF.best_estimator_


# Make predictions on the training set
y_train_pred_RF_B = best_model_RF.predict(X2_train)
# Make predictions on the test set
y_test_pred_RF_B = best_model_RF.predict(X2_test)


# Evaluate the model using F1 score
f1_train_RF_B = f1_score(y2_train, y_train_pred_RF_B, average='weighted')
f1_test_RF_B = f1_score(y2_test, y_test_pred_RF_B, average='weighted')

print("random forest best")
print(f'Best Parameters: {grid_search_RF.best_params_}')
print(f'F1 Score on Training Set: {f1_train_RF_B:.2f}')
print(f'F1 Score on Test Set: {f1_test_RF_B:.2f}')


#_plots

#1  Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, best_model_RF.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()



#2 Generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(best_model_RF, X2_train, y2_train, cv=5, scoring='f1_weighted', n_jobs=-1)

# Calculate mean and standard deviation for training and test scores
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
plt.xlabel('Training Size')
plt.ylabel('F1 Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()


#3 Generate validation curve data
param_range = [50, 100, 200]
train_scores, test_scores = validation_curve(RandomForestClassifier(), X2_train, y2_train, param_name="n_estimators", param_range=param_range, cv=5, scoring='f1_weighted', n_jobs=-1)

# Calculate mean and standard deviation for training and test scores
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

# Plot validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, 'o-', label='Training score')
plt.plot(param_range, test_mean, 'o-', label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
plt.xlabel('Number of Trees')
plt.ylabel('F1 Score')
plt.title('Validation Curve for Random Forest')
plt.legend(loc='best')
plt.show()


#4 Plot Confusion Matrix



ConfusionMatrix = CM_Func(y2_test, y_train_pred_RF_B)
print(ConfusionMatrix)

# plot the matrix:
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y2_test, y_train_pred_RF_B), annot=True,
            fmt='d', cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


############################################### final test ###################################################

# Read the CSV file into a DataFrame df_test
df_test = pd.read_csv(r"X_test.csv")


# Categorical variables (1, 2, 3)

df_test['Gender'] = df_test['Gender'].replace(
    {'Male': 1, 'Female':0})

df_test['Customer Type'] = df_test['Customer Type'].replace(
    {'Loyal Customer': 1, 'disloyal Customer': 0})

df_test['Type of Travel'] = df_test['Type of Travel'].replace(
    {'Personal Travel': 1, 'Business travel': 0})


# remove columns leg room service and arrival delay in minutes
df_test = df_test.drop(columns=['Leg room service', 'Arrival Delay in Minutes'])

#dummy variable df_test
df_test = dummy_categories(df_test, 'Class')
df_test = dummy_categories(df_test, 'Plane colors')

# Assuming df_test is your DataFrame
df_test = standard_scaling(df_test, 'Age')
df_test = standard_scaling(df_test, 'Flight Distance')
df_test = standard_scaling(df_test, 'Inflight wifi service')
df_test = standard_scaling(df_test, 'Departure/Arrival time convenient')
df_test = standard_scaling(df_test, 'Ease of Online booking')
df_test = standard_scaling(df_test, 'Gate location')
df_test = standard_scaling(df_test, 'Food and drink')
df_test = standard_scaling(df_test, 'Seat comfort')
df_test = standard_scaling(df_test, 'On-board service')
df_test = standard_scaling(df_test, 'Baggage handling')
df_test = standard_scaling(df_test, 'Checkin service')
df_test = standard_scaling(df_test, 'Inflight service')
df_test = standard_scaling(df_test, 'Cleanliness')
df_test = standard_scaling(df_test, 'Departure Delay in Minutes')
#df_test = standard_scaling(df_test, 'Service quality')



# Convert all column names to strings
df_test.columns = df_test.columns.astype(str)

# Ensure df_test has the same columns as the training data, in the same order
train_columns = best_model_RF.feature_names_in_
df_test = df_test.reindex(columns=train_columns)

# Now predict
final_pred = best_model_RF.predict(df_test)

# Add predictions to the DataFrame
df_test['Predicted_Satisfaction'] = final_pred

# Save the DataFrame with predictions as xlsx
df_test.to_excel("X_test_with_predictions.xlsx", index=False)

print("Predictions have been added to the DataFrame and saved as 'X_test_with_predictions.xlsx'")

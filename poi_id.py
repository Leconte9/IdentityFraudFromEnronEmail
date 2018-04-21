#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value',  'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'shared_receipt_with_poi', 'deferral_payments'] # You will need to use more features

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                      'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']  # 14
email_features_number = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # 5
email_features_text = ['email_address']
POI_label = ['poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# The number of total users
print  "The Number of Users: ", len(data_dict.keys()) # There are 146 users in dataset.
# The number of poi
count_poi = 0
for POIs in data_dict:
    if data_dict[POIs]['poi'] == True:
        count_poi += 1
    # Replace 'NaN' values to 0s
    for NA_keys in data_dict[POIs]:
        if data_dict[POIs][NA_keys] == 'NaN':
            data_dict[POIs][NA_keys] = 0
print "The Number of POIs: ", count_poi  # There are 18 POI in dataset.

### Task 2: Remove outliers
outlier_tester = ["salary", "bonus", "poi"]
outlier_data = featureFormat(data_dict, outlier_tester)

from operator import itemgetter
for point in outlier_data:
    if point[2] == False:
        salary = point[0]
        bonus = point[1]
        matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

# Remove the outlier(s)
data_dict.pop("TOTAL")

#print len(data_dict.keys())

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# Create new features named: 'messages_from_poi' and 'messages_to_poi/deferral_payments'
for names in my_dataset:
    from_messages = my_dataset[names]['from_messages']
    to_messages = my_dataset[names]['to_messages']
    from_poi_to_this_person = my_dataset[names]['from_poi_to_this_person']
    from_this_person_to_poi = my_dataset[names]['from_this_person_to_poi']
    deferral_payments = my_dataset[names]['deferral_payments']
    if float(from_messages) != 0:
        my_dataset[names]['messages_from_poi'] = from_poi_to_this_person/float(from_messages)
    else:
        my_dataset[names]['messages_from_poi'] = 0
    if float(to_messages) != 0 and deferral_payments != 0:
        my_dataset[names]['messages_to_poi/deferral_payments'] = from_this_person_to_poi/float(to_messages * deferral_payments)
    else:
        my_dataset[names]['messages_to_poi/deferral_payments'] = 0

features_list_new = POI_label + financial_features + email_features_number + ['messages_from_poi'] + ['messages_to_poi/deferral_payments']
#print "The List with all features with 2 new ones is:", features_list_new

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_new, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest, f_classif
featureSelecting = SelectKBest(f_classif, k = 10)
featureSelecting.fit(features, labels)
featureSelected = featureSelecting.get_support()
scores = zip(featureSelecting.scores_, features_list_new[1:], featureSelected)
scoresSorted = sorted(scores, reverse = True)
#print "Scroes are:", scoresSorted
''' 
    scoresSorted =
    [(25.09754152873549, 'exercised_stock_options', True),
    (24.4676540475264, 'total_stock_value', True),
    (21.06000170753657, 'bonus', True),
    (18.575703268041785, 'salary', True),
    (11.5955476597306, 'deferred_income', True),
    (10.072454529369441, 'long_term_incentive', True),
    (9.563091312863575, 'messages_to_poi/deferral_payments', True),
    (9.346700791051488, 'restricted_stock', True),
    (8.866721537107772, 'total_payments', True),
    (8.74648553212908, 'shared_receipt_with_poi', True),
    (7.242730396536018, 'loan_advances', False),
    (6.23420114050674, 'expenses', False),
    (5.344941523147337, 'from_poi_to_this_person', False),
    (5.209650220581797, 'messages_from_poi', False),
    (4.204970858301416, 'other', False),
    (2.426508127242878, 'from_this_person_to_poi', False),
    (2.107655943276091, 'director_fees', False),
    (1.69882434858085, 'to_messages', False),
    (0.2170589303395084, 'deferral_payments', False),
    (0.16416449823428736, 'from_messages', False),
    (0.06498431172371151, 'restricted_stock_deferred', False)]
    '''

#print len(features_list_new)
#print featureSelected

KBest_features = ['poi']
for i in range(len(featureSelected)):
    if featureSelected[i] == True:
#        print i, featureSelected[i], features_list_new[i+1]
        KBest_features.append(features_list_new[i+1])
print "Top 10 features, including new ones, are:", KBest_features
'''
    KBest_features are:
    ['salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'shared_receipt_with_poi', 'messages_to_poi/deferral_payments']
    '''

# scaler
# dataset with new features
from sklearn import preprocessing
n_data = featureFormat(my_dataset, KBest_features, sort_keys = True)
n_labels, n_features = targetFeatureSplit(n_data)
scaler = preprocessing.MinMaxScaler()
n_features = scaler.fit_transform(n_features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#new_data = featureFormat(my_dataset, KBest_features, sort_keys = True)
#n_labels, n_features = targetFeatureSplit(new_data)

# Evaluating Estimator Performance
# To avoid overfitting, part of the available data will be hold as a test set, while using the other part as training set.
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(n_features, n_labels, test_size=0.4, random_state=42)

from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time
from sklearn.model_selection import GridSearchCV

# Provided to give you a starting point. Try a variety of classifiers.
# GaussianNB
print " ---- GaussianNB ---- "
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()

## Training on original clf_nb
#t0 = time()
#clf_nb.fit(features_train, labels_train)
#print "GaussianNB Training Time: ", round(time() - t0, 3), "s"
## Predict on original clf_nb
#t1 = time()
#pred_nb = clf_nb.predict(features_test)
#print "GaussianNB Predict Time: ", round(time() - t1, 3), "s"
#acc_nb = accuracy_score(labels_test, pred_nb)
#pre_nb = precision_score(labels_test, pred_nb)
#rec_nb = recall_score(labels_test, pred_nb)
#print "GaussianNB accuracy: ", acc_nb
#print "GaussianNB precision: ", pre_nb
#print "GaussianNB recall: ", rec_nb
##---- GaussianNB ----
##GaussianNB Training Time:  0.001 s
##GaussianNB Predict Time:  0.0 s
##GaussianNB accuracy:  0.827586206897
##GaussianNB precision:  0.333333333333
##GaussianNB recall:  0.428571428571

grid_nb = GridSearchCV(estimator = clf_nb, param_grid = {})
grid_nb.fit(features_train, labels_train)
pred_grid_nb = grid_nb.predict(features_test)

acc_grid_nb = accuracy_score(labels_test, pred_grid_nb)
pre_grid_nb = precision_score(labels_test, pred_grid_nb)
rec_grid_nb = recall_score(labels_test, pred_grid_nb)
print "GaussianNB accuracy: ", acc_grid_nb
print "GaussianNB precision: ", pre_grid_nb
print "GaussianNB recall: ", rec_grid_nb

# Decision Tree
print " ---- Decision Tree ---- "
from sklearn import tree
clf_tr = tree.DecisionTreeClassifier()

## Training
#t0 = time()
#clf_tr.fit(features_train, labels_train)
#print "Decision Tree Training Time: ", round(time() - t0, 3), "s"
## Predict
#t1 = time()
#pred_tr = clf_tr.predict(features_test)
#print "Decision Tree Predict Time: ", round(time() - t1, 3), "s"
#acc_tr = accuracy_score(labels_test, pred_tr)
#pre_tr = precision_score(labels_test, pred_tr)
#rec_tr = recall_score(labels_test, pred_tr)
#print "Decision Tree accuracy: ", acc_tr
#print "Decision Tree precision: ", pre_tr
#print "Decision Tree recall: ", rec_tr
##    ---- Decision Tree ----
##Decision Tree Training Time:  0.001 s
##Decision Tree Predict Time:  0.0 s
##Decision Tree accuracy:  0.810344827586
##Decision Tree precision:  0.166666666667
##Decision Tree recall:  0.142857142857

grid_tr = GridSearchCV(estimator = clf_tr, param_grid = {'criterion':('gini', 'entropy'), 'splitter':('best','random')})
grid_tr.fit(features_train, labels_train)
pred_grid_tr = grid_tr.predict(features_test)

acc_grid_tr = accuracy_score(labels_test, pred_grid_tr)
pre_grid_tr = precision_score(labels_test, pred_grid_tr)
rec_grid_tr = recall_score(labels_test, pred_grid_tr)
print "Decision Tree accuracy: ", acc_grid_tr
print "Decision Tree precision: ", pre_grid_tr
print "Decision Tree recall: ", rec_grid_tr

bast_param_tr = grid_tr.best_estimator_.get_params()
print "Best params of Decesion Tree Classifier are: ", bast_param_tr
## ---- Decision Tree ----
#GaussianNB accuracy:  0.793103448276
#GaussianNB precision:  0.272727272727
#GaussianNB recall:  0.428571428571
#Best params of Decesion Tree Classifier are: {'presort': False, 'splitter': 'random', 'min_impurity_decrease': 0.0, 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 'random_state': None, 'min_impurity_split': None, 'max_features': None, 'max_depth': None, 'class_weight': None}

# Random Forest
print " ---- Random Forest ---- "
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators = 10)

# Training
t0 = time()
clf_rf.fit(features_train, labels_train)
print "Random Forest Training Time: ", round(time() - t0, 3), "s"
# Predict
t1 = time()
pred_rf = clf_rf.predict(features_test)
print "Random Forest Predict Time: ", round(time() - t1, 3), "s"
acc_rf = accuracy_score(labels_test, pred_rf)
pre_rf = precision_score(labels_test, pred_rf)
rec_rf = recall_score(labels_test, pred_rf)
print "Random Forest accuracy: ", acc_rf
print "Random Forest precision: ", pre_rf
print "Random Forest recall: ", rec_rf
#    ---- Random Forest ----
#Random Forest Training Time:  0.053 s
#Random Forest Predict Time:  0.002 s
#Random Forest accuracy:  0.879310344828
#Random Forest precision:  0.5
#Random Forest recall:  0.285714285714

# Logistic Regression
print " ---- Logistic Regression ---- "
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression(C = 1e5)

# Training
t0 = time()
clf_lr.fit(features_train, labels_train)
print "Logistic Regression Training Time: ", round(time() - t0, 3), "s"
# Predict
t1 = time()
pred_lr = clf_lr.predict(features_test)
print "Logistic Regression Predict Time: ", round(time() - t1, 3), "s"
acc_lr = accuracy_score(labels_test, pred_lr)
pre_lr = precision_score(labels_test, pred_lr)
rec_lr = recall_score(labels_test, pred_lr)
print "Logistic Regression accuracy: ", acc_lr
print "Logistic Regression precision: ", pre_lr
print "Logistic Regression recall: ", rec_lr
#    ---- Logistic Regression ----
#Logistic Regression Training Time:  0.006 s
#Logistic Regression Predict Time:  0.0 s
#Logistic Regression accuracy:  0.862068965517
#Logistic Regression precision:  0.4
#Logistic Regression recall:  0.285714285714

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = clf_nb
features_list = KBest_features

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

#python tester.py
# GaussianNB(priors=None)
# Accuracy: 0.84200	Precision: 0.38622	Recall: 0.31400	F1: 0.34639	F2: 0.32620
# Total predictions: 15000	True positives:  628	False positives:  998	False negatives: 1372	True negatives: 12002


# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#                       max_features=None, max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#                       splitter='best')
# Accuracy: 0.81427	Precision: 0.30131	Recall: 0.29800	F1: 0.29965	F2: 0.29866
# Total predictions: 15000	True positives:  596	False positives: 1382	False negatives: 1404	True negatives: 11618

# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', 
#                       max_depth=None, max_features='auto', max_leaf_nodes=None, 
#                       min_impurity_decrease=0.0, min_impurity_split=None, 
#                       min_samples_leaf=1, min_samples_split=2, 
#                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, 
#                       oob_score=False, random_state=None, verbose=0, 
#                       warm_start=False)
# Accuracy: 0.85767	Precision: 0.41636	Recall: 0.16800 F1: 0.23940	F2: 0.19076
# Total predictions: 15000	True positives:  336	False positives:  471	False negatives: 1664	True negatives: 12529

#LogisticRegression(C=100000.0, class_weight=None, dual=False,
#                   fit_intercept=True, intercept_scaling=1, max_iter=100,
#                   multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
#                   solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
# Accuracy: 0.69527	Precision: 0.10016	Recall: 0.16100	F1: 0.12349	F2: 0.14356
# Total predictions: 15000	True positives:  322	False positives: 2893	False negatives: 1678	True negatives: 10107

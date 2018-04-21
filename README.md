# Identity Fraud From Enron Email

## Project Background
In 2000, Enron, an energy trading company, was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.  

## What skills will be practiced in this project
 - Deal with an imperfect, real-world dataset
 - Validate a machine learning result using test data
 - Evaluate a machine learning result using quantitative metrics
 - Create, select and transform features
 - Compare the performance of machine learning algorithms
 - Tune machine learning algorithms for maximum performance
 - Communicate your machine learning algorithm results clearly

Question 1: Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

## Dataset Overview
The goal of this project is to use machine learning tools to predict Persons of Interest (POIs) from Enron employees, who have committed fraud, based on the public Enron financial and email data. POIs are one of the following three: indicted, settled withadmitting guilt, testified in excchange for immunity. 

As preprocessing to this project, Enron email and financial data have been combined into a dictionary, where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The dataset includes 146 employees, 18 of which are considered POIs while 128 are non-POIs. The features in the dataset fall into three major types, namely financial features, email features and POI labels. There are 21 features in total.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI label: [‘poi’] (boolean, represented as integer)

The most of features have missing values, marked as 'NaN' in the dataset. In order to be a better way to use machine learning tools, I will replace all these missing values by number zeros.

## Outliers
As we all know, outliers can make a big difference on the result we would processing. Identifying and cleaning away outliers is something we should always think about when looking at a dataset for the first time. Visualization will be one of the most powerful tools for finding outliers. I will use the matplotlib.pyplot module to help with the plots. 



Question 2: What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

Question 3: What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

Question 4: What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

Question 5: What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

Question 6: Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]




Referendes:
- [1] https://machinelearningmastery.com/feature-selection-machine-learning-python/
- [2] http://scikit-learn.org/stable/modules/feature_selection.html

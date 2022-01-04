# KaggleSurvey_Compensation_Prediction
A further exploration of 2020 Kaggle Machine Learning & Data Science Survey after the Kaggle Survey Income & Education.

**Objective:**
Given a set of survey responses by data scientists, predict what a survey respondent’s current yearly compensation is by training, validating, and tuning multi-class ordinary classification algorithm.

**Attachments:**
**kagglesurvey_compensation_prediction.ipynb:**  IPython Notebook detailing the analysis performed
**clean_kaggle_data_2020.csv** Input dataset based on the survey

**Process:**
- Clean multiple types of data, including incomplete data, categorical data, and ordinal data.
- Perform exploratory data analysis, feature selection applying the Lasso regression & Chi-square test, and visualize the order of feature importance.
- Implement multi-class ordinal logistic regression algorithm on the training data using 10-fold cross-validation.
- Identify all hyperparameters in the model, choose proper performance-measure metrics, and tune the model.
- Use the optimal model to make classifications on the test set.

**Findings:**
- We have overall imbalanced multiclass dataset and should be careful with the design of an evaluation metric later. Notice that around 60% of the survey respondents have average compensation in the range of $0-10,000. And the observation matches with the common sense that higher salary level is mainly made up of people with longer experience in writing code.

- After applying the Lasso regression and Chi-square test, we keep 30 features & 15 classes in total and notice that the factor geographic region/country has large impact on the compensation level.

-  We define the ordinal multi-class logistic regression algorithm in a function called ordmulti and evaluate the model performance based on F1 score. After model tuning and choice of optimal model, we examine the true target values and predictions on testing data as below. We conclude that the model is biased towards one class and somehow the testing data get more samples of the biased class and achiever higher score. We could further weigh classes inversely proportional to their frequency using random forest or try SMOTE oversampling to improve the performance.

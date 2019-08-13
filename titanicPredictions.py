#%%
# Packages and models
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

import sklearn
import scipy

from scipy.stats import spearmanr

from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, SGDClassifier

from pylab import rcParams

#%%
# Figure parameters
sb.set_style('whitegrid')
rcParams['figure.figsize'] = 5,4

#%%
# Obtaining raw data and describing
test_Address = "C:/Users/Marek/Desktop/Python/Project/Titanic Project/Titanic/Titanic data/test.csv"
train_Address = "C:/Users/Marek/Desktop/Python/Project/Titanic Project/Titanic/Titanic data/train.csv"
survive_address = "C:/Users/Marek/Desktop/Python/Project/Titanic Project/Titanic/Titanic data/gender_submission.csv"

test_raw = pd.read_csv(test_Address)
train_raw = pd.read_csv(train_Address)
test_survive = pd.read_csv(survive_address)

test_raw.columns = ["Passenger ID",'Ticket Class', 'Name', 'Sex','Age','No. of Siblings/Spouses Aboard', 'No. of Parents/Children Aboard','Ticket','Passenger Fair','Cabin Number','Port of Embark']
train_raw.columns = ["Passenger ID",'Survived','Ticket Class', 'Name', 'Sex','Age','No. of Siblings/Spouses Aboard', 'No. of Parents/Children Aboard','Ticket','Passenger Fair','Cabin Number','Port of Embark']
test_survive.columns = ["Passenger ID",'Survived']

print(test_raw.head(5), train_raw.head(5))
print(test_raw.tail(5), train_raw.tail(5))

#%%
# Retrieving Desired Columns 
# Deemed 'Ticket', 'Cabin Number', and Port of Embark.  'Cabin Number' does not provide enough data to use. 'Ticket' number seems to be related to both the ticket class and port of embarking;  
# however, there is too much awkward data (Doubles, different classifications, characters with digits) in the data to worry about at this current moment.
test_extract = test_raw.loc[:,["Passenger ID",'Ticket Class','Sex','Age','No. of Siblings/Spouses Aboard', 'No. of Parents/Children Aboard','Passenger Fair']]
train_extract = train_raw.loc[:,["Passenger ID",'Survived','Ticket Class','Sex','Age','No. of Siblings/Spouses Aboard', 'No. of Parents/Children Aboard','Passenger Fair']]

#%%
# Quick look into the data
print('Train \n \n', test_extract['Age'].describe(),'\n \n Train \n \n', train_extract['Age'].describe())

#%%
# Checking for null values
print('Test \n \n',test_extract.isnull().any(),'\n \n Train \n \n', train_extract.isnull().any())

#%%
# Filling the missing age values with the mean age of the population. Test
test_missing_Age = test_extract['Age'].fillna(test_extract['Age'].mean())
test_extract['Age'] = test_missing_Age
test_extract.isnull().any()

#%%
# Filling the missing age values with the mean age of the population. Train
train_missing_Age = train_extract['Age'].fillna(train_extract['Age'].mean())
train_extract['Age'] = train_missing_Age
train_extract.isnull().any()

#%%
# Filling the the missing fair values with the mean fair of each class.
test_class_fair  = test_extract.loc[:,['Ticket Class','Passenger Fair']]
train_class_fair = train_extract.loc[:,['Ticket Class','Passenger Fair']]

#%%
# Check for missing fairs.
print(
    '1st Class\n',
    test_class_fair[(test_class_fair['Ticket Class'] ==1) & (test_class_fair['Passenger Fair'] ==0)].any(),'\n',
    '\n2nd Class\n',
    test_class_fair[(test_class_fair['Ticket Class'] ==2) & (test_class_fair['Passenger Fair'] ==0)].any(),'\n',
    '\n3rd Class\n',
    test_class_fair[(test_class_fair['Ticket Class'] ==3) & (test_class_fair['Passenger Fair'] ==0)].any()
)

#%%
# Find the indexes of non-missing values. Test
test_missing_fair_indices= { 
    1: test_class_fair[list((test_class_fair['Ticket Class'] ==1) & ((test_class_fair['Passenger Fair'].isnull()) | (test_class_fair['Passenger Fair'] ==False)))]['Passenger Fair'].index,
    2: test_class_fair[list((test_class_fair['Ticket Class'] ==2) & ((test_class_fair['Passenger Fair'].isnull()) | (test_class_fair['Passenger Fair'] ==False)))]['Passenger Fair'].index,
    3: test_class_fair[list((test_class_fair['Ticket Class'] ==3) & ((test_class_fair['Passenger Fair'].isnull()) | (test_class_fair['Passenger Fair'] ==False)))]['Passenger Fair'].index
}
test_missing_fair_indices

#%%
# Calculating the age of non-missing values. Test
test_class_fair_means = {
    1:test_class_fair.iloc[(list((test_class_fair['Ticket Class'] ==1) & (test_class_fair['Passenger Fair'].notnull().index)))].dropna()['Passenger Fair'].mean(),
    2:test_class_fair.iloc[(list((test_class_fair['Ticket Class'] ==2) & (test_class_fair['Passenger Fair'].notnull().index)))].dropna()['Passenger Fair'].mean(),
    3:test_class_fair.iloc[(list((test_class_fair['Ticket Class'] ==3) & (test_class_fair['Passenger Fair'].notnull().index)))].dropna()['Passenger Fair'].mean()
}
test_class_fair_means


#%%
# Find the indexes of non-missing values. Train
train_missing_fair_indices= { 
    1: train_class_fair[list((train_class_fair['Ticket Class'] ==1) & ((train_class_fair['Passenger Fair'].isnull()) | (train_class_fair['Passenger Fair'] ==False)))]['Passenger Fair'].index,
    2: train_class_fair[list((train_class_fair['Ticket Class'] ==2) & ((train_class_fair['Passenger Fair'].isnull()) | (train_class_fair['Passenger Fair'] ==False)))]['Passenger Fair'].index,
    3: train_class_fair[list((train_class_fair['Ticket Class'] ==3) & ((train_class_fair['Passenger Fair'].isnull()) | (train_class_fair['Passenger Fair'] ==False)))]['Passenger Fair'].index
}
train_missing_fair_indices

#%%
# Calculating the age of non-missing values. Train
train_class_fair_means = {
    1:train_class_fair.iloc[(list((train_class_fair['Ticket Class'] ==1) & (train_class_fair['Passenger Fair'].notnull().index)))]['Passenger Fair'].mean(),
    2:train_class_fair.iloc[(list((train_class_fair['Ticket Class'] ==2) & (train_class_fair['Passenger Fair'].notnull().index)))]['Passenger Fair'].mean(),
    3:train_class_fair.iloc[(list((train_class_fair['Ticket Class'] ==3) & (train_class_fair['Passenger Fair'].notnull().index)))]['Passenger Fair'].mean()
}
train_class_fair_means

#%%
# Two main datasets
test  = test_extract.copy()
train = train_extract.copy()

#%%
# Imply means into missing values
test.iloc[list(test_missing_fair_indices[1]),6] =test_class_fair_means[1]
test.iloc[list(test_missing_fair_indices[2]),6] =test_class_fair_means[2]
test.iloc[list(test_missing_fair_indices[3]),6] =test_class_fair_means[3]

train.iloc[list(train_missing_fair_indices[1]),7] =train_class_fair_means[1]
train.iloc[list(train_missing_fair_indices[2]),7] =train_class_fair_means[2]
train.iloc[list(train_missing_fair_indices[3]),7] =train_class_fair_means[3]

#%%
# Check test
test.iloc[list(test_missing_fair_indices[1])]

#%%
# Check train
test_extract.iloc[list(test_missing_fair_indices[1])]

#%%
# Check for nulls
print('Test \n \n',test.isnull().any(),'\n \n Train \n \n', train.isnull().any())

#%%
# Categorical characters to categorical numbers
train['Sex'].replace('male', 0,inplace =True)
train['Sex'].replace('female', 1,inplace =True)
train.drop('Passenger ID',axis =1 ,inplace =True)

test['Sex'].replace('male', 0,inplace =True)
test['Sex'].replace('female', 1,inplace =True)
test.drop('Passenger ID',axis =1 ,inplace =True)

# test_survive.drop('Passenger ID',axis =1 ,inplace =True)

# train
#%%
# Altered database info. Train
train.describe()

#%%
# Altered database info. Test
train.info()

#%%
# Count of passengers by sex
plot_sex_count = sb.countplot(x='Sex',data=train,palette='hls')
plot_sex_count.set(xlabel ='Sex', ylabel='Number of Passenger',xticklabels=['Male','Female'],title ='Count of Passengers by Sex')
plt.show()

#%%
# Survival chance by sex
plot_sex_survival_percentage =sb.barplot(x='Sex',y='Survived',data=train,palette='hls')
plot_sex_survival_percentage.set(xlabel ='Sex', ylabel='Survival Percentage',xticklabels=['Male','Female'],title ='Survival Chance by Sex')
plt.show()

#%%
# Outcome of Passenger count by sex
plot_survived_sex_count = sb.countplot(x='Sex',data=train,hue='Survived', palette='hls')
plot_survived_sex_count.set(xlabel ='Sex', ylabel='Number of Passenger',xticklabels=['Male','Female'],title ='Outcome of Passengers by Sex')
plot_survived_sex_count.legend(['Perished','Survived'])
plt.show()

#%%
# Survival chance by class
plot_class_survival_percentage =sb.barplot(x='Ticket Class',y='Survived',hue='Sex',data=train,palette='hls')
plot_class_survival_percentage.set(xlabel ='Ticket Class', ylabel='Survival Percentage',xticklabels=['1st Class','2nd Class','3rd Class'],title ='Survival Chance by Class')
plt.show()

#%%
# Count by sex and class
plot_class_survival_count =sb.countplot(x='Ticket Class',hue='Sex',data=train,palette='hls')
plot_class_survival_count.set(xlabel ='Ticket Class', ylabel='Number of Passenger',xticklabels=['1st Class','2nd Class','3rd Class'],title ='Count of Passengers by Class')
plot_class_survival_count.legend(['Male','Female'])
plot_class_survival_count.set()
plt.show()
#%%
# Survival count by count and class
plot_class_survival_count =sb.countplot(x='Ticket Class',hue='Survived',data=train,palette='hls')
plot_class_survival_count.set(xlabel ='Ticket Class', ylabel='Number of Passenger',xticklabels=['1st Class','2nd Class','3rd Class'],title ='Count of Passengers by Class')
plot_class_survival_count.legend(['Perished','Survived'])
plot_class_survival_count.set()
plt.show()

#%%
# Age histogram
sb.distplot(train['Age'],kde=False,color='Red',bins=9)

#%%
# Age histogram of survival chance
sb.distplot(train[train['Survived']==0]['Age'],rug=False,color='Red')
sb.distplot(train[train['Survived']==1]['Age'],rug=False,color='DarkGreen')

#%%
# BoxPlot of Cost of Ticket per class
sb.boxplot(x='Ticket Class',y='Passenger Fair',data=train)

#%%
# BoxPlot of Cost of Ticket per class with outliers removed
outlier_removed = train[train['Passenger Fair']< 200]
sb.boxplot(x=outlier_removed['Ticket Class'], y=outlier_removed['Passenger Fair'])

#%%
# Correlation heat map
plt.subplots(figsize=(10,10))
corr_plot =plt.axes()
corr = train.corr()
sb.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap='winter')

#%%
# Spearman coefficient of high correlated parameters
spearmanr_coefficient, p = spearmanr(train['Sex'],train['Survived'])
print('Spearman Rank Correlation Coeffcient %0.3f' %(spearmanr_coefficient))


#%%
# Scaling values
preScale_train = train.iloc[:,1:7].values
y_train = train.iloc[:,0].values
X_train = preprocessing.scale(preScale_train)

X_test = preprocessing.scale(test)

#%% 
# Logisitic Regression
LogReg = LogisticRegression(solver = 'lbfgs')
LogReg.fit(X_train,y_train)

test_prediction_logReg = LogReg.predict(X_test)

accuracy_LogReg = round(LogReg.score(X_train,y_train)*100,2)

#%%
# Decision Tree model
Decision_tree = DecisionTreeClassifier()
Decision_tree.fit(X_train,y_train)

test_prediction_DT = Decision_tree.predict(X_test)

score_decision = round(Decision_tree.score(X_train,y_train)*100,2)

#%%
# K-Nearest Neighbour Model
KNN = KNeighborsClassifier()
KNN.fit(X_train,y_train)

test_prediction_knn = KNN.predict(X_test)

score_knn = round(KNN.score(X_train,y_train)*100,2)

#%%
# Gaussian Model
GaussNB = GaussianNB()
GaussNB.fit(X_train,y_train)

test_prediction_gauss = GaussNB.predict(X_test)

score_gauss = round(GaussNB.score(X_train,y_train)*100,2)

#%%
# Support Vector Classification model
svc = SVC()
svc.fit(X_train,y_train)

test_prediction_svc = svc.predict(X_test)

score_svc = round(svc.score(X_train,y_train)*100,2)

#%%
# Perceptron model
perceptron = Perceptron()
perceptron.fit(X_train,y_train)

test_prediction_perc = perceptron.predict(X_test)

score_perc = round(perceptron.score(X_train,y_train)*100,2)

#%%
# Linear SVC model
linSVC = LinearSVC()
linSVC.fit(X_train,y_train)

test_prediction_linsvc = linSVC.predict(X_test)

score_linsvc = round(linSVC.score(X_train,y_train)*100,2)


#%%
# Stotaic gradient descent model
sgd = SGDClassifier()
sgd.fit(X_train,y_train)


test_prediction_sgd = sgd.predict(X_test)

score_sgd = round(sgd.score(X_train,y_train)*100,2)

#%%
# Random forest model
random_forest = RandomForestClassifier()
random_forest.fit(X_train,y_train)

test_prediction_forest = random_forest.predict(X_test)

score_forest = round(random_forest.score(X_train,y_train)*100,2)

#%%
# Chart of all the models. Comparision
models = pd.DataFrame({
    'Model' : ['Logistic Regression','Decision Tree', 'KNN','Gauss',
    'SVC', 'Perceptron', 'Linear SVC','SGD','Random Forest'] ,
    'Accuracy Score' :[accuracy_LogReg,score_decision,
    score_knn,score_gauss,score_svc,score_perc,
    score_linsvc,score_sgd,score_forest]
})
models.sort_values(by ='Accuracy Score', ascending= False)

#%%

#%%

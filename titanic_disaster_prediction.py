import numpy as np 
import pandas as pd 
import seaborn as sns
import re

from matplotlib import pyplot as plt
from matplotlib import style

import sklearn
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve


#Retrieve the Data

trainDs = pd.read_csv("train.csv")
testDs = pd.read_csv("test.csv")

# Check out the data and check for missing values
print(trainDs.columns.values)
trainDs.info()

pd.set_option('display.expand_frame_repr', False)

"""
We can see after running .info that there are 11 features + the target var (survived)
- 5 Integers
- 5 objects
- 2 floats

PassengerId: Unique ID of the passenger
Pclass : Ticket class
Name: Name of the passenger (Includes titles as well, can be used to determine social class)
Sex: The gender
Age: Age of passenger
SibSp: Number of siblings / spouses aboard
Parch: Number of parents / children aboard
Ticket: Ticket Number
Fare: Passenger fare
Cabin: Cabin number
Embarked: Location where passenger embarked

Survived: Whether or not the passenger survived (target variable)
"""

trainDs.describe()

"""
- We can see that 38% of the passengers in the training set survived the wreckage.
- We can also observe that there is some missing data within the dataset.
-- 
"""

# From first glance at the data, we won't be needing PassengerId since it doesnt correlate with survivability so I will be dropping it from the dataset.

trainDs = trainDs.drop(['PassengerId'], axis = 1)

# We also wont be needing Ticket

trainDs = trainDs.drop(['Ticket'], axis = 1)
testDs = testDs.drop(['Ticket'], axis = 1)

print(trainDs.head(10))

# Checking which data is missing
total = trainDs.isnull().sum().sort_values(ascending=False)
percent1 = trainDs.isnull().sum()/trainDs.isnull().count()*100
percent2 = (round(percent1, 1)).sort_values(ascending=False)
missingData = pd.concat([total, percent2], axis = 1, keys = ['Total', '%'])
print(missingData.head(10))

"""
- We can see that there 77.1% of Cabin entries are missing so we may need to drop it from the dataset.
- 19.9% of the age entries are missing which can be problimatic to fill.
- Embared only has 2 missing values which can be filled. 
"""

"""
- After getting some information on the data. We can now determine which features are useful and which are not. 

- It makes sense to get rid of Ticket and Passenger ID since they have no correlation with survival.

- Titles can possibly be extracted from names later to determine whether or not people of higher class had a higher chance of survival
"""

# Time to visualize some data

facetGrid = sns.FacetGrid(trainDs, row = 'Embarked', height = 5, aspect = 1.5)
facetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = None, order = None, hue_order = None)
facetGrid.add_legend()
plt.show()

"""
- The location from where passengers embarked seems to have a correlation with survival. 
- Men from port C had high survivability compared to the other ports. Females had high survivability if they embarked from port S or Q.
- The ticket class also had a high correlation with survivability. Men and females of ticket class 1 or 2 had a higher survivability than ticket class 3.
-- We can further investiage this by plotting it. 
"""
barPlot = sns.barplot(x = 'Pclass', y = 'Survived', data = trainDs)
plt.show()

# It seems that class 1 and 2 tickets did have a higher survival rate. We can further inspect the data using a histogram

histogram = sns.FacetGrid(trainDs, col='Survived', row = 'Pclass', height = 2, aspect = 1.5)
histogram.map(plt.hist, 'Age', alpha = .5, bins = 20)
histogram.add_legend()
plt.show()

"""
- This confirms that class 3 ticket holders had a much higher mortality rate than the upper class ticket holders.
"""

survived = 'survived'
died = 'died'

fig, axes = plt.subplots(nrows=1, ncols = 2, figsize=(10, 4))
men = trainDs[trainDs['Sex'] == 'male']
women = trainDs[trainDs['Sex'] == 'female']

graph = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins = 18, label = survived, ax = axes[0], kde = False)
graph = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins = 40, label = died, ax = axes[0], kde = False)
graph.legend()
graph.set_title('Male')

graph = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins = 18, label = survived, ax = axes[1], kde = False)
graph = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins = 40, label = died, ax = axes[1], kde = False)
graph.legend()
graph.set_title("Female")
plt.show()

"""
- Females that are between the ages of 14 and 42 have a high chance of survival whereas men between the ages of 18 and 40 had the highest chance of survival.
- The survivability of men between the ages of 5 and 18 is very low. Women between the ages of 7 and 11 had the lowest chance of survival.
- Infants also had a high chance of survivability.

- Creating age groups will be beneficial
"""

# Next we'll look at SibSp and Parch. They can be combined since we essentially just want to check whether a person is alone or has family aboard

dataset = [trainDs, testDs]

for data in dataset:
    data['relatives'] = data['SibSp'] + data['Parch']
    data.loc[data['relatives'] > 0, 'hasFamily'] = 1
    data.loc[data['relatives'] == 0, 'hasFamily'] = 0
    data['hasFamily'] = data['hasFamily'].astype(int)

print("\nWho has family?\n", trainDs['hasFamily'].value_counts())

graph = sns.catplot('relatives', 'Survived', data = trainDs, aspect = 3, kind = "point")
plt.show()

"""
- We can see from the figure above that people that had 1 - 3 relatives aboard had the highest chance of survival whereas having no family or more than 3 relatives had a lower chance
"""

# Now that we have gotten a good look at the data, we need to fill in some of the missing values.
"""
- Cabin has 687 missing values
- Age has 177 missing values
- Embarked has 2 missing values

- After inspecting the data more and doing some research on cruise liners. We can see that each Cabin number start with a letter which refers to the deck and a number pertaining to the room.
-- We can extract the Deck from the passengers whos information we have an convert it to a new feature, give it a numerical value then possibly give the missing passengers a deck.
"""

deck = {"A": 1, "B": 2, "C":3, "D": 4, "E": 5, "F":6, "G": 7, "U": 8}

dataset = [trainDs, testDs]

for data in dataset:
    data['Cabin'] = data['Cabin'].fillna("U0")
    data['Deck'] = data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    data['Deck'] = data['Deck'].map(deck)
    data['Deck'] = data['Deck'].fillna(0)
    data['Deck'] = data['Deck'].astype(int)

trainDs = trainDs.drop(['Cabin'], axis = 1)
testDs = testDs.drop(['Cabin'], axis = 1)
# Checked whether creating the new Deck feature was a success and whether there are still missing values
print(trainDs.head(10))
print("Missing Deck values:", trainDs['Deck'].isnull().sum())

"""
- We have extracted the Deck Letters from the Cabin feature and created a new feature called Deck. 
- We have also filled in the missing Deck variables by assigning Deck 0 to them.
- The data on Cabin is crucial for this model to work so keeping it is important.
"""

# While we are here, we may as well extract titles from Names and convert them to a new feature.

dataset = [trainDs, testDs]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "VIP": 4}

for data in dataset:
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)
    
    # Replacing titles with our own to add to the new Title feature
    data['Title'] = data['Title'].replace(['Countess', 'Lady', 'Capt', 'Don','Dr' 'Col', 'Rev', 'Sir','Major','Dona', 'Jonkheer'], 'VIP')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    # No need to replace Mr. since we are not replacing it with anything
    data['Title'] = data['Title'].map(titles)
    # Not everone has a title so we will add the titleless people to the 0 class.
    data['Title'] = data['Title'].fillna(0)
    data['Title'] = trainDs['Title'].astype(int)

trainDs = trainDs.drop(['Name'], axis = 1)
testDs = testDs.drop(['Name'], axis = 1)




# We now need to deal with the missing Age values. Luckily im taking statistics courses right now so i think we can use the mean age along with std.dev to fill in the missing values.

dataset = [trainDs, testDs]

for data in dataset:
    mean = trainDs['Age'].mean()
    stdDev = trainDs['Age'].std()
    isNull = data['Age'].isnull().sum()

    randomAge = np.random.randint(mean - stdDev, mean + stdDev, size = isNull)

    ageValues = data['Age'].copy()
    ageValues[np.isnan(ageValues)] = randomAge
    data['Age'] = ageValues
    data['Age'] = trainDs['Age'].astype(int)

# We have filled in the missing values, now lets check if theyre all filled in

print("Missing Age values:", trainDs['Age'].isnull().sum())

# Finally, we will deal with the Embarked feature, since there are only 2 missing values we an just replace them with the most common port.

print(trainDs['Embarked'].describe()) # S is the most common port so we will use that value.

dataset = [trainDs, testDs]

for data in dataset:
    data['Embarked'] = data['Embarked'].fillna("S")

print("Missing Embarked values: ", trainDs['Embarked'].isnull().sum())

# We have some more values to convert now such as Sex and Embarked

dataset = [trainDs, testDs]
genders = {"male": 0, "female": 1}

for data in dataset:
    data['Sex'] = data['Sex'].map(genders)

dataset = [trainDs, testDs]
ports = {"S": 0, "C": 1, "Q": 2}

for data in dataset:
    data['Embarked'] = data['Embarked'].map(ports)


#AGE DISTRIBUTION MAY REQUIRE REWORKING


trainDs['AgeGrouping'] = pd.qcut(trainDs['Age'], 7)
print("AgeGrouping qcut \n", trainDs[['AgeGrouping', 'Survived']].groupby(['AgeGrouping'], as_index=False).mean().sort_values(by='AgeGrouping', ascending=True))

dataset = [trainDs, testDs]

for data in dataset:
    # Convert age to integer first
    data['Age'] = data['Age'].astype(int)

    data.loc[data['Age'] <= 17, 'Age'] = 0
    data.loc[(data['Age'] > 17) & (data['Age'] <= 22), 'Age'] = 1
    data.loc[(data['Age'] > 22) & (data['Age'] <= 26), 'Age'] = 2
    data.loc[(data['Age'] > 26) & (data['Age'] <= 30), 'Age'] = 3
    data.loc[(data['Age'] > 30) & (data['Age'] <= 36), 'Age'] = 4
    data.loc[(data['Age'] > 36) & (data['Age'] <= 43), 'Age'] = 5
    data.loc[(data['Age'] > 43) & (data['Age'] <= 80), 'Age'] = 6

# Checking distribution

print("Age distribution \n", trainDs['Age'].value_counts())
trainDs = trainDs.drop(['AgeGrouping'], axis = 1)

# We can group up fares now too

trainDs['FareGrouping'] = pd.qcut(trainDs['Fare'], 6)
print("FareGrouping qcut \n", trainDs[['FareGrouping', 'Survived']].groupby(['FareGrouping'], as_index=False).mean().sort_values(by='FareGrouping', ascending=True))

dataset = [trainDs, testDs]

for data in dataset:
    #print(data['Fare']).value()
    data['Fare'] = data['Fare'].fillna(0)
    data.loc[data['Fare'] <= 7.775, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.775) & (data['Fare'] <= 8.662), 'Fare'] = 1
    data.loc[(data['Fare'] > 8.662) & (data['Fare'] <= 14.454), 'Fare'] = 2
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 26), 'Fare'] = 3
    data.loc[(data['Fare'] > 26) & (data['Fare'] <= 52.369), 'Fare'] = 4
    data.loc[data['Fare'] > 52.369, 'Fare'] = 5
    data['Fare'] = data['Fare'].astype(int)

trainDs = trainDs.drop(['FareGrouping'], axis = 1)
print("Fare distribution \n", trainDs['Age'].value_counts())

dataset = [trainDs, testDs]

print(trainDs.head(10))


# Now we can start building the models.

scores = {}

print("\n\n\n BUILDING MODELS \n\n\n")

xTrain = trainDs.drop("Survived", axis = 1)
yTrain = trainDs["Survived"]
xTest = testDs.drop("PassengerId", axis = 1).copy()

# We still start with SGD with max_iter = 5, 10, and 15

print("----- Stochastic Gradient Descent -----\n")

sgd = linear_model.SGDClassifier(max_iter = 5, tol = None)
sgd.fit(xTrain, yTrain)
yPred = sgd.predict(xTest)
score = sgd.score(xTrain, yTrain)
score = round(score * 100, 2)
print("SGD Score (max_iter = 5): {}%".format(score))
crossVal = cross_val_score(sgd, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

sgd = linear_model.SGDClassifier(max_iter = 10, tol = None)
sgd.fit(xTrain, yTrain)
yPred = sgd.predict(xTest)
score = sgd.score(xTrain, yTrain)
score = round(score * 100, 2)
print("SGD Score (max_iter = 10): {}%".format(score))
crossVal = cross_val_score(sgd, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

sgd = linear_model.SGDClassifier(max_iter = 15, tol = None)
sgd.fit(xTrain, yTrain)
yPred = sgd.predict(xTest)
score = sgd.score(xTrain, yTrain)
score = round(score * 100, 2)
print("SGD Score (max_iter = 15): {}%".format(score))
crossVal = cross_val_score(sgd, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()


# k-NN

print("\n----- k Nearest Neighbor -----\n")

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(xTrain, yTrain)
yPred = knn.predict(xTest)
score = round(knn.score(xTrain, yTrain) * 100, 2)
print("k-NN Score (n_neighbors = 3): {}%".format(score))
crossVal = cross_val_score(knn, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()


knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(xTrain, yTrain)
yPred = knn.predict(xTest)
score = round(knn.score(xTrain, yTrain) * 100, 2)
print("k-NN Score (n_neighbors = 4): {}%".format(score))
crossVal = cross_val_score(knn, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(xTrain, yTrain)
yPred = knn.predict(xTest)
score = round(knn.score(xTrain, yTrain) * 100, 2)
print("k-NN Score (n_neighbors = 5): {}%".format(score))
crossVal = cross_val_score(knn, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

# Random Forest

print("\n----- Random Forest -----\n")

randomForest = RandomForestClassifier(n_estimators= 25)
randomForest.fit(xTrain, yTrain)
yPred = randomForest.predict(xTest)
randomForest.score(xTrain,yTrain)
score = round(randomForest.score(xTrain, yTrain) * 100, 2)
print("Random Forest Score (n_estimator = 25): {}%".format(score))
crossVal = cross_val_score(randomForest, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

randomForest = RandomForestClassifier(n_estimators= 50)
randomForest.fit(xTrain, yTrain)
yPred = randomForest.predict(xTest)
randomForest.score(xTrain,yTrain)
score = round(randomForest.score(xTrain, yTrain) * 100, 2)
print("Random Forest Score (n_estimator = 50): {}%".format(score))
crossVal = cross_val_score(randomForest, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

randomForest = RandomForestClassifier(n_estimators= 75)
randomForest.fit(xTrain, yTrain)
yPred = randomForest.predict(xTest)
randomForest.score(xTrain,yTrain)
score = round(randomForest.score(xTrain, yTrain) * 100, 2)
print("Random Forest Score (n_estimator = 75): {}%".format(score))
crossVal = cross_val_score(randomForest, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()


randomForest = RandomForestClassifier(n_estimators= 100)
randomForest.fit(xTrain, yTrain)
yPred = randomForest.predict(xTest)
randomForest.score(xTrain,yTrain)
score = round(randomForest.score(xTrain, yTrain) * 100, 2)
print("Random Forest Score(n_estimator = 100): {}%".format(score))
crossVal = cross_val_score(randomForest, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

# Decision Tree

print("\n----- Decision Tree -----\n")

decisionTree = DecisionTreeClassifier()
decisionTree.fit(xTrain, yTrain)
yPred = decisionTree.predict(xTest)
score = round(decisionTree.score(xTrain, yTrain) * 100, 2)
print("Decision Tree Score: {}%".format(score))
crossVal = cross_val_score(decisionTree, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

decisionTree = DecisionTreeClassifier(max_depth = 5)
decisionTree.fit(xTrain, yTrain)
yPred = decisionTree.predict(xTest)
score = round(decisionTree.score(xTrain, yTrain) * 100, 2)
print("Decision Tree Score (max_depth = 5): {}%".format(score))
crossVal = cross_val_score(decisionTree, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

decisionTree = DecisionTreeClassifier(max_depth = 10)
decisionTree.fit(xTrain, yTrain)
yPred = decisionTree.predict(xTest)
score = round(decisionTree.score(xTrain, yTrain) * 100, 2)
print("Decision Tree Score (max_depth = 10): {}%".format(score))
crossVal = cross_val_score(decisionTree, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

# Linear Regression

print("\n----- Linear Regression-----\n")

regression = LogisticRegression(solver = 'lbfgs', max_iter = 200)
regression.fit(xTrain, yTrain)
yPred = regression.predict(xTest)
score = round(regression.score(xTrain, yTrain) * 100, 2)
print("Logistic Regression Score (solver = lbfgs, max_iter = 200): {}%".format(score))
crossVal = cross_val_score(regression, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

regression = LogisticRegression(solver = 'lbfgs', max_iter = 300)
regression.fit(xTrain, yTrain)
yPred = regression.predict(xTest)
score = round(regression.score(xTrain, yTrain) * 100, 2)
print("Logistic Regression Score (solver = lbfgs, max_iter = 300): {}%".format(score))
crossVal = cross_val_score(regression, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

regression = LogisticRegression(solver = 'liblinear')
regression.fit(xTrain, yTrain)
yPred = regression.predict(xTest)
score = round(regression.score(xTrain, yTrain) * 100, 2)
print("Logistic Regression Score (solver = liblinear): {}%".format(score))
crossVal = cross_val_score(regression, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

regression = LogisticRegression(solver = 'newton-cg')
regression.fit(xTrain, yTrain)
yPred = regression.predict(xTest)
score = round(regression.score(xTrain, yTrain) * 100, 2)
print("Logistic Regression Score (solver = newton-cg): {}%".format(score))
crossVal = cross_val_score(regression, xTrain, yTrain, scoring = "accuracy", cv = 10)
print("Cross Validation Score = ", crossVal)
print("Mean: {}%".format(round(crossVal.mean() * 100, 2)))
print("Standard Deviation: ", crossVal.std())
print()

# Performing Feature Importance through Random Forest

importance = pd.DataFrame({'feature': xTrain.columns, 'importance': np.round(randomForest.feature_importances_, 3)})
importance = importance.sort_values('importance', ascending=False).set_index('feature')

print(importance.head(15))

importance.plot.bar()
plt.show()

# Parch and hasFamily are not significant to our model so we can drop them and retest.

trainDs = trainDs.drop('hasFamily', axis=1)
testDs = testDs.drop('hasFamily', axis=1)

trainDs = trainDs.drop('Parch', axis=1)
testDs = testDs.drop('Parch', axis=1)

# Random Forest retest with out-of-bag samples

randomForest = RandomForestClassifier(n_estimators=100, oob_score= True)
randomForest.fit(xTrain,yTrain)
yPred = randomForest.predict(xTest)
randomForest.score(xTrain, yTrain)
score = round(randomForest.score(xTrain, yTrain) * 100, 2)
print("\n Random Forest Retest Score: ", round(score, 2,), "%")
print("OOB Score: ", round(randomForest.oob_score_, 4) * 100, "%")

# We still have roughly the same score which is a good thing.
# This means that those features didn't have much impact.
# Also, less features means less over fitting.

# Hyperparameter Tuning

#warnings.warn(CV_WARNING, FutureWarning)

# This block of code is commented out since it takes a long time to run.
# paramGrid = {"criterion": ["gini", "entropy"], "min_samples_leaf": [1, 5, 10, 25, 50, 70], "min_samples_split": [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}

# randomForest = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score= True, random_state=1, n_jobs=-1)

# clf= GridSearchCV(estimator=randomForest, param_grid=paramGrid, n_jobs=-1)
# clf.fit(xTrain,yTrain)

# print(clf.bestparams)

# Testing Random Forest again with new parameters.

randomForest = RandomForestClassifier(criterion="gini", min_samples_leaf=1, min_samples_split=10, n_estimators=100, max_features="auto", oob_score=True, random_state=1, n_jobs=-1)
randomForest.fit(xTrain,yTrain)
yPred = randomForest.predict(xTest)

randomForest.score(xTrain,yTrain)
print("\nTesting Random Forest with new parameters")
print("OOB Score: ", round(randomForest.oob_score_, 4) * 100, "%")


# Evaluation model further.

# Confusion Matrix

predictions = cross_val_predict(randomForest, xTrain, yTrain, cv=3)
print(confusion_matrix(yTrain,predictions))

# Precision and Recall

print("Precision: ", precision_score(yTrain, predictions))
print("Recall: ", recall_score(yTrain, predictions))

# We can combine Precision and Recall into an F-Score

print("F1-Score: ", f1_score(yTrain, predictions))

# Plotting the Precision and Recall

# First, we get the probability of our predictions
yScores = randomForest.predict_proba(xTrain)
yScores = yScores[:,1]

precision, recall, threshold = precision_recall_curve(yTrain, yScores)

def PlotPrecisionAndRecall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0,1])

plt.figure(figsize=(14,7))
PlotPrecisionAndRecall(precision, recall, threshold)
plt.show()

# Testing ROG AUC Curve Plot

# We're computing TPR and FPR
falsePositiveRate, truePositiveRate, threshold = roc_curve(yTrain, yScores)

def PlotRocCurve(falsePositiveRate, truePositiveRate, label=None):
    plt.plot(falsePositiveRate, truePositiveRate, linewidth=2, label=label)
    plt.plot([0,1],[0,1], 'r', linewidth=4)
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14,7))
PlotRocCurve(falsePositiveRate, truePositiveRate)
plt.show()

# Finally, we can attain the ROG AUC Score to check the accuray of our classification model.

rScore = roc_auc_score(yTrain, yScores)
print("ROC AUC Score: ", round(rScore * 100, 2), "%")

# 94.85%!

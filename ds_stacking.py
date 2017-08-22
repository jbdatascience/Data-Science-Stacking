import pandas as pd
import numpy as np

# machine learning
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score


import warnings
warnings.filterwarnings('ignore')

# Load .csv files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]


# Remove unusable cols
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)
combine = [train_df, test_df]

#Find if someone had a cabin
for dataset in combine:
    dataset['HasCabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

combine[0] = combine[0].drop(['Cabin'], axis = 1)
combine[1] = combine[1].drop(['Cabin'], axis = 1)

# Extract the title from a person's name and create its own col
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

for dataset in combine:
    dataset['IsMr'] = 0
    dataset['IsMiss'] = 0
    dataset['IsMrs'] = 0
    dataset['IsMas'] = 0
    dataset['IsRare'] = 0
    dataset.loc[(dataset.Title == 1),  'IsMr'] = 1
    dataset.loc[(dataset.Title == 2), 'IsMiss'] = 1
    dataset.loc[(dataset.Title == 3), 'IsMrs'] = 1
    dataset.loc[(dataset.Title == 4), 'IsMas'] = 1
    dataset.loc[(dataset.Title == 5), 'IsRare'] = 1



combine[0] = combine[0].drop(['Name', 'Title', 'PassengerId'], axis=1)
combine[1] = combine[1].drop(['Name', 'Title'], axis=1)

# Converting sex M/F to 0/1.
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# Use the median age of people with the same gender and ticket class for those with NaN as age
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            #This joins those with same sex and ticket class by age (while keeping out NaN ages)
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)


# make age groups
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[(dataset.SibSp.isnull() | dataset.SibSp == 0) & (dataset.SibSp.isnull() | dataset.Parch == 0), 'IsAlone'] = 1

combine[0] = combine[0].drop(['Parch', 'SibSp'], axis=1)
combine[1] = combine[1].drop(['Parch', 'SibSp'], axis=1)

# Fixing up the Embarked thing
# getting most frequent port
freq_port = combine[0].Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

for dataset in combine:
    dataset['DepS'] = 0
    dataset['DepC'] = 0
    dataset['DepQ'] = 0
    dataset.loc[(dataset.Embarked == 0),  'DepS'] = 1
    dataset.loc[(dataset.Embarked == 1),  'DepC'] = 1
    dataset.loc[(dataset.Embarked == 2),  'DepQ'] = 1


# Fare price
combine[0] = combine[0].fillna(combine[0].dropna().median(), inplace=True)
combine[1] = combine[1].fillna(combine[1].dropna().median(), inplace=True)
combine[0]['FareBand'] = pd.qcut(train_df['Fare'], 4)
combine[1]['FareBand'] = pd.qcut(train_df['Fare'], 4)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

combine[0] = combine[0].drop(['FareBand'], axis=1)
combine[1] = combine[1].drop(['FareBand'], axis=1)

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
    dataset['Fare*Embark'] = dataset.Fare * dataset.Embarked

combine[0] = combine[0].drop(['Embarked'], axis=1)
combine[1] = combine[1].drop(['Embarked'], axis=1)

#print(combine[0].head(10))

# Training
X_all = combine[0].drop("Survived", axis=1)
Y_all = combine[0]["Survived"]

# Splitting up a training and test (validation) set
frac_test = 0.4
frac_test_2 = 0.5
X_train, x_2, y_train, y_2 = train_test_split(X_all, Y_all, test_size = frac_test, random_state=23)
x_train2, x_test, y_train2, y_test = train_test_split(x_2, y_2, test_size = frac_test_2, random_state=23)


# Training different models.

#LR
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
pred_LR = logreg.predict(x_test)
print("LR:", accuracy_score(y_test, pred_LR))

# SVM
svc = SVC()
svc.fit(X_train, y_train)
pred_SVM = svc.predict(x_test)
print("SVM:", accuracy_score(y_test, pred_SVM))

# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=21)
decision_tree.fit(X_train, y_train)
predictions_DT = decision_tree.predict(x_test)
print("Decision Tree:", accuracy_score(y_test, predictions_DT))

# Neural Net
nn = MLPClassifier(hidden_layer_sizes=(13,7,3), random_state=22)
nn.fit(X_train,y_train)
predictions_NN = nn.predict(x_test)
print("Neural Net:", accuracy_score(y_test, predictions_NN))

# KNN
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,y_train)
predictions_KNN = knn.predict(x_test)
print("KNN:", accuracy_score(y_test, predictions_KNN))

# Random Forest
rf = RandomForestClassifier(n_estimators = 10, random_state=25)
rf.fit(X_train,y_train)
predictions_RF = rf.predict(x_test)
print("Random Forest:", accuracy_score(y_test, predictions_RF))

# Next layer model
predictions_LR_train = logreg.predict(x_train2)
predictions_SVM_train = svc.predict(x_train2)
predictions_DT_train = decision_tree.predict(x_train2)
predictions_NN_train = nn.predict(x_train2)
predictions_KNN_train = knn.predict(x_train2)
predictions_RF_train = rf.predict(x_train2)


predictions_LR_train = predictions_LR_train.reshape(-1, 1)
predictions_SVM_train = predictions_SVM_train.reshape(-1, 1)
predictions_DT_train = predictions_DT_train.reshape(-1, 1)
predictions_NN_train = predictions_NN_train.reshape(-1, 1)
predictions_KNN_train = predictions_KNN_train.reshape(-1, 1)
predictions_RF_train = predictions_RF_train.reshape(-1, 1)

next_x_train = np.concatenate((predictions_LR_train, predictions_SVM_train,
                               predictions_DT_train, predictions_NN_train, predictions_KNN_train,
                               predictions_RF_train), axis=1)

predictions_LR = pred_LR.reshape(-1, 1)
predictions_SVM = pred_SVM.reshape(-1, 1)
predictions_DT = predictions_DT.reshape(-1, 1)
predictions_NN = predictions_NN.reshape(-1, 1)
predictions_KNN = predictions_KNN.reshape(-1, 1)
predictions_RF = predictions_RF.reshape(-1, 1)

next_x_test = np.concatenate((predictions_LR, predictions_SVM, predictions_DT, predictions_NN,
                             predictions_KNN, predictions_RF), axis=1)



dt = KNeighborsClassifier(n_neighbors = 10)
dt.fit(next_x_train, y_train2)
#dt.fit(next_x_train, y_2)

final_pred = dt.predict(next_x_test)
print("2nd Layer Final:", accuracy_score(y_test, final_pred))


# Submission
# x_t = combine[1].drop('PassengerId', axis=1)
#
# predictions_LR_test = logreg.predict(x_t)
# predictions_SVM_test = svc.predict(x_t)
# predictions_DT_test = decision_tree.predict(x_t)
# predictions_NN_test = nn.predict(x_t)
# predictions_KNN_test = knn.predict(x_t)
# predictions_RF_test = rf.predict(x_t)
#
# predictions_LR_test = predictions_LR_test.reshape(-1, 1)
# predictions_SVM_test = predictions_SVM_test.reshape(-1, 1)
# predictions_DT_test = predictions_DT_test.reshape(-1, 1)
# predictions_NN_test = predictions_NN_test.reshape(-1, 1)
# predictions_KNN_test = predictions_KNN_test.reshape(-1, 1)
# predictions_RF_test = predictions_RF_test.reshape(-1, 1)
#
# next_x_test = np.concatenate((predictions_LR_test, predictions_SVM_test,
#                                predictions_DT_test, predictions_NN_test, predictions_KNN_test,
#                                predictions_RF_test), axis=1)
#
# submit_pred = dt.predict(next_x_test)
#
# sub = pd.DataFrame({"PassengerId": test_df["PassengerId"], 'Survived': submit_pred})
# sub.to_csv("Titanic_Submission.csv", index=False)



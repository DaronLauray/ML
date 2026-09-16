# %%
#Beta testing and Development: "ML"
#Credit D. Pierre Lauray


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree
from sklearn import preprocessing
from sklearn import utils
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier



# import dependencies and doc

mgm = pd.read_csv('mgm_grand_NBA.csv', encoding='latin1')



# %%
teams=[]

team_values={}
clean_odds={"O/U":" "}

for name in mgm["home_team"]:
    teams.append(name)

teams_clean = list(dict.fromkeys(teams))



#loop team names and assign values to dictionary
i = 0
while i <= len(teams_clean)-1:
    team_values.update({teams_clean[i]:i+1000})
    i+=1
print(team_values)
print(teams_clean)



# %%

from sklearn.ensemble import HistGradientBoostingClassifier
#y,"spread_home_won","spread_away_won","money_home_won","money_away_won"
#X,.replace({True:1, False:0})

X= mgm[["total_over_odds","total_under_odds","home_team","away_team","pregame_odds"]].replace(team_values)
y= mgm[["total_over_won", "total_under_won"]]

#grab column
a = X["pregame_odds"].replace({", O/U": ","}, regex= True)
#split string into list of float strings
d=[]
e=[]
i =0

while i < len(a):
    string = str(a[i])
    x= string.split(",")
    #convert list components to floats over first then under
    d.append(float(x[0].replace("O/U", "")))
    e.append(float(x[-1].replace("O/U", "")))
    i = i+1  
    
#create new column(s)

X["pre_game_over"] = d
X["pre_game_under"] = e

X= X.drop("pregame_odds", axis=1)

#preprocess data by filling in missing values and removing rows with missing target values

X=X.fillna(X.mean())


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.10, random_state=0, stratify=y)

#convert y values to categorical values
model= DecisionTreeClassifier()


#print(y_encoded)
model.fit(X_train,y_train)

#model accuracy checks
predictions = model.predict(X_test)
score= accuracy_score(y_test,predictions)
weighted_score = np.mean(predictions == y_test)

print(weighted_score)
print(score)

#plt.plot(X_test, y_test, "o", label="test data");
#plt.plot(X_test, predictions, "o", label="prediction data");


# %%
# THE START OF MODEL TESTING AND VISUALIZATION FLOW PRACTICE


x = np.linspace(-3,3,100)
rng= np.random.RandomState(45)
y = np.sin(4*x) + x + rng.uniform(size=len(x))
plt.plot(x,y, "o");

X

# %%
#1d array to 2d array
#one column with 100 rows
print('before', x.shape)
#two columns with 100 rows, and the second row has only one feature
X= x[:,np.newaxis]
print('after', X.shape)

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25, random_state=42)


# %%
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#regression model is set

# %%
#set linear equation components i.e. y = m(x) + b

print('weight_coeff: ', regressor.coef_)
print('intercept: ', regressor.intercept_)

min = X.min() * regressor.coef_[0]+ regressor.intercept_
max = X.max() * regressor.coef_[0] + regressor.intercept_

plt.plot([X.min(), X.max()], [min, max])
plt.plot(X_train,y_train, "o");

# %%
#set predictions based on the model
y_pred_train = regressor.predict(X_train)  

plt.plot(X_train, y_train, "o", label="training data");
plt.plot(X_train, y_pred_train, "o", label="prediction data");
plt.plot([X.min(), X.max()], [min, max], label="model line or 'fit' ");
plt.legend(loc="best");

# %%
#Visualize test accuracy

y_pred_test = regressor.predict(X_test)  

plt.plot(X_test, y_test, "o", label="test data");
plt.plot(X_test, y_pred_test, "o", label="prediction data");
plt.plot([X.min(), X.max()], [min, max], label="model line or 'fit' ");
plt.legend(loc="best");

print('true value of test:', regressor.score(X_test, y_test))
print('predicted value of test based on MAE:', mean_absolute_error(y_test, y_pred_test))
print('predicted value of test based on MSE:', mean_squared_error(y_test, y_pred_test))

# %%
# calulate payout based on odds and bet amount!!!
#br= float(input("How much $ are you betting?"))
#neg = float(input("What are the negative odds?"))
#pos = float(input("What are the positive odds?"))

#def neg_odds():
    #os_1 =((100/neg) + 1) * br
    #print(f' The pay out for negative odds is ${os_1}')

#def pos_odds():
    #os_2 = ((pos/100) + 1)* br
    #print(f' The pay out for positive odds is ${os_2}')





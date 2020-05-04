# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
df=pd.read_csv("D:\TEJAS FORMAT\EXCELR ASSIGMENTS\COMPLETED\DECISION TREE\FRAUD\Fraud_check.csv")
fraud=df
fraud.head()

fraud["income"]="<=30000"
fraud.loc[fraud["Taxable.Income"]>=30000,"income"]="Good"
fraud.loc[fraud["Taxable.Income"]<=30000,"income"]="Risky"
fraud["income"].unique()
fraud=fraud.drop(["Taxable.Income"],axis=1)
fraud.rename(columns={"Undergrad":"UG","Marital.Status":"Marital","City.Population":"Population","Work.Experience":"exp"},inplace=True)
colnames=list(fraud.columns)
predictors=colnames[:5]
target=colnames[5]
######Use label encoder########
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
select_columns=["UG","Marital","Urban","income"]
le.fit(fraud[select_columns].values.flatten())
fraud[select_columns]=fraud[select_columns].apply(le.fit_transform)

#########Spilt the data into train and test###########
from sklearn.model_selection import train_test_split
train,test=train_test_split(fraud,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(criterion="entropy")
train_model=model.fit(train[predictors],train[target])
train_pred=model.predict(train[predictors])
pd.crosstab(train[target],train_pred)
####Accuracy##########
acc_train=np.mean(train.income == model.predict(train[predictors]))#######100%

##########Test ##########
test_pred=model.predict(test[predictors])
pd.crosstab(test[target],test_pred)
#######accuracy#########
test_acc=np.mean(test.income==model.predict(test[predictors]))########66%

#######Visualisation############
import pydot
from sklearn.tree import export_graphviz
dot_data=export_graphviz(model,out_file=None,filled=True,rounded=True,special_characters=True)
(graph,)=pydot.graph_from_dot_data(dot_data)

#####pdf
graph.write_pdf('fraud.pdf')
graph.write_png("fraud.png")

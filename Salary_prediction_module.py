
#this is to import numpy as np
import numpy as np

#this is to import pandas as pd
import pandas as pd

#this is to import matplotlib library
import matplotlib.pyplot as plt

#Todo: add comment
salary_data = pd.read_csv('salary_Data.csv')


#create a variable to store the independent column values
x =salary_data.iloc[:,0:1].values

#create a variable to store the dependent column values
y =salary_data.iloc[:,1:2].values

#Todo: add comment
from sklearn.model_selection import train_test_split

#Todo: add comment
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#Todo: add comment
from sklearn.linear_model import LinearRegression

#Todo: add comment
salary_module = LinearRegression()

#Todo: add comment
salary_module.fit(x_train,y_train)

#Todo: add comment
salary_prediction =salary_module.predict(x_test)


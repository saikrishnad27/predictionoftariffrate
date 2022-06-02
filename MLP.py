import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# load the data
df_train= pd.read_excel(r"C:\Users\saikr\Desktop\tariff9.xls")
print(df_train.head())
#converting the multi dimensional data to single dimension
number=LabelEncoder()
df_train['road']=number.fit_transform(df_train['road'].astype('str'))
df_train['demand']=number.fit_transform(df_train['demand'].astype('str'))
df_train['seasonalimpact']=number.fit_transform(df_train['seasonalimpact'].astype('str'))
#cleaning the data
df_train=df_train.fillna(df_train.mean())
print(df_train.head())
#build the correlation matrix
matrix=df_train.corr()
f,ax=plt.subplots(figsize=(16,12))
xr=sns.heatmap(matrix,vmax=0.8,square=True)
print(matrix)
interesting_variables=matrix['tariff'].sort_values(ascending=False)
# Filter out the target variables (tariff) and variables with a low correlation score (v such that -0.5<= v <= 0.5)
interesting_variables = interesting_variables[abs(interesting_variables) >= 0.5]
interesting_variables = interesting_variables[interesting_variables.index != 'tariff']
print(interesting_variables)
# visualize the relationship between the features and the response using scatterplots
cols = interesting_variables.index.values.tolist() + ['tariff']
sns.pairplot(df_train[cols],height=2.5)
print(plt.show())
# splitting the data for testing and training and building the lasso regression model
X = df_train.values[:,0:9]
y = df_train['tariff']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# now we train the model using perceptron
# define no of iterations
iter =1125
alpha = 0.1 # learning rate
# create percpetron instance:
perc = MLPRegressor(max_iter=iter, learning_rate_init=alpha, random_state=0)
# fit the model
perc.fit(X_train,y_train)
y_pred=perc.predict(X_test)
#perfomance evalution metrics
mae=mean_absolute_error(y_pred,y_test)
print("mean absolute error is:")
print(mae)
# Build a plot
plt.scatter(y_pred, y_test)
plt.xlabel('Prediction')
plt.ylabel('Real value')
# Now add the perfect prediction line
diagonal = np.linspace(0, np.max(y_test), 100)
plt.plot(diagonal, diagonal, '-r')
print(plt.show())

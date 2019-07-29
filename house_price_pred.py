import pandas as pd
from pandas import read_csv
import numpy
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plot
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import linear_model                
from sklearn.preprocessing import LabelEncoder 
from sklearn import linear_model 
from sklearn.metrics import mean_absolute_error 

dataset = pd.read_csv("demo_2.csv")
#print(dataset.shape)
#print(dataset.head(10))
print(dataset.describe())

dataset.hist()
#plot.show()

array=dataset.values
X=array[:,:5]
Y=array[:,5] 

# here model will make 7 sets of data (as, seed=7 ) 
validation_size=0.20
seed=7
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)

##scoring part  
scoring='accuracy'
reg = linear_model.LinearRegression()
reg = reg.fit(X_train,Y_train)
predictions=reg.predict(X_test)  # predicts outcome(here house price) for the test dataset 
accuracy = reg.score(X_test, Y_test) #finds the residual sum squared error, i.e; (1-u/v) where u = sum((y_test - y_pred)^2)
print("Accuracy = ",accuracy)        #and v is sum(( y_test - mean(y_test) )^2) 
print("MAE: ", mean_absolute_error(Y_test,predictions))
print("RMSE: ", numpy.sqrt(mean_absolute_error(Y_test,predictions)))
## setting plot style 
plot.style.use('ggplot')
 
## plotting residual errors in training data
plot.scatter(reg.predict(X_train), reg.predict(X_train) - Y_train,
            color = "green", s = 10, label = 'Train data') 
 
## plotting residual errors in test data
plot.scatter(reg.predict(X_test), reg.predict(X_test) - Y_test,
            color = "blue", s = 10, label = 'Test data')
 
## plotting line for zero residual error
plot.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
 
## plotting legend
plot.legend(loc = 'upper right')
 
## plot title
plot.title("Residual error")
 
## function to show plot
plot.show()

fig, ax = plot.subplots()
ax.scatter(Y_test, predictions , edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plot.show()

#predicting Absolute mean error 
error = mean_absolute_error(Y_test , predictions  )  

error = mean_absolute_error(Y_test , predictions  )
print('Mean Absolute error = ' , end='')
print(error)



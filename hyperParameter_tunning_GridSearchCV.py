  
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

  
iris = datasets.load_iris()
x = iris.data
y = iris.target

  
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42)

  
svm = SVC()

  
#giving in dic
param_grid = {'C':[0.1,1,10,100],'kernel':['linear','rbf','poly'],'gamma':['scale','auto',0.1,0.01,0.001]}

  
grid_search = GridSearchCV(svm, param_grid,cv=5,scoring='accuracy')

  
grid_search.fit(x_train,y_train)

  
#best param found
grid_search.best_params_

  
y_prediction = grid_search.predict(x_test)

  
#best accuracy found
accuracy_score(y_test,y_prediction)

  


  




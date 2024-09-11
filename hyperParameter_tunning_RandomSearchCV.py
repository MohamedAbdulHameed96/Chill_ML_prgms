  
from sklearn.datasets import load_digits
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

  
digits = load_digits()
x,y = digits.data,digits.target

  
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42)

  
classifier = RandomForestClassifier()

  
# define the hyperparameter to search
param_grid = {
    'n_estimators': [10,50, 100, 200],
    'max_depth': [None,10,20,30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    #'max_features': ['auto', 'sqrt', 'log2']
}


#create randomizedSearchCV
random_search = RandomizedSearchCV(classifier,param_distributions=param_grid,n_iter=10,cv=5,random_state=42)

#fit the model
random_search.fit(x_train,y_train)

#print the best hyperparameter found
print('Best Hyperparameter:',random_search.best_params_)

#Evaluate the model on the test set
accuracy = random_search.score(x_test,y_test)
print("Test Accuracy:", accuracy)



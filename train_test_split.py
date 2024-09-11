  
import pandas as pd

  
from sklearn.model_selection import train_test_split

  
data = {
    'Feature_1' : [1,2,3,4,5,6,7,8,9,10],
    'Feature_2' : [10,9,8,7,6,5,4,3,2,1],
    'Target' : [0,0,1,0,1,1,0,1,1,0]
}

  
df = pd.DataFrame(data)

  
#Mostly X caps and y small
X =df[['Feature_1','Feature_2']]
y =df[['Target']]

  
#spliting x->train & test, y-> train & test
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25,random_state=42)

  
#checking train and test
len(X_train)
#len(X_test)
#len(y_train)
#len(y_test)

  




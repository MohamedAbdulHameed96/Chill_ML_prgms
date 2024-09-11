  
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold

  
iris = load_iris()  #load dataset
x,y = iris.data, iris.target   # x= feature and y = target

  
k_fold = KFold(n_splits=5,shuffle=True,random_state=42)   # kFold func (spliting,shuffling,default random state value)

  
#150 rows and 4 cols
x.shape

  
#spltting 5 parts equally train and test and shown in result 
for train_index,test_index in k_fold.split(x):
    x_train,x_test = x[train_index],x[test_index]
    y_train,y_test = y[train_index],y[test_index]
    print("train_shape:",x_train.shape)
    print("test_shape:",x_test.shape)

  


  


  




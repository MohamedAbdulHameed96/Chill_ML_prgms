 
pip install scikit-learn keras

 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

 
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int) #binary classification: setosa(class 0 ) vs other

 
#split the data training and testing sets
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

 
#Standardize the features
scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)    #learn and transform
X_test = scaler.transform(X_test)         #transform



 
#create a sequential model
model = Sequential()

# Add input layers and first hidden layer to the model
model.add(Dense(units=6, activation='relu', input_dim=4))  # 4 features in the iris dataset

#Add anothe hidden layer
model.add(Dense(units=6, activation='relu'))  

#Add the output layer with sigmoid activation for binary classification
model.add(Dense(units=1, activation='sigmoid'))  

 
#Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentrophy',metrics=['accuracy'])

 
#Display the model summary
model.summary()

 
print(f"X_train type: {type(X_train)}, shape: {X_train.shape}")
print(f"y_train type: {type(y_train)}, shape: {y_train.shape}")

 
#Train the model
model.fit(X_train, y_train, epochs=15, batch_size=8, validation_split=0.2)

 
y_pred =model.predict(X_test)
y_pred = (y_pred > 0.5)

#Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test,y_test)
print(f'Test loss:{loss:.4f},Test Accuracy:{accuracy:.4f}')



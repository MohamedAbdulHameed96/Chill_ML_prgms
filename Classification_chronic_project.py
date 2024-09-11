
import numpy as np
import pandas as pd


df_data = pd.read_csv('kidney_disease.csv')

df_data.shape


df_data.info()


df_data.head()

df_data.drop('id',axis=1,inplace=True)


df_data.head(3)


df_data.describe()


#data preprocessing starts
#renaming col names
df_data.columns = ["Age","Blood Pressure","Specific Gravity","Albumin","Sugar","Red Blood Cells","Pus Cell",
                   "Pus Cell Clumps","Bacteria","Blod Glucose Random","Blood Urea","Serum Creatinine","Sodium",
                   "Potassium","Hemoglobin","Packed Cell Volume","White Blood Cell Count","Red Blood Cell Count",
                   "Hypertension","Diabetes Mellitus","Coronary Artery Disease","Appetite","Pedal Edema",
                   "Anemia","Class"]


df_data.head()


#Adding col with array 
text_columns = ["Packed Cell Volume","White Blood Cell Count","Red Blood Cell Count"]
for i in text_columns:
    print(f"{i}:{df_data[i].dtype}")



#chaning to obj to numeric col
def convert_text_to_numeric(df_data,column):
    df_data[column] = pd.to_numeric(df_data[column], errors='coerce')  #check pandas parameter doc

for column in text_columns:
    convert_text_to_numeric(df_data,column)
    print(f"{column}: {df_data[column].dtype}")


#fetching missing values
missing = df_data.isnull().sum()
missing[missing>0].sort_values(ascending=False).head(20)


#filling missing values mean,median,mode depends upon the usecase
#mean can do numerical col values, but categorical value->mode is used 
def mean_value_imputation(df_data,column):
    mean_value = df_data[column].mean()
    df_data[column].fillna(value=mean_value, inplace= True)

def mode_value_imputation(df_data,column):
    mode = df_data[column].mode()[0]   #should be mention index because it creating 
    df_data[column]= df_data[column].fillna(mode)    #ippadiyum panala 


 
df_data.columns

 
#using mean and mode for list comprehension
num_cols = [col for col in df_data.columns if df_data[col].dtype !='object']

#filling mean value in missing value
for col_name in num_cols:
    mean_value_imputation(df_data, col_name)

 
#Categorical col
cat_cols = [col for col in df_data.columns if df_data[col].dtype =='object']

#filling mode value in missing value
for col_name in cat_cols:
    mode_value_imputation(df_data, col_name)

 
#fetching missing values
missing = df_data.isnull().sum()
missing[missing>0].sort_values(ascending=False).head(20)

 
df_data.head()

 
#Fetching duplicates in categorical values
print(f"Diabetes Mellitus :- {df_data['Diabetes Mellitus'].unique()}")
print(f"Coronary Artery Disease :- {df_data['Coronary Artery Disease'].unique()}")
print(f"Class :- {df_data['Class'].unique()}")

 
df_data['Diabetes Mellitus'] = df_data['Diabetes Mellitus'].replace(to_replace={' yes':'yes','\tno':'no','\tyes':'yes'})
df_data['Coronary Artery Disease'] = df_data['Coronary Artery Disease'].replace(to_replace={'\tno':'no'})
df_data['Class'] = df_data['Class'].replace(to_replace={'ckd\t':'ckd','notckd':'not ckd'})

 
print(f"Diabetes Mellitus :- {df_data['Diabetes Mellitus'].unique()}")
print(f"Coronary Artery Disease :- {df_data['Coronary Artery Disease'].unique()}")
print(f"Class :- {df_data['Class'].unique()}")

 
df_data.head()

 
#feature Encoding 
#Changing categorical values to numerical values
#large category means using the recommended technique, here there is two category -> mapping techinque
df_data['Class'] = df_data['Class'].map({'ckd': 1,'not ckd':0})
df_data['Red Blood Cells'] = df_data['Red Blood Cells'].map({'normal': 1,'abnormal':0})
df_data['Pus Cell'] = df_data['Pus Cell'].map({'normal': 1,'abnormal':0})
df_data['Pus Cell Clumps'] = df_data['Pus Cell Clumps'].map({'present': 1,'notpresent':0})
df_data['Bacteria'] = df_data['Bacteria'].map({'present': 1,'notpresent':0})
df_data['Hypertension'] = df_data['Hypertension'].map({'yes': 1,'no':0})
df_data['Diabetes Mellitus'] = df_data['Diabetes Mellitus'].map({'yes': 1,'no':0})
df_data['Coronary Artery Disease'] = df_data['Coronary Artery Disease'].map({'yes': 1,'no':0})
df_data['Appetite'] = df_data['Appetite'].map({'good': 1,'poor':0})
df_data['Pedal Edema'] = df_data['Pedal Edema'].map({'yes': 1,'no':0})
df_data['Anemia'] = df_data['Anemia'].map({'yes': 1,'no':0})


 
df_data.head(5)

 
#finding correlation
import matplotlib.pyplot as plt
import seaborn as sns

 
plt.figure(figsize=(15,8))
sns.heatmap(df_data.corr(), annot=True, linewidths=0.5)
plt.show()

 
target_corr =df_data.corr()['Class'].abs().sort_values(ascending=False)[1:]
target_corr

 
df_data['Class'].value_counts()

 
from sklearn.model_selection import train_test_split

 
X = df_data.drop('Class',axis=1)
y =df_data["Class"]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25,random_state=25)

print(f" 'X_train' shape : {X_train.shape}")
print(f" 'X_test' shape : {X_test.shape}")

 
from sklearn.tree import DecisionTreeClassifier

dct = DecisionTreeClassifier()

#fit-> understanding the features and train the data
dct.fit(X_train,y_train)

 
#to verify to test the data
y_pred_dct = dct.predict(X_test)
y_pred_dct

 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

 
models = []
models.append(('Naive Bayes',GaussianNB()))
models.append(('KNN',KNeighborsClassifier(n_neighbors=8)))
models.append(('RandomForestClassifier',RandomForestClassifier()))
models.append(('DecisionTreeClassifier',DecisionTreeClassifier()))
models.append(('SVM',SVC(kernel='linear')))


 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

 
for name,model in models:
    print(name,model)
    print()
    model.fit(X_train,y_train)
    y_pred =model.predict(X_test)
    print("Confusion_Matrix")
    print(confusion_matrix(y_test,y_pred))
    print('\n')
    print("accuracy_score",accuracy_score(y_test,y_pred))
    print('\n')
    print("precision_score",precision_score(y_test,y_pred))
    print('\n')
    print("recall_score",recall_score(y_test,y_pred))
    print('\n')
    print("f1_score",f1_score(y_test,y_pred))
    print('\n')



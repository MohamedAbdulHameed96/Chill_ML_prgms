# %%
pip install pandas

# %%
pip install matplotlib

# %%
import pandas as pd

# %%
#creating series (1d- array) - It supports homogeneous (datatype) with default index
data = [10,20,30,40,50]
series = pd.Series(data)
series

# %%
#creating custom index
custom_index = ['A','B','C','D','E']
series = pd.Series(data, index= custom_index)
series

# %%
#print particular
series['D'] 

# %%
#Add value to everyone
result = series + 3
result

# %%
#datafram is hetetogeneous -> all datatypes are accepted
#can create from files -> numpy array, csv, dict

#creating python dict to DataFrame
data = {
    'Name': ['Abdul','Aki', 'Abdul Basith'],
    'Age' : [27,25,27],
    'city': ['a','b','c']
}

# %%
df = pd.DataFrame(data)
df

# %%
#particular column selection
df['Name']

# %%
#particular row selection
df.loc[0]

# %%
#using csv file
df = pd.read_csv('titanic_train.csv')

# %%
df

# %%
df.head()

# %%
#Specify the print the rows
pd.options.display.max_rows=100

# %%
df.info()

# %%
#see descriptive statistics only work in numerical values
df.describe()

# %%
df.shape

# %%
#check null values
df.isna().sum()

# %%
#drop null values. eg: if any values have nan it will drop
df.dropna()

# %%
#fill null values and affect the values in original dataset
df.fillna(20,inplace=True)

# %%
#specific column to fill the value
df['Age'].fillna(20, inplace=True)

# %%
#using mean, median, mode to fill the null values
#using mean
x=df['Age'].mean()

# %%
#df['Age'].fillna(x,inplace=True) 

# %%
#using mode -> [0] is used because .mode() returns a Series with the mode values
#  we're selecting the first (and in this case, only) value.
x=df['Age'].mode()[0]

# %%
#df['Age'].fillna(x,inplace=True) 

# %%
#find and print the duplicate values
df.drop_duplicates(inplace=True)

# %%
df

# %%
#rename the column
df.rename(columns={'Name': 'FullName'},inplace=True)

# %%
df

# %%
#data Manipulation
#Add new column
#df['Extra Service'] = ['yes','no']

# %%
df

# %%
#Groupby
df_group=df.groupby("Sex")
df_group

# %%
#display groupby
df_group.apply(display)

# %%
df_group.size()

# %%
df_group.nunique()

# %%
df_group['Age'].nlargest()
#df_group['Age].nlargest(2)
#df_group['Age].smallest(2)

# %%
#Aggregate_Function
#All col agg.func()
df.groupby("Sex").Age.agg(['max','min','count'])
#groupby specific column and mean by specific column
#grouped_by = df.groupby("Sex")['Age'].mean()
#grouped_by

# %%
df.groupby(['Survived','Sex']).agg(['count'])

# %%
#Find the relationship between the columns. i.e., correlation
df.corr()


# %%
# display dataframe
print(df)

# %%
# correlation between column 1 and column2
print(df['Survived'].corr(df['Sex']))

# %%
#Convert the column to float
#df['Sex'] = df['Sex'].astype(float)


# %%
# correlation between column 2 and column3
print(data['column2'].corr(data['column3']))
 


# %%
# correlation between column 1 and column3
print(data['column1'].corr(data['column3']))

# %%
df

# %%




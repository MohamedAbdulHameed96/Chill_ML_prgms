  
#no natural order na onehotencoder
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


  
#both the things has h=
data = {
    'fruits' : ['apple','banana','mango','kiwi'],
    'colors' : ['red','yellow','yellow', 'green']
}

  
df =pd.DataFrame(data)

  
encoder = OneHotEncoder()

  
encoded_colors = encoder.fit_transform(df[['colors']]).toarray()

  
encoded_df =pd.DataFrame(encoded_colors,columns=encoder.get_feature_names_out(['colors']))


  
df_encoded =pd.concat([df,encoded_df],axis=1)

  
df_encoded

  




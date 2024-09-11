  
#label encoding for naturally order values
import pandas as pd
from sklearn.preprocessing import LabelEncoder

  
data = {
    'item' : ['shirt', 'jeans', 'dress','t-shirt', 'skirt'],
    'color': ['blue','black','red','white','yellow'],
    'size' : ['medium','large','small','medium','large']
}

  
df = pd.DataFrame(data)

  
#creating label encoder
encoder = LabelEncoder()

  
 #categorical to numerical
df['encoder_size'] =encoder.fit_transform(df['size'])

  
df

  




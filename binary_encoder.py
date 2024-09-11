
import pandas as pd
from category_encoders import BinaryEncoder


data = {
    'Device' : ['laptop','tv','smartphone','microwave','tablet'],
    'Portable' : ['yes','no','yes','no','yes']
}                                                                                                                                       


df = pd.DataFrame(data)


encoder = BinaryEncoder(cols=['Portable'])


df_encoded = encoder.fit_transform(df)


df_encoded



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

with open('C:\Users\Arenter\Documents\GitHub\Recommendation-Systems\heroku_deploy\meta_Grocery_and_Gourmet_Food.json', 'r') as f:
    data = f.readlines()

# remove the trailing "\n" from each line
data = map(lambda x: x.rstrip(), data)
data_json_str = "[" + ','.join(data) + "]"

# now, load it into pandas
food_dict = pd.read_json(data_json_str)

food_dict.fillna('NaN',inplace = True)

food_dict.to_csv('food_dict.csv', index=False))
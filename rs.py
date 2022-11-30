import pandas as pd
import numpy as np

df = pd.DataFrame()

df = df.append({'id':1, 'user_id':0, 'item_id':0, 'rating': 1, 'timestamp':1}, ignore_index=True)
df = df.append({'id':2, 'user_id':0, 'item_id':1, 'rating': 1, 'timestamp':1}, ignore_index=True)
df = df.append({'id':3, 'user_id':1, 'item_id':1, 'rating': 1, 'timestamp':1}, ignore_index=True)
df = df.append({'id':4, 'user_id':1, 'item_id':2, 'rating': 1, 'timestamp':1}, ignore_index=True)
df = df.append({'id':5, 'user_id':2, 'item_id':4, 'rating': 1, 'timestamp':1}, ignore_index=True)
df = df.append({'id':6, 'user_id':2, 'item_id':5, 'rating': 1, 'timestamp':1}, ignore_index=True)
df = df.append({'id':7, 'user_id':3, 'item_id':3, 'rating': 1, 'timestamp':1}, ignore_index=True)
df = df.append({'id':8, 'user_id':3, 'item_id':4, 'rating': 1, 'timestamp':1}, ignore_index=True)
df = df.append({'id':9, 'user_id':4, 'item_id':6, 'rating': 1, 'timestamp':1}, ignore_index=True)

df['id'] = df['id'].astype('int')
df['user_id'] = df['user_id'].astype('int')
df['item_id'] = df['item_id'].astype('int')
df['rating'] = df['rating'].astype('int')
df['timestamp'] = df['timestamp'].astype('int')

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

ratings = np.zeros((n_users, n_items))

for row in df.itertuples():
    ratings[row[0]-1, row[1]-1] = row[2]




print(ratings)
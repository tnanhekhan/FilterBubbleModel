from collections import defaultdict
from surprise import SVD
from surprise import Dataset
import pandas as pd
from surprise import Reader


df = pd.DataFrame()

df = df.append({'id':1, 'user_id':1, 'item_id':1, 'rating': 5, 'timestamp':1}, ignore_index=True)
df = df.append({'id':2, 'user_id':1, 'item_id':2, 'rating': 5, 'timestamp':1}, ignore_index=True)
df = df.append({'id':3, 'user_id':2, 'item_id':2, 'rating': 5, 'timestamp':1}, ignore_index=True)
df = df.append({'id':4, 'user_id':2, 'item_id':3, 'rating': 5, 'timestamp':1}, ignore_index=True)
df = df.append({'id':5, 'user_id':3, 'item_id':5, 'rating': 5, 'timestamp':1}, ignore_index=True)
df = df.append({'id':6, 'user_id':3, 'item_id':6, 'rating': 5, 'timestamp':1}, ignore_index=True)
df = df.append({'id':7, 'user_id':4, 'item_id':4, 'rating': 5, 'timestamp':1}, ignore_index=True)
df = df.append({'id':8, 'user_id':4, 'item_id':5, 'rating': 5, 'timestamp':1}, ignore_index=True)
df = df.append({'id':9, 'user_id':5, 'item_id':7, 'rating': 5, 'timestamp':1}, ignore_index=True)

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(0, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# First train an SVD algorithm on the movielens dataset.
def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
print(testset)

top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])

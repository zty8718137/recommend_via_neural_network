# -*- coding:utf-8 -*-

'''
   author: Tianyu Zhong
   Created on 12/2/2018
'''
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import pandas as pd
import random
from Deep_CF.Config import top_k

with open("../save/matrix/movie_matrix.p", "rb") as file1:
    item_matrix = pickle.load(file1)
with open("../save/matrix/user_matrix.p", "rb") as file2:
    user_matrix = pickle.load(file2)
with open("../save/matrix/history.json", "r") as file3:
    history = json.load(file3)
users_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
users = pd.read_table('../ml-1m/users.dat', sep='::', header=None, names=users_title, engine = 'python')
movies_title = ['MovieID', 'Title', 'Genres']
movies = pd.read_table('../ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine = 'python')
nn = NearestNeighbors(n_neighbors=top_k + 1)
nn.fit(item_matrix)

def rating_movie(userid, itemid):
    user_vec = user_matrix[userid-1]
    item_vec = item_matrix[itemid-1]
    res = np.sum(user_vec * item_vec).astype(float)
    print(int(res + 0.5))

# Recommend similar items for user
def get_sim_item(itemid):
    print("Target : ", movies[movies["MovieID"] == itemid].values[0])
    item_vec = item_matrix[itemid-1]
    idx = nn.kneighbors(item_vec.reshape(1,-1), return_distance=False)[0]
    for id in idx:
        if id+1!=itemid : print(movies[movies["MovieID"] == id+1].values[0])

# Recommend for users accordingly
# Find the highest rating item that this user have not seen before
def rec_for_user(userid):
    user_vec = user_matrix[userid - 1]
    rating = np.dot(user_vec, item_matrix.T).reshape(1,-1)
    seen = history[str(userid)]
    cnt = 0
    for idx in np.argsort(-rating):
        idx += 1
        if idx not in seen and cnt < 20:
            print(movies[movies["MovieID"] == idx].values[0])
            cnt += 1

# Recommend also-like items
def rec_also_like(itemid):
    rating = np.dot(item_matrix[itemid-1], user_matrix.T)
    top_user = np.argsort(-rating)[:20]
    item_set = []
    for user in top_user:
        user_vec = user_matrix[user-1]
        rating_user = np.dot(user_vec, item_matrix.T)
        top_item = np.argsort(-rating_user)[:5].tolist()
        item_set += top_item
    item_set = set(item_set)
    if len(item_set) > 20 : rec = random.sample(item_set, 20)
    else: rec = item_set
    for idx in rec:
        print(movies[movies["MovieID"] == idx+1].values[0])

if __name__ == "__main__":
    rating_movie(1, 1)


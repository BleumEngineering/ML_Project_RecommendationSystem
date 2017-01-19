#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:32:11 2017

@author: forest
"""


from crab.models.classes import MatrixPreferenceDataModel
from crab.metrics.pairwise import euclidean_distances

from crab.recommenders.knn.neighborhood_strategies import AllNeighborsStrategy
from crab.recommenders.knn.neighborhood_strategies import NearestNeighborsStrategy
from crab.similarities.basic_similarities import UserSimilarity
from crab.recommenders.knn import UserBasedRecommender

from crab.similarities.basic_similarities import ItemSimilarity
from crab.recommenders.knn import ItemBasedRecommender
from crab.recommenders.knn.item_strategies import ItemsNeighborhoodStrategy

# setup data
movies = {'Marcel Caraciolo': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
 'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
 'The Night Listener': 3.0},
'Luciana Nunes': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
 'You, Me and Dupree': 3.5},
'Leopoldo Pires': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
 'Superman Returns': 3.5, 'The Night Listener': 4.0},
'Lorena Abreu': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
 'The Night Listener': 4.5, 'Superman Returns': 4.0,
 'You, Me and Dupree': 2.5},
'Steve Gates': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
 'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
 'You, Me and Dupree': 2.0},
'Sheldom': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
 'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
'Penny Frewman': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0},
'Maria Gabriela': {}}

# test neighborhood strategy
model = MatrixPreferenceDataModel(movies)
strategy = AllNeighborsStrategy()
neighborhood_users = strategy.user_neighborhood('Lorena Abreu', model)
print('1: AllNeighborsStrategy Test - %s' %neighborhood_users)
print('1: total size - %i' %neighborhood_users.size)

# recommendation by user similarity

user_strategy = NearestNeighborsStrategy()
user_similarity = UserSimilarity(model, euclidean_distances)
user_recsys = UserBasedRecommender(model, user_similarity, user_strategy)
user_recomm_items = user_recsys.recommend('Leopoldo Pires')
print('2: recommendation by user similarity - %s' %user_recomm_items)
print('2: total size - %i' %len(user_recomm_items))

user_recomm_items = user_recsys.recommended_because('Leopoldo Pires', 'Snakes on a Plane', 2)
print('3: recommendate_because by user similarity - %s' %user_recomm_items)
print('3: total size - %i' %len(user_recomm_items))

# Recommendation by item similarity

items_strategy = ItemsNeighborhoodStrategy()
item_similarity = ItemSimilarity(model, euclidean_distances)
item_recsys = ItemBasedRecommender(model, item_similarity, items_strategy)
item_recomm_items = item_recsys.recommend('Leopoldo Pires')
print('4: recommendation by item similarity - %s' %item_recomm_items)
print('4: total size - %i' %len(item_recomm_items))

item_recomm_items = item_recsys.recommended_because('Leopoldo Pires', 'Just My Luck',2)
print('5: recommendate_because by item similarity - %s' %item_recomm_items)
print('5: total size - %i' %len(item_recomm_items))








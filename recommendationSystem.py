#Recommendation system
#April 30, 2017

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#get the movies
data = fetch_movielens(min_rating=4.0)

#print the data we got earlier
print(repr(data['train']))
print(repr(data['test']))

#let's create a model
model = LightFM(loss='warp')

#train the model
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):

	#number of users and movies in the training data
	n_users, n_items = data['train'].shape

	#generate recommendations for each user we input
	for user_id in user_ids:

		#movies the user already liked
		known_positives = data['item_labels'] [data['train'].tocsr()[user_id].indices]

		#the movies our model says the user will like
		score = model.predict(user_id, np.arange(n_items))
		#rank in the order of most liked to least
		top_items= data['item_labels'][np.argsort(-score)]

		#print the results
		print("User %s" %user_id)
		print("     Known positives:")

		for x in known_positives[:3]:
			print("            %s" %x)

		print("     Recommend:")

		for x in top_items[:3]:
			print("            %s" %x)


sample_recommendation(model, data, [3, 25, 45])

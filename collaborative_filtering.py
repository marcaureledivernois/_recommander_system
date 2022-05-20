import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# reading csv files
colnames = ['user_id', 'item_id','rating','timestamp']
ratings =  pd.read_csv('u.data', sep="\t", names=colnames, header=None)
colnames = ['item_id', 'title','release_date','video_release_date','imdb_url','unknown','action','adventure'
            ,'animation','children','comedy','crime','documentary','drama','fantasy','filmnoir','horror'
            ,'musical','mystery','romance','scifi','thriller','war','western']
items =  pd.read_csv('u.item', sep="|", names=colnames, header=None, encoding='latin-1')

ratings_t = ratings.pivot(index='user_id', columns='item_id', values='rating')
user_avg_ratings = ratings_t.mean(axis=1)
ratings_t_centered = ratings_t.sub(user_avg_ratings,axis='index') #centered ratings
ratings_t_centered = ratings_t_centered.fillna(0)

user_sims = pd.DataFrame(cosine_similarity(ratings_t_centered))
user_sims.index = ratings_t_centered.index
user_sims.columns = ratings_t_centered.index
item_sims = pd.DataFrame(cosine_similarity(ratings_t_centered.T))
item_sims.index = ratings_t_centered.columns
item_sims.columns = ratings_t_centered.columns


#user-based : find top 10 most similar users of target user.

k = 10
target_user_idx = 1

film_unseen = [i for i,x in enumerate(ratings_t_centered.loc[target_user_idx,:] == 0, start=1) if x]
recommendations = []

for target_film in film_unseen:
    users_not_having_seen_film = ratings_t_centered.loc[:,target_film] == 0
    user_sims_copy = user_sims.drop(ratings_t_centered[users_not_having_seen_film].index).copy()
    user_sim = pd.DataFrame(user_sims_copy.loc[:,target_user_idx]).sort_values(by=target_user_idx, ascending=False)
    similar_users = user_sim[1:k+1]
    similar_users.loc[:,'weight'] = 1/similar_users[target_user_idx]
    target_film_rating_similar_users = ratings_t_centered.loc[similar_users.index.tolist(),target_film]

    pred_centered_rating = np.sum(target_film_rating_similar_users * similar_users['weight']) / np.sum(similar_users['weight'])
    pred_rating = pred_centered_rating + user_avg_ratings[target_user_idx]

    recommendations.append([target_film, pred_rating])

recommendations = pd.DataFrame(recommendations).sort_values(by=1,ascending=False)
recommendations = recommendations.merge(items[['item_id','title']],how='left',left_on=0,right_on='item_id')


#item-based : find top 10 most similar items of target item

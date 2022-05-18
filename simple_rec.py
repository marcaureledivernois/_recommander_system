#################### SIMPLE RECOMMENDER ########################################
# compute a weighted rating taking into account the number of votes it has accumulated
# IMDB formula
# Weighted Rating  = v/(v+m) * R + m /(v+m) * C
#     v is the number of votes for the movie;
#     m is the minimum votes required to be listed in the chart; hyperparameter, lets say 90 percentile
#     R is the average rating of the movie;
#     C is the mean vote across the whole report.
##################################################################################
import pandas as pd

metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

C = metadata['vote_average'].mean()
m = metadata['vote_count'].quantile(0.90)

q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20)
### Recommander Systems

They are used to predict the "rating" or "preference" that a user would give to an item. 

Broadly, recommender systems can be classified into 3 types:

1. **Simple recommenders**: offer generalized recommendations to every user, based on movie popularity and/or genre. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience. An example could be IMDB Top 250.

2. **Content-based recommenders**: suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person likes a particular item, he or she will also like an item that is similar to it. And to recommend that, it will make use of the user's past item metadata. A good example could be YouTube, where based on your history, it suggests you new videos that you could potentially watch.

3. **Collaborative filtering engines**: See below. They can be user-based (find similar users) or item-based (find similar items).

## Collaborative filtering

These systems are widely used, and they try to predict the rating or preference that a user would give an item-based on past ratings and preferences of similar users. Collaborative filters do not require item metadata like its content-based counterparts.
   1. Memory-based : statistical techniques are applied to the entire dataset to calculate predictions. First, find similar users then calculate rating of item based on rating of similar users.
      1. **User-based**: For a user U, with a set of similar users determined based on rating vectors consisting of given item ratings, the rating for an item I, which hasn’t been rated, is found by picking out N users from the similarity list who have rated the item I and calculating the rating based on these N ratings.
      2. **Item-based**: For an item I, with a set of similar items determined based on rating vectors consisting of received user ratings, the rating by a user U, who hasn’t rated it, is found by picking out N items from the similarity list that have been rated by U and calculating the rating based on these N ratings.
   2. Model-based : Dimensionality reduction. Reduce or compress the large but sparse user-item matrix. If the matrix is mostly empty, reducing dimensions can improve the performance of the algorithm in terms of both space and time. You can use various methods like matrix factorization or autoencoders to do this.
      1. **Matrix-factorization**
      2. **Autoencoders**
   
## When to chose user-based or item-based ?

Item-based collaborative filtering was developed by Amazon. In a system where there are more 
users than items, item-based filtering is faster and more stable than user-based. 
It is effective because usually, the average rating received by an item doesn’t change 
as quickly as the average rating given by a user to different items. It’s also known to 
perform better than the user-based approach when the ratings matrix is sparse.

Although, the item-based approach performs poorly for datasets with browsing or entertainment 
related items such as MovieLens, where the recommendations it gives out seem very obvious to the 
target users. Such datasets see better results with matrix factorization techniques, or with hybrid 
recommenders that also take into account 
the content of the data like the genre by using content-based filtering.

### Distance

Cosine distance takes the angle into account, which is preferable to euclidean distance because of **tough raters**.
Also, it is good practice to remove the mean of rating of every user to get rid of personal rating toughness. The cosine 
of the angle between the adjusted vectors is called the **centered cosine**. This also has the advantage of being able
to fill the missing value with the average rating of every user. If we did not adjust the vectors, then filling the missing value
with the average rating would make similar users dissimilar.

### Aggregate Rating of top-k similar users

Let's say k=10, the top 3 users might be very similar, and the remaining 7 are not that similar.
So when we aggregate their ratings, we wan't to weigh in the similarity of every user.

Let S be the similarity factor (usually the inverse of the cosinedistance), R_u the rating of user u.

![equation](https://latex.codecogs.com/svg.image?R&space;=&space;\sum_{u=1}^n&space;R_u&space;*&space;S_u&space;/&space;\sum_{u=1}^n&space;S_u)

## Credits 

[DataCamp](https://www.datacamp.com/tutorial/recommender-systems-python)

[RealPython](https://realpython.com/build-recommendation-engine-collaborative-filtering/#the-dataset)
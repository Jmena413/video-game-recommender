import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_user_item_matrix(df, user_col, item_col, rating_col):
    user_item_matrix = df.pivot_table(index=user_col, columns=item_col, values=rating_col).fillna(0)
    return user_item_matrix

def compute_similarity(matrix):
    similarity_matrix = cosine_similarity(matrix)
    return pd.DataFram(similarity_matrix, index=matrix.index, columns=matrix.index)

def recommend_items(similarity_matrix, user_item_matrix, user, num_recommendations=5):
    user_ratings = user_item_matrix.loc[user]
    similar_users = similarity_matrix[user].sort_values(ascending=False).index[1:]
    weighted_sum = user_item_matrix.loc[similar_users].T.dot(similarity_matrix[user][similar_users])
    recommendations = (weighted_sum / similarity_matrix[user][similar_users].sum()).sort_values(ascending=False)
    recommendations = recommendations[~recommendations.index.isin(user_ratings[user_ratings > 0].index)]

    return recommendations.head(num_recommendations)
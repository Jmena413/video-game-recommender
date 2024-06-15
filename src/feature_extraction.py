from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

def extract_tfidf_features(text_series, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(text_series)
    tfidf_features = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    return tfidf_features, vectorizer

def one_hot_encode_features(df, column):
    encoder = OneHotEncoder()
    encoded_matrix = encoder.fit_transform(df[[column]])
    encoded_df = pd.DataFrame(encoded_matrix.toarray(), columns=encoder.get_feature_names_out())
    return encoded_df

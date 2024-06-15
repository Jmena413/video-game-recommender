import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Drop rows with missing values
    df = df.dropna(subset=['user_score', 'game', 'genre', 'platform'])

    #encode catergorical data
    label_encoder = LabelEncoder()
    df['genre'] = label_encoder.fit_transform(df['genre'])
    df['platform'] = label_encoder.fit_transform(df['platform'])

    # normalize numerical features if  necessary

    df['user_score'] = df['user_score'].astype(float)

    return df
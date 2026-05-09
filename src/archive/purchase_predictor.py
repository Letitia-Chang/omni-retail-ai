import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

class PurchasePredictor:
    def __init__(self, model_path):
        self.model = XGBClassifier()
        self.model.load_model(model_path)

        # Encoders
        self.article_encoder = LabelEncoder()
        self.colour_encoder = LabelEncoder()
        self.season_encoder = LabelEncoder()
        self.user_type_encoder = LabelEncoder()

    def encode_features(self, df):
        df = df.copy()

        df["articleType"] = self.article_encoder.fit_transform(df["articleType"])
        df["baseColour"] = self.colour_encoder.fit_transform(df["baseColour"])
        df["season"] = self.season_encoder.fit_transform(df["season"])
        df["user_type"] = self.user_type_encoder.fit_transform(df["user_type"])

        return df

    def decode_features(self, df):
        df = df.copy()

        df["articleType"] = self.article_encoder.inverse_transform(df["articleType"])
        df["baseColour"] = self.colour_encoder.inverse_transform(df["baseColour"])
        df["season"] = self.season_encoder.inverse_transform(df["season"])
        df["user_type"] = self.user_type_encoder.inverse_transform(df["user_type"])

        return df

    def predict(self, df):
        df = self.encode_features(df)

        feature_cols = ["articleType", "baseColour", "season", "price", "user_type", "click_count"]
        X = df[feature_cols]

        df["purchase_prob"] = self.model.predict_proba(X)[:, 1]
        df["purchase_pred"] = (df["purchase_prob"] >= 0.5).astype(int)

        df = self.decode_features(df)

        return df
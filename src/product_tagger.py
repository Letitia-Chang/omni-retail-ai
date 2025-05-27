import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ProductTagger:
    def __init__(self, base_dir, image_dir):
        self.base_dir = base_dir
        self.image_dir = image_dir

        # Load models
        self.article_model = load_model(os.path.join(base_dir, "models", "type_model_mobilenet.h5"))
        self.colour_model = load_model(os.path.join(base_dir, "models", "colour_model_mobilenet.h5"))
        self.season_model = load_model(os.path.join(base_dir, "models", "season_model_mobilenet.h5"))

        # Load class label maps
        self.article_classes = pd.read_csv(os.path.join(base_dir, "data", "processed", "type_tag_classes.csv")).squeeze().tolist()
        self.colour_classes = pd.read_csv(os.path.join(base_dir, "data", "processed", "colour_tag_classes.csv")).squeeze().tolist()
        self.season_classes = pd.read_csv(os.path.join(base_dir, "data", "processed", "season_tag_classes.csv")).squeeze().tolist()

    def preprocess_image(self, img_path, target_size=(128, 128)):
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict_labels(self, img_path):
        img = self.preprocess_image(img_path)

        pred_article = self.article_model.predict(img)[0]
        pred_colour = self.colour_model.predict(img)[0]
        pred_season = self.season_model.predict(img)[0]

        article_label = self.article_classes[np.argmax(pred_article)]
        colour_label = self.colour_classes[np.argmax(pred_colour)]
        season_label = self.season_classes[np.argmax(pred_season)]

        return article_label, colour_label, season_label

    def filter_missing_images(self, df):
        df["has_image"] = df["id"].astype(str).apply(
            lambda img_id: os.path.exists(os.path.join(self.image_dir, f"{img_id}.jpg"))
        )

        missing = df[~df["has_image"]]
        if not missing.empty:
            print("❗ Missing Images:")
            for img_id in missing["id"]:
                print(f"Image {img_id} not found")

        return df[df["has_image"]].drop(columns=["has_image"])

    def predict_on_dataframe(self, df):
        df = self.filter_missing_images(df)

        article_preds, colour_preds, season_preds = [], [], []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Tagging Products"):
            img_id = str(row["id"])
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")

            if not os.path.exists(img_path):
                article_preds.append(None)
                colour_preds.append(None)
                season_preds.append(None)
                continue

            article, colour, season = self.predict_labels(img_path)
            article_preds.append(article)
            colour_preds.append(colour)
            season_preds.append(season)

        df["articleType"] = article_preds
        df["baseColour"] = colour_preds
        df["season"] = season_preds

        return df

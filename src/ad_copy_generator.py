import requests
import re
import pandas as pd
from tqdm import tqdm

class AdGenerator:
    def __init__(self, model_name, api_url, headers):
        self.model_name = model_name
        self.api_url = api_url
        self.headers = headers

    # Build prompt for a product row
    def build_prompt(self, row):
        return (
            f"Write a short ad line for this product.\n"
            f"Product: {row['productDisplayName']}.\n"
            f"Type: {row['articleType']}, Color: {row['baseColour']}, Season: {row['season']}.\n"
        )

    # Query the model using the API
    def call_model(self, prompt):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.model_name
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        return response.json()["choices"][0]["message"]['content']

    # Clean ad output
    def clean_ad(self, ad):
        try:
            # Extract ad copy content
            modified = re.findall(r'\*\*\"(.*?)\"\*\*', ad)[0]
        except IndexError:
            modified = ad.strip()
        return modified.replace('"', '').replace("'", "")

    # Main function: generate ads for all products
    def generate_ads(self, df):
        prompts = df.apply(self.build_prompt, axis=1)
        ads = []

        for prompt in tqdm(prompts, desc="📝 Generating Ads"):
            raw_ad = self.call_model(prompt)
            cleaned_ad = self.clean_ad(raw_ad)
            ads.append(cleaned_ad)

        df["ad_copy"] = ads
        return df
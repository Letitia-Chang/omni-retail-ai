import os
import pandas as pd

class PromotionRanker:
    def __init__(self, df, base_dir, user_types=None):
        self.df = df.copy()
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, "data", "processed")
        self.user_types = user_types or ["budget", "fashionista", "color_lover"]
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_scores(self):
        self.df["inventory_weight"] = self.df["inventory"] / self.df["inventory"].max()
        self.df["promotion_score"] = self.df["purchase_prob"] * self.df["inventory_weight"]
        self.df = self.df.sort_values("promotion_score", ascending=False).reset_index(drop=True)
        return self.df

    def export_segment_campaigns(self, top_n=10):
        for segment in self.user_types:
            segment_df = self.df.copy()
            segment_df["user_type"] = segment

            file_path = os.path.join(self.output_dir, f"campaign_{segment}.csv")
            segment_df.head(top_n).to_csv(file_path, index=False)
            print(f"📤 Saved top {top_n} for user type '{segment}' → {file_path}")

    def export_rankings(self, top_n=50):
        all_path = os.path.join(self.output_dir, "final_campaign_ranking.csv")
        top_path = os.path.join(self.output_dir, "top_promoted_products.csv")

        self.df.to_csv(all_path, index=False)
        self.df.head(top_n).to_csv(top_path, index=False)

        print(f"📊 Full ranked product list saved to → {all_path}")
        print(f"⭐ Top {top_n} promoted products saved to → {top_path}")

    def run_full_strategy(self, top_n=50):
        self.compute_scores()
        self.export_segment_campaigns(top_n)
        self.export_rankings(top_n)
        print(f"✅ Promotion strategy complete. All files in → {self.output_dir}")
        return self.df

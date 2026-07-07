import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR, RAG_MODEL_DIR
from src.rag.product_index import build_product_index, save_product_index


def main():
    print("Loading enriched product catalog...")

    articles = pd.read_csv(PROCESSED_DATA_DIR / "articles_enriched.csv")

    print(f"Embedding {len(articles)} products and building FAISS index...")

    index, article_ids, vectorizer, svd = build_product_index(articles)

    save_product_index(index, article_ids, vectorizer, svd, RAG_MODEL_DIR)

    print(f"Saved product index ({index.ntotal} vectors) to: {RAG_MODEL_DIR}")


if __name__ == "__main__":
    main()

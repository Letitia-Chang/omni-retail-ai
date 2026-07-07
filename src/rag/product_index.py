import faiss
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.features.product_enrichment import create_product_context

INDEX_FILENAME = "product_catalog.faiss"
ARTICLE_IDS_FILENAME = "product_catalog_article_ids.pkl"
VECTORIZER_FILENAME = "product_catalog_tfidf.pkl"
SVD_FILENAME = "product_catalog_svd.pkl"


def embed_texts(texts, vectorizer: TfidfVectorizer, svd: TruncatedSVD) -> np.ndarray:
    """TF-IDF -> SVD (LSA) -> L2-normalized dense vectors, for cosine similarity via inner product."""
    tfidf = vectorizer.transform(texts)
    dense = svd.transform(tfidf)
    return normalize(dense).astype("float32")


def build_product_index(articles: pd.DataFrame, n_components: int = 128):
    """Embed every article's product context (TF-IDF + SVD) and build a FAISS
    index over the resulting dense vectors.

    Returns (index, article_ids, vectorizer, svd) — article_ids[i] is the
    article_id for the vector at row i of the index, since FAISS only stores
    vectors, not labels; vectorizer/svd are needed again at query time to
    embed the incoming search text the same way.
    """
    contexts = articles.apply(create_product_context, axis=1).tolist()

    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
    tfidf = vectorizer.fit_transform(contexts)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = normalize(svd.fit_transform(tfidf)).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    article_ids = articles["article_id"].tolist()

    return index, article_ids, vectorizer, svd


def save_product_index(index, article_ids, vectorizer, svd, model_dir) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(model_dir / INDEX_FILENAME))
    joblib.dump(article_ids, model_dir / ARTICLE_IDS_FILENAME)
    joblib.dump(vectorizer, model_dir / VECTORIZER_FILENAME)
    joblib.dump(svd, model_dir / SVD_FILENAME)


def load_product_index(model_dir):
    index = faiss.read_index(str(model_dir / INDEX_FILENAME))
    article_ids = joblib.load(model_dir / ARTICLE_IDS_FILENAME)
    vectorizer = joblib.load(model_dir / VECTORIZER_FILENAME)
    svd = joblib.load(model_dir / SVD_FILENAME)
    return index, article_ids, vectorizer, svd


def retrieve_similar_products(
    query_text: str,
    index,
    article_ids,
    articles: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    svd: TruncatedSVD,
    top_k: int = 5,
) -> pd.DataFrame:
    """Embed `query_text` and return the top_k nearest articles by cosine similarity."""
    query_embedding = embed_texts([query_text], vectorizer, svd)

    scores, indices = index.search(query_embedding, top_k)

    matched_ids = [article_ids[i] for i in indices[0]]
    matches = articles.set_index("article_id").loc[matched_ids].reset_index()
    matches["similarity_score"] = scores[0]

    return matches

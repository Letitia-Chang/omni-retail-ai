# 🛍️ OmniRetail AI

**An End-to-End AI System for Smarter Retail Marketing**

> Automatically tag products, generate ad copy, predict purchase intent, and prioritize promotions based on real-time inventory — all in one intelligent pipeline.


## ❗ Motivation: Why Build OmniRetail AI?

Modern retail teams face growing pressure to scale marketing efforts across thousands of products, channels, and customers — but the tooling is outdated and disconnected.

### ⚠️ Problem 1: Manual e-commerce product tagging & campaign generation

- **Current workflow**: Marketing teams manually pull product descriptions and images from ERP systems, categorize them (e.g., by color, type, season), and write promotional copy one-by-one.
- **Pain points**:
  - Tedious and repetitive — especially for large inventories or multi-brand campaigns
  - Inconsistent copy quality and tone
  - Not scalable for fast-changing inventory or seasonal pushes

### ⚠️ Problem 2: Predicting purchase intent & optimizing ad strategy

- **Current issue**: Brands lack data-driven methods to forecast customer interest or decide which products deserve promotion.
- **Pain points**:
  - No personalization or targeting based on user type or product appeal
  - Promotions often ignore stock levels — leading to wasted spend on out-of-stock items
  - Difficult to coordinate ad campaigns with sales and inventory realities


## ✅ OmniRetail AI Solves This By:

- 🏷️ Automatically tagging and classifying products using CV models
- ✍️ Generating brand-consistent ad copy via LLMs
- 🧠 Predicting likelihood of purchase from realistic user behavior
- 🎯 Prioritizing promotions by predicted demand × available inventory
- 📋 Generating segmented CSV campaigns ready for deployment


## 🌍 Project Overview

OmniRetail AI simulates a full-stack marketing workflow using image classification, prompt-tuned text generation, behavioral prediction, and inventory-aware optimization.

It was built as a **portfolio project** to showcase applied ML, simulation, and production thinking.


## 🔥 Demo Highlights

- 🔍 **Product** **Tagging:** Classifies article type, color, and season using MobileNetV2
- ✍️ **Ad Copy Generation:** Uses DeepSeek-v3 via Hugging Face API to generate stylish, short-form ads
- 📊 **Purchase Prediction:** Predicts purchase probability using XGBoost and simulated clickstream
- 📦 **Inventory-Aware Ranking:** Combines intent score × inventory to prioritize products for promotion
- 🧠 **User Segmentation:** Tailors campaigns for user personas (budget, fashionista, color-lover)


## 🧪 Tech Stack

- 🐍 Python, Pandas, NumPy
- 🤖 TensorFlow, Keras (MobileNetV2), XGBoost
- 🔤 Hugging Face Transformers (DeepSeek-v3)
- 📊 Scikit-learn, Matplotlib
- 🧹 Data Simulation & Rule-Based Logic
- 📦 KaggleHub for dataset download


## 📂 Project Structure

```bash
omni-retail-ai/
├── data/
│   ├── raw/
│   │   ├── styles.csv              # Metadata from DeepFashion
│   │   └── images/                 # Product images (~44K)
│   ├── processed/                  # Labeled and scored product data
│   ├── sample/                     # Smaller demo-ready CSVs
├── models/                         # Trained MobileNetV2 CNN models
├── notebooks/
│   ├── 1_product_tagging.ipynb     # CNN model for articleType
│   ├── 2_colour_tagging.ipynb      # CNN model for baseColour
│   ├── 3_season_tagging.ipynb      # CNN model for season
│   ├── 4_ad_copy_generation.ipynb  # LLM-based ad generation via API
│   ├── 5_purchase_prediction.ipynb # XGBoost on simulated clickstream
│   ├── 6_inventory_strategy.ipynb  # Inventory × intent ranking
│   ├── 7_omni_retail_demo.ipynb    # End-to-end demo
├── src/
│   ├── download_data.py            # Download + extract DeepFashion
│   ├── product_tagger.py           # Predict image-based tags
│   ├── ad_copy_generator.py        # LLM wrapper for ad copy
│   ├── purchase_predictor.py       # XGBoost wrapper for intent model
│   ├── promotion_ranker.py         # Ranker and segmenter
├── requirements.txt
└── README.md
```


## 🔁 Feature-by-Feature Breakdown

| Notebook                      | Description                                                                 |
| ----------------------------- | --------------------------------------------------------------------------- |
| `1_product_tagging.ipynb`     | Classifies product images into article types using CNNs                     |
| `2_colour_tagging.ipynb`      | Predicts product base color using MobileNetV2                               |
| `3_season_tagging.ipynb`      | Predicts seasonal use (Spring/Fall/etc.) from images                        |
| `4_ad_copy_generation.ipynb`  | Generates catchy ad text using DeepSeek LLM                                 |
| `5_purchase_prediction.ipynb` | Simulates user-product clicks and trains XGBoost model to predict purchases |
| `6_inventory_strategy.ipynb`  | Combines inventory × demand for promotion ranking                           |
| `7_omni_retail_demo.ipynb`    | Executes the entire pipeline on sample product set                          |


## 🚀 Run Locally

1. **Clone this repo**

```bash
git clone https://github.com/your-username/OmniRetail-AI.git
cd OmniRetail-AI
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download the dataset (one-time setup)**

```bash
python src/download_data.py
```

4. **Explore notebooks or build a custom pipeline!**

> ⚠️ Note: The full image dataset is not included in the repository due to size.
> Please run `src/download_data.py` to fetch it locally using KaggleHub.


## 📋 License

This project is licensed under the [MIT License](LICENSE).


## 📬 Contact

Built with 💡 by Ting Ya Chang — Data & AI Enthusiast

🔗 [LinkedIn](https://www.linkedin.com/in/ting-ya-chang-analytics/)  
📫 Email: letitiachang0807@gmail.com

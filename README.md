# Suicidal Ideation Detection Using NLP, Machine Learning, and Deep Learning

A text-classification pipeline that identifies suicidal ideation in tweets using classical machine learning classifiers built on lexical and sentiment-based features. The project covers the full workflow from raw, weakly-labeled Twitter data through text preprocessing, semi-supervised labeling, and multi-model classification and evaluation.

> **Note on sensitive content:** This project analyzes text related to suicidal ideation for research and detection purposes. It is intended to support early-warning and moderation systems, not to provide clinical diagnosis. If you or someone you know is struggling, please reach out to a crisis line or your local emergency services.

## Overview

The goal of this project is to classify tweets as suicidal (`1`) or non-suicidal (`0`) using traditional ML classifiers trained on bag-of-words / n-gram text features. The pipeline is split into three notebooks, run in order:

1. **`Labeling data.ipynb`** — Combines labeled and unlabeled datasets and produces a single labeled dataset.
2. **`preprocessing.ipynb`** — Cleans and normalizes the tweet text.
3. **`machine learning.ipynb`** — Vectorizes the text and trains/evaluates six classifiers.

## Pipeline

### 1. Data Labeling (`Labeling data.ipynb`)

- Loads two pre-labeled datasets (`data.csv`, `suicidal_data.csv`) and two unlabeled raw Twitter datasets (`TwitterRawData.csv`, `suicideTweetData.csv`).
- Merges and de-duplicates the labeled sources on tweet text.
- For the unlabeled data, applies a **semi-supervised pseudo-labeling** approach:
  - Cleans tweets (tokenization, lowercasing, URL removal) using NLTK's `TweetTokenizer`.
  - Scores each tweet's sentiment using both **VADER** (NLTK's `SentimentIntensityAnalyzer`) and **TextBlob**.
  - Keeps only tweets where both sentiment methods agree (both negative or both positive), and uses that agreement as a proxy label (negative sentiment → suicidal, positive → non-suicidal).
- Combines the pseudo-labeled and originally labeled data, drops nulls and duplicates, producing the final combined dataset (`finalDf.csv`) of **~49,000 labeled tweets**.

### 2. Preprocessing (`preprocessing.ipynb`)

Cleans the combined dataset in preparation for feature extraction:
- Removes user mentions (`@user`), URLs, emojis, signatures, and non-standard characters via regex.
- Expands common chat abbreviations and contractions (e.g., "ASAP" → "As Soon As Possible").
- Strips numbers and English stopwords (NLTK).
- Removes rare words (bottom 10 least-frequent terms) to reduce noise.
- Applies **POS-tag-aware lemmatization** using NLTK's `WordNetLemmatizer`, mapping POS tags to noun/verb/adjective/adverb forms.
- Generates word clouds and class-balance visualizations to inspect the cleaned corpus (`cleaned_df.csv`).

### 3. Machine Learning (`machine learning.ipynb`)

- Loads the cleaned dataset, shuffles it, and removes short words (≤3 characters).
- Vectorizes text with **`CountVectorizer`** (unigrams, English stopwords removed, custom regex tokenizer).
- Splits into 80% train / 20% test.
- Trains and evaluates six classifiers, each with 5-fold cross-validation:

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Random Forest (`n_estimators=200`, entropy) | 99.8% | **92.9%** |
| Support Vector Classifier (SVC) | 95.3% | 91.9% |
| SGD Classifier (hinge loss) | 94.9% | 91.4% |
| Logistic Regression | 94.9% | 91.2% |
| Multinomial Naive Bayes | 90.8% | 85.3% |
| Complement Naive Bayes | 88.7% | 84.6% |

- Compares all models using accuracy, precision, recall, and F1-score, with bar-chart and ROC/AUC visualizations for side-by-side comparison.
- **Random Forest** achieved the best test-set performance, followed closely by SVC, SGD, and Logistic Regression.

## Datasets

| File | Description | Size |
|---|---|---|
| `dataset/Labeled/data.csv` | Pre-labeled tweet dataset | 7,122 rows |
| `dataset/Labeled/suicidal_data.csv` | Pre-labeled tweet dataset | 9,119 rows |
| `dataset/Unlabeled/TwitterRawData.csv` | Raw scraped tweets (unlabeled) | 52,622 rows |
| `dataset/Unlabeled/suicideTweetData.csv` | Raw scraped tweets (unlabeled) | 52,615 rows |
| `dataset/created/finalDf.csv` | Combined labeled dataset after pseudo-labeling | 49,178 rows |
| `dataset/created/cleaned_df.csv` | Fully cleaned/preprocessed text ready for modeling | 49,178 rows |

## Tech Stack

- **Language:** Python 3 (developed on Google Colab)
- **NLP:** `nltk` (stopwords, WordNet lemmatizer, POS tagging, TweetTokenizer, VADER sentiment), `TextBlob`, `langdetect`
- **ML:** `scikit-learn` — `CountVectorizer`, `MultinomialNB`, `ComplementNB`, `SGDClassifier`, `LogisticRegression`, `RandomForestClassifier`, `SVC`, `GridSearchCV`, `cross_val_score`
- **Data/Viz:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `wordcloud`, `plotly`/`chart-studio`

## How to Run

The notebooks were built for Google Colab and mount Google Drive for file storage. To run locally:

1. Clone the repository and install dependencies:
   ```bash
   pip install pandas numpy scikit-learn nltk textblob langdetect wordcloud seaborn matplotlib plotly chart-studio
   ```
2. Download required NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('vader_lexicon')
   ```
3. Update the `path` variable at the top of each notebook to point to your local `dataset/` directory instead of the Google Drive path.
4. Run the notebooks **in order**:
   1. `Labeling data.ipynb`
   2. `preprocessing.ipynb`
   3. `machine learning.ipynb`

## Project Structure

```
.
├── Labeling data.ipynb          # Merge labeled data + pseudo-label unlabeled data via sentiment
├── preprocessing.ipynb          # Text cleaning, lemmatization, word clouds
├── machine learning.ipynb       # Vectorization, model training, evaluation & comparison
├── dataset/
│   ├── Labeled/                 # Original pre-labeled tweet datasets
│   ├── Unlabeled/                # Raw scraped tweet datasets
│   └── created/                 # Intermediate & final processed datasets
└── README.md
```

## Future Improvements

- Incorporate deep learning models (LSTM, BERT-based transformers) for comparison against classical ML baselines.
- Replace sentiment-based pseudo-labeling with a human-verified or more robust weak-supervision approach.
- Expand external lexical/semantic resources (subjectivity lexicons, POS-tag features) as noted in the original methodology.

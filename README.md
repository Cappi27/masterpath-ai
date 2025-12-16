# University Research Profile & AI Trend Analysis Dashboard

This project implements a complete **Data Science + AI + Visualization pipeline** on large-scale **Scopus Engineering Literature (2018–2023)** to analyze:

- Global research trends by topic
- Institutional research profiles
- Collaboration networks
- AI-based document classification
- BERT-powered semantic recommendation
- Interactive geographic and network visualizations

The system follows a strict **three-module architecture** as required:
1. **Data Module**
2. **AI Module**
3. **Visualization Module (Streamlit Dashboard)**

---

## Key Features

- Large-scale JSON ingestion (20,000+ papers)
- Advanced data cleaning & normalization
- Topic modeling and clustering
- BERT embeddings for semantic understanding
- Supervised AI classification with full metrics
- Collaboration network graph analysis
- Global geographic visualization of institutions
- Interactive multi-page Streamlit dashboard
- AI-powered research recommender system

---

## Project Structure

Project_II/
│
├── data/
│   ├── processed_data_v2.csv                (auto-generated)
│   ├── institution_year_output.csv         (auto-generated)
│   ├── country_year_output.csv             (auto-generated)
│   ├── institution_ai_summary.csv          (auto-generated)
│   ├── bert_embeddings.csv                 (auto-generated)
│   ├── ai_model_metrics.csv                (auto-generated)
│   ├── topic_info.csv                      (auto-generated)
│   ├── country_centroids.csv               (external reference file)
│
├── 1_Data_module.ipynb
├── 2_AI_module.ipynb
├── _Streamlit_app.py
├── ScopusData2018-2023/                     (raw JSON dataset)
└── README.md

Files marked **auto-generated** are created automatically when you run the Data and AI modules.  
`country_centroids.csv` is an **external geographic lookup file** used for map visualization.

---

## Module Overview

---

### 1 Data Module — `_Data_module.ipynb`

**Purpose:**  
Transforms raw Scopus JSON files into clean, structured analytical datasets.

**Main Tasks:**
- Recursive JSON parsing
- Title, abstract, author, affiliation extraction
- Institution name cleaning
- Date normalization
- Country-level aggregation
- Institution–year time-series construction
- Collaboration edge generation
- Exploratory Data Analysis (EDA)

**Auto-Generated Outputs:**
- `institution_year_output.csv`
- `country_year_output.csv`
- `institution_ai_summary.csv`

**External Input:**
- `country_centroids.csv` (used to attach latitude & longitude for maps)

---

### 2 AI Module — `_AI_module.ipynb`

**Purpose:**  
Applies machine learning and deep learning techniques for topic analysis and classification.

**Models Implemented:**
- Topic clustering
- BERT text embeddings (Sentence Transformers)
- Logistic Regression classifier
- Semantic similarity recommender system

**Evaluation Metrics:**
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC–AUC  

**Auto-Generated Outputs:**
- `bert_embeddings.csv`
- `ai_model_metrics.csv`
- `topic_info.csv`
- Updated institution cluster assignments inside `institution_ai_summary.csv`

---

### 3 Visualization Module — `_streamlit_app.py`

Launches the interactive multi-page dashboard:

```bash
streamlit run _streamlit_app.py
# CS4800-Computational-Social-Science


🩺 Understanding Menopause on Reddit Using NLP
A Computational Analysis of Themes & Sentiments in Menopause-Related Discussions
📌 Overview

This project applies Natural Language Processing (NLP) methods to understand how people discuss menopause and perimenopause on Reddit. Using a pipeline involving data scraping, preprocessing, topic modeling (LDA), and sentiment analysis (VADER), the study uncovers dominant themes, emotional patterns, and unmet support needs expressed by users across menopause-related communities.

This repository contains the code, report, figures, and documentation associated with the project.

🚀 Key Objectives

Collect and preprocess Reddit posts from menopause-related subreddits

Identify high-level themes using Latent Dirichlet Allocation (LDA)

Analyze emotional tone using VADER sentiment analysis

Visualize topic–sentiment interactions

Draw insights applicable to healthcare communication, digital health tools, and support systems

🛠️ Tech Stack

Python 3.10+

Libraries:

praw or psaw (Reddit scraping)

pandas, numpy

spaCy, nltk

gensim (LDA)

matplotlib, seaborn, wordcloud

vaderSentiment

pyLDAvis for topic visualization

📂 Repository Structure
Menopause-Reddit-Analysis/
│
├── data/
│   ├── raw/                # Raw scraped Reddit posts
│   ├── processed/          # Cleaned + lemmatized corpus
│
├── notebooks/
│   ├── 01_scraping.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_lda_topic_modeling.ipynb
│   ├── 04_sentiment_analysis.ipynb
│   ├── 05_visualizations.ipynb
│
├── src/
│   ├── scraping.py
│   ├── cleaning.py
│   ├── lda_model.py
│   ├── sentiment.py
│   ├── utils.py
│
├── results/
│   ├── lda_topics.csv
│   ├── sentiment_scores.csv
│   ├── figures/
│       ├── coherence_plot.png
│       ├── topic_wordclouds/
│       ├── sentiment_distribution.png
│       ├── pyldavis_intertopic_map.html
│
├── report/
│   ├── IEEE_Report.pdf
│   ├── IEEE_Report.tex (optional)
│
├── README.md
└── requirements.txt

🧼 Data Preprocessing Pipeline

Key steps:

PII Removal

Emails, usernames, URLs, phone numbers removed

Text Cleaning

Lowercasing, emoji removal, punctuation normalization

Tokenization & Stopword Removal

Lemmatization using spaCy

Length Filtering for noise reduction

Vectorization using Gensim dictionary + BoW

This produces a clean corpus ideal for topic modeling.

🔍 Topic Modeling (LDA)

Tried K = 5 to 10 topics

Selected K = 7 using coherence score maximization

Final topics identified:

ID	Theme	Description
1	Cycle & Hormonal Adjustments	Menstrual irregularity, HRT, hot flashes
2	Doctor Consultations & HRT	Medical advice, estrogen, progesterone
3	Life Reflections	Mood changes, self-perception, aging
4	Sleep & Hot Flash Issues	Night sweats, insomnia
5	Anxiety & Pain	Emotional distress, chronic pain
6	Medical Research	Breast cancer risk, studies
7	Intimacy & Relationships	Sexual health, communication

Visualizations include word clouds and a pyLDAvis intertopic map.

😊 Sentiment Analysis (VADER)

Posts are classified into:

Positive

Neutral

Negative

Key findings:

Anxiety & Pain → highest negative sentiment

Sleep/Hot Flashes → predominantly negative

Doctor Consultations & Medical Research → more neutral/positive

Intimacy → mixed emotional tone

Charts include sentiment distribution per topic and overall emotional patterns.

📊 Visualizations

The project generates:

Coherence Score Plot

Topic Word Clouds

LDA Intertopic Map (pyLDAvis)

Sentiment Distribution Bar Charts

Topic–Sentiment Stacked Bars

All figures are stored in the results/figures/ directory.

🎯 Key Insights

Reddit serves as a major platform for menopause-related support

Users express high emotional burden (anxiety, pain, sleep issues)

Conversations around HRT and doctor visits are more hopeful

There is a need for better digital health tools and clinical communication

Social media analysis provides real-time public-health perspective

📌 Future Improvements

Introduce transformer-based models (BERT, RoBERTa)

Implement multi-label emotion classification

Temporal analysis of symptom progression

Expand dataset scope across other platforms (Twitter, Facebook groups)

Build a dashboard for real-time monitoring

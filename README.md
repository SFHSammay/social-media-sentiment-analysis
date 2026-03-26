# social-media-sentiment-analysis

## Overview
This project proposes a social media sentiment analysis system that classifies posts and comments as positive, negative, or neutral using publicly available datasets. 

We first retrieve relevant posts using information retrieval (IR) models, and then analyze their sentiments.

## Pipeline
data acquisition  
→ preprocessing  
→ retrieval (TF-IDF, BM25, QL)  
→ top-k documents  
→ retrieval evaluation (Precision@k, MAP)  
→ sentiment analysis (TF-IDF / Word2Vec → model → prediction)  
→ sentiment evaluation (Accuracy, F1)  
→ comparison  

## Milestone Progress
The milestone focuses on the early stages of the full pipeline, including data acquisition, preprocessing, and a baseline sentiment analysis component.

### Current implementation
labeled social media text  
→ preprocessing  
→ TF-IDF representation  
→ Logistic Regression model  
→ sentiment prediction  
→ evaluation (Accuracy, F1)
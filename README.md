# social-media-sentiment-analysis

## Overview
This is our final project for CS410 SP26.

This project proposes a social media sentiment analysis system that classifies posts and comments as positive, negative, or neutral using publicly available datasets. 

We first retrieve relevant posts using information retrieval (IR) models, and then analyze their sentiments.

## How to run
In order to run this project successfully, please have a Google Gemini API key defined in your path (with the variable name as GEMINI_API_KEY) before attempting to run. Execute the main.py file in order to run this project.

## Final Project State
The current state focuses on the final state of the full pipeline, including data acquisition, preprocessing, and a baseline sentiment analysis component, comparison, etc.

## Final implementation
data acquisition (Kaggle Twitter Dataset)  
→ preprocessing  
→ inverted index  
→ retrieval (TF-IDF, BM25, QL+JM Smoothing, QL+DP Smoothing)  
→ top-k documents for each model  
→ pooled retreived documents  
→ ground truth relevance labeling (Gemma, Gemini)  
→ retrieval evaluation (Precision@k, AP, MAP)  
→ sentiment analysis (TF-IDF + Logistic Regression)  
→ sentiment evaluation (Accuracy, F1)
→ comparison  

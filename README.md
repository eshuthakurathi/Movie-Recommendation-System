# Movie Recommendation System

## Project Overview
This project implements a content-based movie recommendation system using Python and Machine Learning techniques. It recommends movies based on similarities between their textual features.

## Dataset
- Dataset: TMDB 5000 Movie Dataset
- Source: Kaggle
- File: `tmdb_5000_movies.csv`
- Features: Movie title, overview, genres, and keywords.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Recommendation Technique
- Text Preprocessing
- TF-IDF Vectorization
- Cosine Similarity
- Content-Based Filtering

## Project Structure
```text
Task20/
├── datasets/
│   └── tmdb_5000_movies.csv
├── outputs/
├── app.py
├── requirements.txt
└── README.md
```

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Features
- Load and preprocess movie data.
- Convert text features into TF-IDF vectors.
- Calculate cosine similarity between movies.
- Recommend similar movies based on user selection.
- Display recommendations through a Streamlit interface.

## Project Type
Content-Based Movie Recommendation System.
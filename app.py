
import os
import ast
import re
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

DATA_PATH = os.path.join("datasets", "tmdb_5000_movies.csv")


def extract_names(value):
    try:
        items = ast.literal_eval(value)
        return " ".join(
            item["name"].replace(" ", "")
            for item in items
            if isinstance(item, dict) and "name" in item
        )
    except (ValueError, SyntaxError, TypeError):
        return ""


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(words)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    original_shape = df.shape
    original_columns = df.columns.tolist()

    required_columns = ["title", "overview", "genres", "keywords"]
    df = df[required_columns].copy()

    df.fillna("", inplace=True)

    df["genres"] = df["genres"].apply(extract_names)
    df["keywords"] = df["keywords"].apply(extract_names)

    df["combined_features"] = (
        df["overview"].astype(str) + " " +
        df["genres"].astype(str) + " " +
        df["keywords"].astype(str)
    )

    df["clean_text"] = df["combined_features"].apply(clean_text)

    df = df[df["title"].astype(str).str.strip() != ""]
    df.reset_index(drop=True, inplace=True)

    return df, original_shape, original_columns


@st.cache_resource
def build_similarity_model(text_data):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(text_data)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return vectorizer, tfidf_matrix, similarity_matrix


def recommend(item_name, df, similarity_matrix, top_n=5):
    movie_indices = df.index[
        df["title"].str.lower() == item_name.lower()
    ].tolist()

    if not movie_indices:
        return pd.DataFrame()

    movie_index = movie_indices[0]

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = [
        item for item in similarity_scores
        if item[0] != movie_index
    ]

    top_movies = similarity_scores[:top_n]

    recommended_indices = [item[0] for item in top_movies]

    recommendations = df.iloc[recommended_indices][
        ["title", "overview", "genres"]
    ].copy()

    recommendations["similarity_score"] = [
        round(score * 100, 2)
        for _, score in top_movies
    ]

    return recommendations


st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies similar to your favourite films "
    "using content-based filtering."
)

if not os.path.exists(DATA_PATH):
    st.error(
        "Dataset not found. Place tmdb_5000_movies.csv "
        "inside the datasets folder."
    )
    st.stop()

try:
    df, original_shape, original_columns = load_data()

    st.subheader("Task 1: Load & Understand Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Movies", original_shape[0])
    col2.metric("Total Columns", original_shape[1])
    col3.metric("Processed Movies", len(df))

    st.write("**Dataset Shape:**", original_shape)

    st.write("**Column Names:**")
    st.write(original_columns)

    st.write("**First 5 Rows:**")

    with st.expander("View Original Dataset"):
        original_df = pd.read_csv(DATA_PATH)
        st.dataframe(original_df.head(), use_container_width=True)

    st.write("**Essential Details:**")
    st.write(df[["title", "overview", "genres", "keywords"]].describe())

    st.write("**Text Columns Used for Recommendations:**")
    st.write(["overview", "genres", "keywords"])

    st.subheader("Task 2: Text Preprocessing")

    st.write(
        "Text is converted to lowercase, punctuation and special "
        "characters are removed, stopwords are removed, and missing "
        "values are replaced with empty strings."
    )

    with st.expander("View Cleaned Text"):
        st.dataframe(
            df[["title", "combined_features", "clean_text"]].head(10),
            use_container_width=True
        )

    st.success("Text preprocessing completed successfully.")

    st.subheader("Task 3: TF-IDF Vectorization")

    vectorizer, tfidf_matrix, similarity_matrix = build_similarity_model(
        df["clean_text"].tolist()
    )

    st.success("TF-IDF vectorization completed successfully.")

    col1, col2 = st.columns(2)

    col1.metric("TF-IDF Matrix Rows", tfidf_matrix.shape[0])
    col2.metric("TF-IDF Matrix Columns", tfidf_matrix.shape[1])

    st.write("**TF-IDF Matrix Shape:**", tfidf_matrix.shape)

    st.subheader("Task 4: Cosine Similarity")

    st.write(
        "Cosine similarity measures the similarity between two movie "
        "vectors. A score closer to 1 indicates greater similarity."
    )

    st.write("**Similarity Matrix Shape:**", similarity_matrix.shape)

    st.success("Cosine similarity matrix generated successfully.")

    st.subheader("Task 5: Build Recommendation Function")

    movie_list = sorted(df["title"].unique().tolist())

    selected_movie = st.selectbox(
        "Select a movie",
        movie_list
    )

    number_of_recommendations = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("Recommend Movies", type="primary"):

        recommendations = recommend(
            selected_movie,
            df,
            similarity_matrix,
            number_of_recommendations
        )

        if recommendations.empty:
            st.warning("No recommendations found.")

        else:
            st.subheader(f"Movies similar to {selected_movie}")

            for _, movie in recommendations.iterrows():

                with st.container(border=True):
                    st.markdown(f"### {movie['title']}")
                    st.write(movie["overview"])
                    st.caption(f"Genres: {movie['genres']}")
                    st.caption(
                        f"Similarity Score: "
                        f"{movie['similarity_score']}%"
                    )

    st.sidebar.title("About")

    st.sidebar.write("Content-Based Movie Recommendation System")
    st.sidebar.write("Algorithm: TF-IDF + Cosine Similarity")
    st.sidebar.write("Interface: Streamlit")

    st.sidebar.success("Recommendation system is ready.")

except Exception as error:
    st.error(f"An error occurred: {error}")

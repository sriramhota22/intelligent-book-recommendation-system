import streamlit as st
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk

nltk.download("stopwords")


# Page title
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Intelligent Book Recommendation System")
st.write("Book Recommendation using Text Mining and Machine Learning")


# Loading dataset
@st.cache_data
def load_data():
    books = pd.read_csv("Goodreadss Books.csv", engine="python")

    books = books[
        ["title", "author", "description", "genres", "avg_rating"]
    ]

    books = books.dropna()
    books = books.drop_duplicates()
    books = books.reset_index(drop=True)

    books["content"] = (
        books["description"].str[:500]
        + " "
        + books["genres"]
    )

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):

        text = text.lower()

        text = re.sub(r"[^\w\s]", "", text)

        words = text.split()

        words = [
            stemmer.stem(word)
            for word in words
            if word not in stop_words
        ]

        return " ".join(words)

    books["content"] = books["content"].apply(preprocess_text)

    tfidf = TfidfVectorizer(max_features=5000)

    tfidf_matrix = tfidf.fit_transform(books["content"])

    similarity = cosine_similarity(tfidf_matrix)

    return books, similarity


books, similarity = load_data()


def recommend_books(book_name, num_recommendations=5):

    book_index = books[books["title"] == book_name].index[0]

    similarity_scores = list(
        enumerate(similarity[book_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommended_books = similarity_scores[
        1:num_recommendations + 1
    ]

    recommendations = []

    for book in recommended_books:

        recommendations.append(
            books.iloc[book[0]][
                ["title", "author", "avg_rating"]
            ]
        )

    return (
        pd.DataFrame(recommendations)
        .reset_index(drop=True)
    )


selected_book = st.selectbox(
    "Select a Book",
    sorted(books["title"].unique())
)


if st.button("Recommend Books"):

    result = recommend_books(selected_book)

    st.subheader("Recommended Books")

    st.dataframe(
        result,
        use_container_width=True
    )
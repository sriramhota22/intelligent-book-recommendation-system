import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords only if needed
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# Page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Intelligent Book Recommendation System")
st.write("Book Recommendation using Text Mining and Machine Learning")


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

        text = str(text).lower()

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

    return books, tfidf_matrix


with st.spinner("Loading recommendation engine..."):
    books, tfidf_matrix = load_data()

st.success("Recommendation engine loaded successfully.")


def recommend_books(book_name, num_recommendations=5):

    book_index = books[books["title"] == book_name].index[0]

    book_vector = tfidf_matrix[book_index]

    similarity_scores = cosine_similarity(
        book_vector,
        tfidf_matrix
    ).flatten()

    similarity_scores = list(enumerate(similarity_scores))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []
    seen_titles = set()

    for index, score in similarity_scores[1:]:

        title = books.iloc[index]["title"]

        if title not in seen_titles:

            recommendations.append({
                "Title": books.iloc[index]["title"],
                "Author": books.iloc[index]["author"],
                "Average Rating": books.iloc[index]["avg_rating"]
            })

            seen_titles.add(title)

        if len(recommendations) == num_recommendations:
            break

    return pd.DataFrame(recommendations)


selected_book = st.selectbox(
    "Select a Book",
    sorted(books["title"].unique())
)

if st.button("Recommend Books"):

    recommendations = recommend_books(selected_book)

    st.subheader("Recommended Books")

    st.dataframe(
        recommendations,
        use_container_width=True
    )
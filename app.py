# app.py

import streamlit as st
import pickle
import pandas as pd
from collections import Counter
import requests

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

OMDB_API_KEY = "7fc646d4"


def fetch_poster(movie_title: str):
    try:
        url = (
            f"http://www.omdbapi.com/"
            f"?t={requests.utils.quote(movie_title)}"
            f"&apikey={OMDB_API_KEY}"
        )
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("Response") == "True":
            poster = data.get("Poster")
            if poster and poster != "N/A":
                return poster
    except Exception:
        pass
    return None


def get_imdb_rating(title, rating_dict):
    key = str(title).lower().strip()
    val = rating_dict.get(key, "N/A")
    return val


@st.cache_resource
def load_all_models():
    # TMDB
    tmdb_df = pickle.load(open("tmdb_df.pkl", "rb"))
    tfidf_tmdb = pickle.load(open("tfidf_tmdb.pkl", "rb"))
    knn_tmdb = pickle.load(open("knn_tmdb.pkl", "rb"))
    X_tmdb = tfidf_tmdb.transform(tmdb_df["tags"])

    # IMDb
    imdb_df = pickle.load(open("imdb_df.pkl", "rb"))
    tfidf_imdb = pickle.load(open("tfidf_imdb.pkl", "rb"))
    knn_imdb = pickle.load(open("knn_imdb.pkl", "rb"))
    X_imdb = tfidf_imdb.transform(imdb_df["tags"])

    # Rotten Tomatoes
    rotten_df = pickle.load(open("rotten_df.pkl", "rb"))
    tfidf_rotten = pickle.load(open("tfidf_rotten.pkl", "rb"))
    knn_rotten = pickle.load(open("knn_rotten.pkl", "rb"))
    X_rotten = tfidf_rotten.transform(rotten_df["tags"])

    imdb_rating_dict = pickle.load(open("imdb_rating_dict.pkl", "rb"))

    title_map = {}
    for df in [tmdb_df, imdb_df, rotten_df]:
        for _, row in df.iterrows():
            clean = str(row["title_clean"])
            if clean not in title_map:
                title_map[clean] = str(row["title"])
    all_display_titles = sorted(title_map.values())

    all_genres = set()
    for df in [tmdb_df, imdb_df, rotten_df]:
        for genres_list in df["genres"]:
            all_genres.update([g.lower().strip() for g in genres_list])
    all_genres = sorted(all_genres)

    return (
        tmdb_df, imdb_df, rotten_df,
        X_tmdb, X_imdb, X_rotten,
        knn_tmdb, knn_imdb, knn_rotten,
        imdb_rating_dict,
        title_map, all_display_titles,
        all_genres
    )


def recommend_from_tmdb(title, tmdb_df, X_tmdb, knn_tmdb, n=5):
    key = str(title).lower().strip()
    if key not in tmdb_df["title_clean"].values:
        return []
    pos = tmdb_df[tmdb_df["title_clean"] == key].index[0]
    _, indices = knn_tmdb.kneighbors(X_tmdb[pos], n_neighbors=n + 1)
    return tmdb_df.iloc[indices[0][1:]]["title"].tolist()


def recommend_from_imdb(title, imdb_df, X_imdb, knn_imdb, n=5):
    key = str(title).lower().strip()
    if key not in imdb_df["title_clean"].values:
        return []
    pos = imdb_df[imdb_df["title_clean"] == key].index[0]
    _, indices = knn_imdb.kneighbors(X_imdb[pos], n_neighbors=n + 1)
    return imdb_df.iloc[indices[0][1:]]["title"].tolist()


def recommend_from_rotten(title, rotten_df, X_rotten, knn_rotten, n=5):
    key = str(title).lower().strip()
    if key not in rotten_df["title_clean"].values:
        return []
    pos = rotten_df[rotten_df["title_clean"] == key].index[0]
    _, indices = knn_rotten.kneighbors(X_rotten[pos], n_neighbors=n + 1)
    return rotten_df.iloc[indices[0][1:]]["title"].tolist()


def recommend_ensemble(
        title,
        tmdb_df, imdb_df, rotten_df,
        X_tmdb, X_imdb, X_rotten,
        knn_tmdb, knn_imdb, knn_rotten,
        n=5
):
    key = str(title).lower().strip()
    all_recs = []

    if key in tmdb_df["title_clean"].values:
        all_recs += recommend_from_tmdb(title, tmdb_df, X_tmdb, knn_tmdb, n=n)
    if key in imdb_df["title_clean"].values:
        all_recs += recommend_from_imdb(title, imdb_df, X_imdb, knn_imdb, n=n)
    if key in rotten_df["title_clean"].values:
        all_recs += recommend_from_rotten(title, rotten_df, X_rotten, knn_rotten, n=n)

    if not all_recs:
        return []

    counts = Counter(all_recs)
    filtered = [(t, c) for t, c in counts.items() if t.lower().strip() != key]
    filtered_sorted = sorted(filtered, key=lambda x: (-x[1], x[0]))
    return [t for t, c in filtered_sorted][:n]


def recommend_by_genre_all(genre, tmdb_df, imdb_df, rotten_df, imdb_rating_dict, n=10):
    genre_key = genre.lower().strip()
    rows = []
    seen = set()

    for df in [tmdb_df, imdb_df, rotten_df]:
        for _, row in df.iterrows():
            genres_in_row = [g.lower().strip() for g in row["genres"]]
            if genre_key in genres_in_row:
                title_key = str(row["title"]).lower().strip()
                if title_key not in seen:
                    seen.add(title_key)
                    rating = imdb_rating_dict.get(title_key, "N/A")
                    rows.append({"title": row["title"], "imdb_rating": rating})

    if not rows:
        return pd.DataFrame()

    df_result = pd.DataFrame(rows)
    df_rated = df_result[df_result["imdb_rating"] != "N/A"].copy()
    df_rated["imdb_rating"] = pd.to_numeric(df_rated["imdb_rating"], errors="coerce")
    df_rated = df_rated.dropna(subset=["imdb_rating"])
    return df_rated.sort_values("imdb_rating", ascending=False).head(n).reset_index(drop=True)


(
    tmdb_df, imdb_df, rotten_df,
    X_tmdb, X_imdb, X_rotten,
    knn_tmdb, knn_imdb, knn_rotten,
    imdb_rating_dict,
    title_map, all_display_titles,
    all_genres
) = load_all_models()

st.markdown(
    "<h1 style='text-align: center;'>🎬 Movie Recommendation System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; font-size: 16px;'>"
    "Content-based movie recommendations using TMDB, IMDb, and Rotten Tomatoes Datasets"

    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

mode = st.radio("🔎 Search movies by:", ["Movie Title", "Genre"])

if mode == "Movie Title":
    selected_movie = st.selectbox("🎥 Select a movie:", all_display_titles)

    if st.button("🔍 Recommend"):
        if not selected_movie:
            st.warning("Please select a movie.")
        else:
            st.subheader(f"Recommendations for: **{selected_movie}**")

            recommended_titles = recommend_ensemble(
                selected_movie,
                tmdb_df, imdb_df, rotten_df,
                X_tmdb, X_imdb, X_rotten,
                knn_tmdb, knn_imdb, knn_rotten,
                n=5
            )

            if not recommended_titles:
                st.warning("No recommendations found across any dataset.")
            else:
                cols = st.columns(len(recommended_titles))

                for col, name in zip(cols, recommended_titles):
                    with col:
                        rating = get_imdb_rating(name, imdb_rating_dict)
                        rating_display = (
                            f"⭐ IMDb: {rating}"
                            if rating != "N/A"
                            else "⭐ Rating: N/A"
                        )
                        st.markdown(
                            f"<p style='text-align:center; font-weight:600;'>{name}</p>"
                            f"<p style='text-align:center; color:#f5c518;'>{rating_display}</p>",
                            unsafe_allow_html=True,
                        )
                        poster_url = fetch_poster(name)
                        if poster_url:
                            st.image(poster_url, use_container_width=True)
                        else:
                            st.markdown(
                                "<p style='text-align:center; color:gray;'>"
                                "No poster available</p>",
                                unsafe_allow_html=True,
                            )


else:
    selected_genre = st.selectbox("🎭 Select a genre:", all_genres)

    if st.button("🔍 Recommend"):
        if not selected_genre:
            st.warning("Please select a genre.")
        else:
            st.subheader(
                f"Top Rated **{selected_genre.title()}** Movies "
                f"(Sorted by IMDb Rating):"
            )

            recs = recommend_by_genre_all(
                selected_genre,
                tmdb_df, imdb_df, rotten_df,
                imdb_rating_dict,
                n=10
            )

            if recs.empty:
                st.error(
                    "No rated movies found for this genre. "
                    "Try a different genre."
                )
            else:
                titles = recs["title"].tolist()
                ratings = recs["imdb_rating"].tolist()

                for row_start in range(0, len(titles), 5):
                    batch_titles = titles[row_start: row_start + 5]
                    batch_ratings = ratings[row_start: row_start + 5]

                    cols = st.columns(len(batch_titles))

                    for col, name, rating in zip(cols, batch_titles, batch_ratings):
                        with col:
                            st.markdown(
                                f"<p style='text-align:center; font-weight:600;'>{name}</p>"
                                f"<p style='text-align:center; color:#f5c518;'>"
                                f"⭐ IMDb: {rating:.1f}</p>",
                                unsafe_allow_html=True,
                            )
                            poster_url = fetch_poster(name)
                            if poster_url:
                                st.image(poster_url, use_container_width=True)
                            else:
                                st.markdown(
                                    "<p style='text-align:center; color:gray;'>"
                                    "No poster available</p>",
                                    unsafe_allow_html=True,
                                )
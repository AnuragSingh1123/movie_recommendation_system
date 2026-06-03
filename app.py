import streamlit as st
import pickle
import pandas as pd
from collections import Counter
import requests
import ast

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


def parse_cast_list(cast_data):
    """Safely normalizes cast data from pickled dataframes."""
    if isinstance(cast_data, (list, set, tuple)):
        return [str(a).strip().lower() for a in cast_data if a]
    try:
        if cast_data is None or pd.isna(cast_data):
            return []
    except (TypeError, ValueError):
        return []
    if isinstance(cast_data, str) and cast_data.strip().startswith('['):
        try:
            parsed = ast.literal_eval(cast_data)
            if isinstance(parsed, list):
                return [str(a).strip().lower() for a in parsed if a]
        except (ValueError, SyntaxError):
            pass
    if isinstance(cast_data, str):
        return [a.strip().lower() for a in cast_data.split(',') if a.strip()]
    return []


def format_actor_name(token):
    """Converts spaceless lowercase names back to readable names for the dropdown menu."""
    special_cases = {
        "tomcruise": "Tom Cruise",
        "tomhanks": "Tom Hanks",
        "bradpitt": "Brad Pitt",
        "leonardodicaprio": "Leonardo DiCaprio",
        "johnnydepp": "Johnny Depp",
        "willsmith": "Will Smith",
        "scarlettjohansson": "Scarlett Johansson",
        "samworthington": "Sam Worthington"
    }
    if token in special_cases:
        return special_cases[token]

    import re
    result = re.sub(r"(\w)([A-Z])", r"\1 \2", token.title())
    return result


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

    # Bollywood (New Dataset Pickles Integration)
    bolly_df = pickle.load(open("bolly_df.pkl", "rb"))
    tfidf_bolly = pickle.load(open("bolly_tfidf.pkl", "rb"))
    knn_bolly = pickle.load(open("bolly_knn.pkl", "rb"))
    X_bolly = tfidf_bolly.transform(bolly_df["tags"])

    imdb_rating_dict = pickle.load(open("imdb_rating_dict.pkl", "rb"))

    try:
        actor_display_map = pickle.load(open("actor_display_map.pkl", "rb"))
    except FileNotFoundError:
        actor_display_map = {}

    title_map = {}
    for df in [tmdb_df, imdb_df, rotten_df, bolly_df]:
        for _, row in df.iterrows():
            clean = str(row["title_clean"])
            if clean not in title_map:
                title_map[clean] = str(row["title"])
    all_display_titles = sorted(title_map.values())

    all_genres = set()
    for df in [tmdb_df, imdb_df, rotten_df, bolly_df]:
        for genres_list in df["genres"]:
            all_genres.update([g.lower().strip() for g in genres_list])
    all_genres = sorted(all_genres)

    confirmed_cast_tokens = set()
    for df in [tmdb_df, imdb_df, rotten_df, bolly_df]:
        col = "cast" if "cast" in df.columns else ("actors" if "actors" in df.columns else None)
        if col:
            for val in df[col]:
                for token in parse_cast_list(val):
                    confirmed_cast_tokens.add(token)

    all_actors_display = []
    if actor_display_map:
        all_actors_display = sorted([
            display_name
            for key, display_name in actor_display_map.items()
            if key in confirmed_cast_tokens
        ])
    else:
        all_actors_display = sorted([format_actor_name(t) for t in confirmed_cast_tokens if t])

    return (
        tmdb_df, imdb_df, rotten_df, bolly_df,
        X_tmdb, X_imdb, X_rotten, X_bolly,
        knn_tmdb, knn_imdb, knn_rotten, knn_bolly,
        imdb_rating_dict,
        title_map, all_display_titles,
        all_genres, all_actors_display
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


def recommend_from_bolly(title, bolly_df, X_bolly, knn_bolly, n=5):
    key = str(title).lower().strip()
    if key not in bolly_df["title_clean"].values:
        return []
    pos = bolly_df[bolly_df["title_clean"] == key].index[0]
    _, indices = knn_bolly.kneighbors(X_bolly[pos], n_neighbors=n + 1)
    return bolly_df.iloc[indices[0][1:]]["title"].tolist()


def recommend_ensemble(
        title,
        tmdb_df, imdb_df, rotten_df, bolly_df,
        X_tmdb, X_imdb, X_rotten, X_bolly,
        knn_tmdb, knn_imdb, knn_rotten, knn_bolly,
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
    if key in bolly_df["title_clean"].values:
        all_recs += recommend_from_bolly(title, bolly_df, X_bolly, knn_bolly, n=n)

    if not all_recs:
        return []

    counts = Counter(all_recs)
    filtered = [(t, c) for t, c in counts.items() if t.lower().strip() != key]
    filtered_sorted = sorted(filtered, key=lambda x: (-x[1], x[0]))
    return [t for t, c in filtered_sorted][:n]


def recommend_by_genre_all(genre, tmdb_df, imdb_df, rotten_df, bolly_df, imdb_rating_dict, n=10):
    genre_key = genre.lower().strip()
    rows = []
    seen = set()

    for df in [tmdb_df, imdb_df, rotten_df, bolly_df]:
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


def get_all_movies_by_actor(actor_name, tmdb_df, imdb_df, rotten_df, bolly_df, imdb_rating_dict):
    """Scans all dataframes to pull out EVERY movie featuring this actor, sorted top rated to least rated."""
    lookup_key = actor_name.replace(" ", "").lower().strip()
    rows = []
    seen = set()

    for df in [tmdb_df, imdb_df, rotten_df, bolly_df]:
        col_name = "cast" if "cast" in df.columns else ("actors" if "actors" in df.columns else None)
        if not col_name:
            continue

        for _, row in df.iterrows():
            cast_in_row = parse_cast_list(row[col_name])
            if lookup_key in cast_in_row:
                title_key = str(row["title"]).lower().strip()
                if title_key not in seen:
                    seen.add(title_key)
                    rating = imdb_rating_dict.get(title_key, None)
                    rows.append({"title": row["title"], "imdb_rating": rating})

    if not rows:
        return pd.DataFrame()

    df_result = pd.DataFrame(rows)
    df_result["imdb_rating"] = pd.to_numeric(df_result["imdb_rating"], errors="coerce")
    df_result = df_result.sort_values("imdb_rating", ascending=False, na_position='last')
    return df_result.reset_index(drop=True)


(
    tmdb_df, imdb_df, rotten_df, bolly_df,
    X_tmdb, X_imdb, X_rotten, X_bolly,
    knn_tmdb, knn_imdb, knn_rotten, knn_bolly,
    imdb_rating_dict,
    title_map, all_display_titles,
    all_genres, all_actors
) = load_all_models()


# ─────────────────────────────────────────────
# Shared card renderer
# ─────────────────────────────────────────────
def render_movie_card(col, name, rating=None):
    with col:
        if rating is None:
            raw = get_imdb_rating(name, imdb_rating_dict)
            try:
                rating_val = f"{float(raw):.1f}" if raw != "N/A" else "N/A"
            except (TypeError, ValueError):
                rating_val = "N/A"
        else:
            try:
                rating_val = f"{float(rating):.1f}" if pd.notna(rating) else "N/A"
            except (TypeError, ValueError):
                rating_val = "N/A"

        display_name = name if len(name) <= 30 else name[:28] + "…"

        st.markdown(
            f"""
            <div style="text-align:center; padding:4px 2px;">
              <p style="font-weight:700; font-size:13px; min-height:36px;
                        line-height:1.3; margin-bottom:4px; overflow:hidden;">
                {display_name}
              </p>
              <p style="color:#f5c518; font-size:13px; font-weight:600;
                        margin-bottom:6px;">
                ⭐ {rating_val}
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        poster_url = fetch_poster(name)
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            st.markdown(
                """
                <div style="height:220px; background:#1e1e1e; border-radius:8px;
                            display:flex; align-items:center; justify-content:center;
                            color:#666; font-size:12px; text-align:center;">
                  🎬<br>No Poster
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align: center;'>🎬 Movie Recommendation System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; font-size: 16px;'>"
    "Content-based movie recommendations using TMDB, IMDb, Rotten Tomatoes, and Bollywood Datasets"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

mode = st.radio("🔎 Search movies by:", ["Movie Title", "Genre", "Actor"])

# ─────────────────────────────────────────────
# Movie Title mode
# ─────────────────────────────────────────────
if mode == "Movie Title":
    selected_movie = st.selectbox("🎥 Select a movie:", all_display_titles)

    if st.button("🔍 Recommend"):
        if not selected_movie:
            st.warning("Please select a movie.")
        else:
            st.subheader(f"Recommendations for: **{selected_movie}**")

            recommended_titles = recommend_ensemble(
                selected_movie,
                tmdb_df, imdb_df, rotten_df, bolly_df,
                X_tmdb, X_imdb, X_rotten, X_bolly,
                knn_tmdb, knn_imdb, knn_rotten, knn_bolly,
                n=5
            )

            if not recommended_titles:
                st.warning("No recommendations found across any dataset.")
            else:
                cols = st.columns(5)
                for col, name in zip(cols, recommended_titles):
                    render_movie_card(col, name)

# ─────────────────────────────────────────────
# Genre mode
# ─────────────────────────────────────────────
elif mode == "Genre":
    selected_genre = st.selectbox("🎭 Select a genre:", all_genres)

    if st.button("🔍 Recommend"):
        if not selected_genre:
            st.warning("Please select a genre.")
        else:
            st.subheader(
                f"Top 10 **{selected_genre.title()}** Movies "
                f"(Sorted by IMDb Rating):"
            )

            recs = recommend_by_genre_all(
                selected_genre,
                tmdb_df, imdb_df, rotten_df, bolly_df,
                imdb_rating_dict,
                n=10
            )

            if recs.empty:
                st.error("No rated movies found for this genre. Try a different genre.")
            else:
                titles  = recs["title"].tolist()
                ratings = recs["imdb_rating"].tolist()

                for row_start in range(0, len(titles), 5):
                    batch_titles  = titles[row_start: row_start + 5]
                    batch_ratings = ratings[row_start: row_start + 5]
                    cols = st.columns(5)
                    for col, name, rating in zip(cols, batch_titles, batch_ratings):
                        render_movie_card(col, name, rating)
                    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Actor mode
# ─────────────────────────────────────────────
else:
    if not all_actors:
        st.error("No actor data found. Ensure your datasets contain a cast column.")
    else:
        selected_actor = st.selectbox("🌟 Select an actor:", all_actors)

        if st.button("🔍 Find Movies"):
            movies_df = get_all_movies_by_actor(
                selected_actor,
                tmdb_df, imdb_df, rotten_df, bolly_df,
                imdb_rating_dict
            )

            if movies_df.empty:
                st.error(f"No movies found featuring **{selected_actor}** in the dataset.")
            else:
                movies_df = movies_df.head(10)
                total = len(movies_df)

                st.subheader(
                    f"🎞️ Top {total} Movies Featuring **{selected_actor}** "
                    f"(Sorted by IMDb Rating)"
                )

                titles  = movies_df["title"].tolist()
                ratings = movies_df["imdb_rating"].tolist()

                for row_start in range(0, total, 5):
                    batch_titles  = titles[row_start: row_start + 5]
                    batch_ratings = ratings[row_start: row_start + 5]
                    cols = st.columns(5)
                    for col, name, rating in zip(cols, batch_titles, batch_ratings):
                        render_movie_card(col, name, rating)
                    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)
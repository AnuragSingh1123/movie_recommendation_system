import streamlit as st
import pickle
import pandas as pd
from collections import Counter
import requests
import ast

# ─────────────────────────────────────────────
# Page Configurations & Mobile Responsive UI Styling
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Custom Streaming Application CSS Injection (Handles Mobile Scaling)
st.markdown("""
    <style>
    /* Elegant movie card border-radius and smooth hover transitions */
    div[data-testid="stImage"] img {
        border-radius: 8px !important;
        object-fit: cover !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    div[data-testid="stImage"] img:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 12px rgba(245, 197, 24, 0.2);
    }

    /* ENHANCED MOBILE RESPONSIVENESS PATCH */
    /* When Streamlit columns wrap vertically on smartphones, fluidly scale 
       and center posters so they never look artificially blown-up or overly large. */
    @media (max-width: 768px) {
        div[data-testid="stImage"] img {
            max-width: 65% !important;
            height: auto !important;
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
        }
        /* Keep text centered and clean on narrow screen layouts */
        div[style*="text-align:center"] {
            margin-top: 12px;
            margin-bottom: 8px;
        }
    }
    </style>
""", unsafe_allow_html=True)

OMDB_API_KEY = "7fc646d4"

# Ultra-reliable, developer-whitelisted asset from placehold.co
# styled as a dark sleek movie card that can never be intercepted or blocked.
DEFAULT_POSTER_URL = "https://placehold.co/400x600/1a1a1a/ffffff?text=No+Poster+Available"


def extract_title_from_image(image_file, all_titles):
    """
    Safely extracts text from a camera snapshot using EasyOCR or Pytesseract,
    then applies an adjacent fuzzy-string matcher to locate the closest title in the dataset.
    """
    try:
        from PIL import Image
        import io
        import difflib

        img = Image.open(image_file)
        detected_text = ""

        # 1. Try scanning via EasyOCR
        try:
            import easyocr
            import numpy as np
            reader = easyocr.Reader(['en'], gpu=False)
            results = reader.readtext(np.array(img))
            detected_text = " ".join([res[1] for res in results])
        except ImportError:
            # 2. Try scanning via Pytesseract as a local fallback
            try:
                import pytesseract
                detected_text = pytesseract.image_to_string(img)
            except ImportError:
                st.error(
                    "⚠️ **OCR Libraries Missing:** To use the camera poster-scanning feature, please install an OCR library in your terminal:")
                st.code("pip install easyocr\n# OR\pip install pytesseract pillow")
                return None

        # Run fuzzy lookups if a phrase string is recovered from image arrays
        text_clean = detected_text.strip()
        if text_clean:
            # Use cutoff=0.3 to remain highly permissive of stylized movie font faces and layouts
            matches = difflib.get_close_matches(text_clean, all_titles, n=1, cutoff=0.3)
            if matches:
                return matches[0]

    except Exception as e:
        st.error(f"Error processing captured camera frame: {e}")
    return None


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

    # Bollywood
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


def recommend_by_shared_genre(source_genres, exclude_titles, tmdb_df, imdb_df, rotten_df, bolly_df, imdb_rating_dict,
                              n=5):
    if not source_genres: return []
    genre_keys = [g.lower().strip() for g in source_genres]
    rows = []
    seen = set(str(t).lower().strip() for t in exclude_titles)

    for df in [tmdb_df, imdb_df, rotten_df, bolly_df]:
        for _, row in df.iterrows():
            title_key = str(row["title"]).lower().strip()
            if title_key not in seen:
                genres_in_row = [g.lower().strip() for g in row["genres"]]
                overlap = len(set(genre_keys).intersection(genres_in_row))
                if overlap > 0:
                    seen.add(title_key)
                    rating = imdb_rating_dict.get(title_key, "N/A")
                    rows.append({"title": row["title"], "rating": rating, "overlap": overlap})

    if not rows: return []
    df_res = pd.DataFrame(rows)
    df_res["num_rating"] = pd.to_numeric(df_res["rating"], errors="coerce").fillna(0.0)
    df_res = df_res.sort_values(by=["overlap", "num_rating"], ascending=False)
    return df_res.head(n)[["title", "rating"]].to_dict(orient="records")


def recommend_by_shared_cast(source_cast, exclude_titles, tmdb_df, imdb_df, rotten_df, bolly_df, imdb_rating_dict, n=5):
    if not source_cast: return []
    cast_keys = [c.lower().strip() for c in source_cast]
    rows = []
    seen = set(str(t).lower().strip() for t in exclude_titles)

    for df in [tmdb_df, imdb_df, rotten_df, bolly_df]:
        col_name = "cast" if "cast" in df.columns else ("actors" if "actors" in df.columns else None)
        if not col_name: continue
        for _, row in df.iterrows():
            title_key = str(row["title"]).lower().strip()
            if title_key not in seen:
                cast_in_row = parse_cast_list(row[col_name])
                overlap = len(set(cast_keys).intersection(cast_in_row))
                if overlap > 0:
                    seen.add(title_key)
                    rating = imdb_rating_dict.get(title_key, "N/A")
                    rows.append({"title": row["title"], "rating": rating, "overlap": overlap})

    if not rows: return []
    df_res = pd.DataFrame(rows)
    df_res["num_rating"] = pd.to_numeric(df_res["rating"], errors="coerce").fillna(0.0)
    df_res = df_res.sort_values(by=["overlap", "num_rating"], ascending=False)
    return df_res.head(n)[["title", "rating"]].to_dict(orient="records")


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
# Shared Card Renderer with Default Fallback & Scaling
# ─────────────────────────────────────────────
def render_movie_card(col, name, rating=None, default_reason="Curated Pick"):
    """Renders a single movie card with automatic fallback placeholder and mobile view constraints."""
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

        # Dynamic Discovery Logic badge selection
        if rating_val != "N/A":
            num_rating = float(rating_val)
            if num_rating >= 8.2:
                reason_text = "🏆 Masterpiece"
            elif num_rating >= 7.5:
                reason_text = "🔥 Highly Rated"
            else:
                reason_text = default_reason
        else:
            reason_text = default_reason

        display_name = name if len(name) <= 30 else name[:28] + "…"

        st.markdown(
            f"""
            <div style="text-align:center; padding:4px 2px;">
              <p style="font-weight:700; font-size:13px; min-height:36px;
                        line-height:1.3; margin-bottom:4px; overflow:hidden;">
                {display_name}
              </p>
              <p style="color:#f5c518; font-size:12px; font-weight:600; margin-bottom:6px;">
                ⭐ {rating_val} &nbsp;|&nbsp; <span style="color:#888; font-weight:400; font-size:11px;">{reason_text}</span>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Poster lookup with fallback configuration handling
        poster_url = fetch_poster(name)
        if not poster_url:
            poster_url = DEFAULT_POSTER_URL

        st.image(poster_url, use_container_width=True)


# ─────────────────────────────────────────────
# Page Header Rendering
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
# Movie Title Mode
# ─────────────────────────────────────────────
if mode == "Movie Title":
    # Check if a movie was just captured from camera to set default dropdown index
    default_index = 0
    if "scanned_movie" in st.session_state and st.session_state["scanned_movie"] in all_display_titles:
        default_index = all_display_titles.index(st.session_state["scanned_movie"])

    selected_movie = st.selectbox("🎥 Select a movie:", all_display_titles, index=default_index)

    # Cohesive Expandable Row for Live Media Image Scanning
    with st.expander("📷 Scan Movie Poster or Title Text via Camera"):
        cam_image = st.camera_input("Take a photo of the movie name or poster")
        if cam_image:
            with st.spinner("Processing image pixels and extracting title text..."):
                scanned_name = extract_title_from_image(cam_image, all_display_titles)
                if scanned_name:
                    st.success(f"🎯 Auto-detected Movie Match: **{scanned_name}**")
                    st.session_state["scanned_movie"] = scanned_name
                    # Cross-version standard safe viewport refresh triggers
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()
                else:
                    st.warning(
                        "Could not clearly identify a matching movie title from the image. Ensure text is clear, or choose from the dropdown list box above.")

    if st.button("🔍 Recommend"):
        if not selected_movie:
            st.warning("Please select a movie.")
        else:
            source_genres = []
            source_cast = []
            for df in [tmdb_df, imdb_df, rotten_df, bolly_df]:
                if selected_movie.lower().strip() in df["title_clean"].values:
                    row = df[df["title_clean"] == selected_movie.lower().strip()].iloc[0]
                    if "genres" in row and isinstance(row["genres"], list):
                        source_genres = row["genres"]
                    if "cast" in row:
                        source_cast = parse_cast_list(row["cast"])
                    break

            # --- ROW 1: ENSEMBLE STRATEGY MATCHES ---
            st.markdown("#### 🎯 Top Picks For You")
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
                    render_movie_card(col, name, default_reason="🎯 Plot Match")

            st.markdown("<div style='margin-bottom:24px'></div>", unsafe_allow_html=True)

            # --- ROW 2: GENRE ALIGNED MATCHES ---
            st.markdown("#### 🎭 More From These Genres")
            exclude_set = [selected_movie] + recommended_titles
            genre_recs = recommend_by_shared_genre(
                source_genres, exclude_set,
                tmdb_df, imdb_df, rotten_df, bolly_df,
                imdb_rating_dict, n=5
            )

            if not genre_recs:
                st.caption("No matching structural genres located.")
            else:
                cols = st.columns(5)
                for col, item in zip(cols, genre_recs):
                    render_movie_card(col, item["title"], rating=item["rating"], default_reason="🎭 Genre Fit")

            st.markdown("<div style='margin-bottom:24px'></div>", unsafe_allow_html=True)

            # --- ROW 3: CAST STAR SPOTLIGHT MATCHES ---
            st.markdown("#### 🌟 Starring Your Favorite Cast")
            exclude_set += [item["title"] for item in genre_recs]
            cast_recs = recommend_by_shared_cast(
                source_cast, exclude_set,
                tmdb_df, imdb_df, rotten_df, bolly_df,
                imdb_rating_dict, n=5
            )

            if not cast_recs:
                st.caption("No alternate cast-aligned films discovered.")
            else:
                cols = st.columns(5)
                for col, item in zip(cols, cast_recs):
                    render_movie_card(col, item["title"], rating=item["rating"], default_reason="🌟 Star Cast")

# ─────────────────────────────────────────────
# Genre Mode
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
                titles = recs["title"].tolist()
                ratings = recs["imdb_rating"].tolist()

                for row_start in range(0, len(titles), 5):
                    batch_titles = titles[row_start: row_start + 5]
                    batch_ratings = ratings[row_start: row_start + 5]
                    cols = st.columns(5)
                    for col, name, rating in zip(cols, batch_titles, batch_ratings):
                        render_movie_card(col, name, rating, default_reason="🎭 Genre Classic")
                    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Actor Mode
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

                titles = movies_df["title"].tolist()
                ratings = movies_df["imdb_rating"].tolist()

                for row_start in range(0, total, 5):
                    batch_titles = titles[row_start: row_start + 5]
                    batch_ratings = ratings[row_start: row_start + 5]
                    cols = st.columns(5)
                    for col, name, rating in zip(cols, batch_titles, batch_ratings):
                        render_movie_card(col, name, rating, default_reason="🌟 Iconic Role")
                    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)
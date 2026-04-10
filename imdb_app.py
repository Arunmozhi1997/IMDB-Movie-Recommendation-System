"""
=============================================================
  IMDB 2024 Movie Recommendation System — Streamlit App
  Run: streamlit run app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
import re
import nltk
import contractions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# ─── NLTK downloads ───────────────────────────────────────────────────────────
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 IMDB Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #e6edf3; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="metric-container"] label { color: #8b949e !important; font-size: 12px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #58a6ff !important; font-size: 26px; font-weight: 700;
    }

    /* Movie result cards */
    .movie-card {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 14px;
        transition: border-color 0.2s;
    }
    .movie-card:hover { border-color: #58a6ff; }

    /* Rank badge */
    .rank-badge {
        display: inline-block;
        background-color: #1f6feb;
        color: white;
        font-size: 13px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 8px;
    }

    /* Movie title */
    .movie-title {
        font-size: 18px;
        font-weight: 700;
        color: #e6edf3;
        margin: 4px 0 8px 0;
    }

    /* Storyline text */
    .storyline-text {
        font-size: 13px;
        color: #8b949e;
        line-height: 1.6;
        margin-bottom: 12px;
    }

    /* Meta tags (rating, duration) */
    .meta-tag {
        display: inline-block;
        background-color: #21262d;
        color: #c9d1d9;
        font-size: 12px;
        padding: 3px 10px;
        border-radius: 6px;
        margin-right: 6px;
    }

    /* Similarity pill */
    .sim-high   { background-color: #1a472a; color: #3fb950; border-radius: 8px;
                  padding: 4px 12px; font-size: 13px; font-weight: 700; float: right; }
    .sim-medium { background-color: #2d2006; color: #ffa657; border-radius: 8px;
                  padding: 4px 12px; font-size: 13px; font-weight: 700; float: right; }
    .sim-low    { background-color: #2d1414; color: #f78166; border-radius: 8px;
                  padding: 4px 12px; font-size: 13px; font-weight: 700; float: right; }

    /* Section headers */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #e6edf3;
        border-left: 4px solid #58a6ff;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #21262d; }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* Text area */
    textarea {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    /* Button */
    .stButton > button {
        background-color: #1f6feb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-size: 15px;
        font-weight: 600;
        width: 100%;
        transition: background-color 0.2s;
    }
    .stButton > button:hover { background-color: #388bfd; }

    /* Divider */
    hr { border-color: #21262d; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODELS (cached — runs only once) ───────────────────────────────────
@st.cache_resource
def load_model():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    tfidf_matrix = sp.load_npz("tfidf_matrix.npz")
    return tfidf, tfidf_matrix

@st.cache_data
def load_data():
    df = pd.read_csv("imdb_cleaned.csv")
    return df

# ─── TEXT PREPROCESSING (must match clean_and_model.py exactly) ──────────────
@st.cache_resource
def load_nlp():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

def text_preprocessing(text):
    stop_words, lemmatizer = load_nlp()
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([w for w in text.split() if w.isalpha()])
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words])

# ─── RECOMMENDATION FUNCTION ─────────────────────────────────────────────────
def recommend_movies(user_story, top_n=5):
    cleaned = text_preprocessing(user_story)
    if not cleaned.strip():
        return pd.DataFrame()
    user_vector       = tfidf.transform([cleaned])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices       = similarity_scores.argsort()[::-1][:top_n]
    results           = df.iloc[top_indices][['title', 'clean_storyline', 'rating', 'duration_mins']].copy()
    results['similarity_score'] = (similarity_scores[top_indices] * 100).round(2)
    results = results.reset_index(drop=True)
    results.index = results.index + 1
    return results

# ─── SIMILARITY COLOR HELPER ─────────────────────────────────────────────────
def sim_pill(score):
    if score >= 30:
        return f'<span class="sim-high">🟢 {score}% match</span>'
    elif score >= 15:
        return f'<span class="sim-medium">🟡 {score}% match</span>'
    else:
        return f'<span class="sim-low">🔴 {score}% match</span>'

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
try:
    tfidf, tfidf_matrix = load_model()
    df = load_data()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}\n\nMake sure you have run clean_and_model.py first.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 Movie Recommender")
    st.markdown("*IMDB 2024 — NLP Powered*")
    st.markdown("---")

    # Dataset stats
    st.markdown("### 📊 Dataset Stats")
    st.metric("Total Movies",   f"{len(df):,}")
    st.metric("Avg IMDb Rating", f"{df['rating'].mean():.2f} ⭐")
    st.metric("Avg Duration",    f"{int(df['duration_mins'].mean())} min")

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Settings")
    top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    min_rating = st.slider("Minimum IMDb rating filter", min_value=1.0,
                           max_value=9.0, value=1.0, step=0.5)

    st.markdown("---")

    # How it works
    st.markdown("### 💡 How It Works")
    st.markdown("""
    1. Your text is **cleaned** and tokenized
    2. **TF-IDF** converts it to a number vector
    3. **Cosine Similarity** finds closest movies
    4. Top matches are ranked and displayed
    """)

    st.markdown("---")
    st.markdown("### 🎯 Match Score Guide")
    st.markdown("""
    🟢 **≥ 30%** — Strong match  
    🟡 **15–30%** — Moderate match  
    🔴 **< 15%** — Weak match  
    """)

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# 🎬 IMDB 2024 Movie Recommendation System")
st.markdown("*Describe a movie plot and we'll find the most similar 2024 movies using NLP*")
st.markdown("---")

# ─── KPI ROW ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎬 Movies in DB",    f"{len(df):,}")
col2.metric("⭐ Avg Rating",       f"{df['rating'].mean():.2f}")
col3.metric("🕐 Avg Duration",     f"{int(df['duration_mins'].mean())} min")
col4.metric("📖 TF-IDF Features",  f"{tfidf_matrix.shape[1]:,}")

st.markdown("---")

# ─── EXAMPLE BUTTONS ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🚀 Quick Examples</div>', unsafe_allow_html=True)

examples = {
    "⚔️ War / Action":    "Soldiers fight in a brutal war, facing impossible odds to survive and protect their country from enemy forces.",
    "🚀 Sci-Fi":          "In a distant future, astronauts travel to a new galaxy to find a habitable planet and save humanity from extinction.",
    "💕 Romance":         "Two strangers from completely different worlds meet by chance and slowly fall in love despite all obstacles.",
    "🔍 Crime / Thriller":"A detective investigates a mysterious murder in a city filled with corruption, chasing a dangerous killer.",
    "🧙 Fantasy":         "A young hero discovers magical powers and must defeat an ancient evil sorcerer threatening to destroy the kingdom.",
    "😂 Comedy":          "A group of friends go on a hilarious road trip filled with unexpected mishaps and laugh-out-loud moments.",
}

# Store selected example in session state
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = ""

cols = st.columns(3)
for i, (label, text) in enumerate(examples.items()):
    if cols[i % 3].button(label, use_container_width=True):
        st.session_state.selected_example = text

st.markdown("---")

# ─── INPUT AREA ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">✍️ Enter Your Storyline</div>', unsafe_allow_html=True)

user_story = st.text_area(
    label="Describe the kind of movie you want:",
    value=st.session_state.selected_example,
    height=130,
    placeholder="e.g. A group of mercenaries must infiltrate a heavily guarded facility to steal a powerful weapon before it falls into the wrong hands...",
    label_visibility="collapsed"
)

# Word count live feedback
if user_story.strip():
    word_count = len(user_story.split())
    if word_count < 5:
        st.warning(f"⚠️ Only {word_count} words — add more detail for better recommendations.")
    elif word_count < 15:
        st.info(f"ℹ️ {word_count} words — decent, but more detail improves accuracy.")
    else:
        st.success(f"✅ {word_count} words — great detail for accurate recommendations!")

st.markdown("<br>", unsafe_allow_html=True)
search_btn = st.button("🔍 Find Similar Movies", type="primary")

# ═════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ═════════════════════════════════════════════════════════════════════════════
if search_btn:
    if not user_story.strip():
        st.warning("Please enter a storyline first.")
    else:
        with st.spinner("Analysing storyline and finding best matches..."):
            results = recommend_movies(user_story, top_n=top_n)

            # Apply minimum rating filter from sidebar
            if min_rating > 1.0:
                results = results[results['rating'] >= min_rating].reset_index(drop=True)
                results.index = results.index + 1

        if results.empty:
            st.error("No results found. Try a longer or more descriptive storyline.")
        else:
            st.markdown("---")
            st.markdown(
                f'<div class="section-header">🎯 Top {len(results)} Recommendations</div>',
                unsafe_allow_html=True
            )

            # ── Result cards + chart side by side ────────────────────────────
            left_col, right_col = st.columns([3, 2])

            with left_col:
                for rank, row in results.iterrows():
                    # Duration formatting
                    dur_mins = int(row['duration_mins'])
                    h = dur_mins // 60
                    m = dur_mins % 60
                    dur_str = f"{h}h {m}m" if h > 0 else f"{m}m"

                    # Storyline display (truncate if too long)
                    story = str(row['clean_storyline'])
                    story_display = story[:220] + "..." if len(story) > 220 else story

                    st.markdown(f"""
                    <div class="movie-card">
                        <span class="rank-badge">#{rank}</span>
                        {sim_pill(row['similarity_score'])}
                        <div class="movie-title">{row['title']}</div>
                        <div class="storyline-text">{story_display}</div>
                        <span class="meta-tag">⭐ {row['rating']}</span>
                        <span class="meta-tag">🕐 {dur_str}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with right_col:
                st.markdown("#### 📊 Similarity Score Comparison")

                # Bar chart
                fig, ax = plt.subplots(figsize=(6, len(results) * 0.85 + 1))
                fig.patch.set_facecolor('#0d1117')
                ax.set_facecolor('#161b22')

                titles_short = [t[:25] + "…" if len(t) > 25 else t
                                for t in results['title'].values]
                scores       = results['similarity_score'].values
                bar_colors   = ['#3fb950' if s >= 30 else '#ffa657' if s >= 15
                                else '#f78166' for s in scores]

                bars = ax.barh(titles_short[::-1], scores[::-1],
                               color=bar_colors[::-1], edgecolor='#0d1117',
                               height=0.6)

                for bar, score in zip(bars, scores[::-1]):
                    ax.text(bar.get_width() + 0.3,
                            bar.get_y() + bar.get_height() / 2,
                            f"{score}%", va='center',
                            color='#e6edf3', fontsize=9)

                ax.set_xlim(0, max(scores) * 1.35)
                ax.tick_params(axis='x', colors='#8b949e', labelsize=8)
                ax.tick_params(axis='y', colors='#e6edf3', labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#21262d')
                ax.grid(axis='x', color='#21262d', linewidth=0.5,
                        linestyle='--', alpha=0.7)
                ax.set_xlabel("Match %", color='#8b949e', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # ── Score table ──────────────────────────────────────────────
                st.markdown("#### 📋 Score Table")
                table_df = results[['title', 'rating', 'similarity_score']].copy()
                table_df.columns = ['Movie', 'Rating', 'Match %']
                table_df['Movie'] = table_df['Movie'].str[:28]
                st.dataframe(
                    table_df,
                    use_container_width=True,
                    hide_index=False
                )

# ═════════════════════════════════════════════════════════════════════════════
#  DATASET EXPLORER (collapsed by default)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("📊 Explore the Dataset", expanded=False):

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🏆 Top Movies", "🗂️ Browse Data"])

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#161b22')
            ax.hist(df['rating'].dropna(), bins=25,
                    color='#58a6ff', alpha=0.85, edgecolor='#0d1117')
            ax.axvline(df['rating'].mean(), color='#ffa657',
                       linestyle='--', linewidth=1.5,
                       label=f"Mean: {df['rating'].mean():.2f}")
            ax.legend(fontsize=8, labelcolor='#e6edf3', facecolor='#161b22')
            ax.tick_params(colors='#8b949e')
            ax.set_title("Rating Distribution", color='#e6edf3', fontweight='bold')
            ax.set_xlabel("IMDb Rating", color='#8b949e', fontsize=9)
            ax.set_ylabel("Count",       color='#8b949e', fontsize=9)
            for spine in ax.spines.values(): spine.set_edgecolor('#21262d')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#161b22')
            dur = df['duration_mins'][(df['duration_mins'] > 30) &
                                       (df['duration_mins'] < 240)]
            ax.hist(dur, bins=30, color='#ffa657', alpha=0.85, edgecolor='#0d1117')
            ax.axvline(dur.mean(), color='#f78166',
                       linestyle='--', linewidth=1.5,
                       label=f"Mean: {dur.mean():.0f} min")
            ax.legend(fontsize=8, labelcolor='#e6edf3', facecolor='#161b22')
            ax.tick_params(colors='#8b949e')
            ax.set_title("Duration Distribution", color='#e6edf3', fontweight='bold')
            ax.set_xlabel("Minutes", color='#8b949e', fontsize=9)
            ax.set_ylabel("Count",   color='#8b949e', fontsize=9)
            for spine in ax.spines.values(): spine.set_edgecolor('#21262d')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**🔥 Most Popular (by votes)**")
            top_votes = (df[df['vote_count'] > 0]
                         .nlargest(10, 'vote_count')[['title', 'rating', 'vote_count']]
                         .reset_index(drop=True))
            top_votes.index = top_votes.index + 1
            top_votes['vote_count'] = top_votes['vote_count'].apply(
                lambda x: f"{int(x/1000)}K" if x >= 1000 else str(int(x))
            )
            top_votes.columns = ['Title', 'Rating', 'Votes']
            st.dataframe(top_votes, use_container_width=True)

        with c2:
            st.markdown("**⭐ Highest Rated (≥ 1K votes)**")
            top_rated = (df[df['vote_count'] >= 1000]
                         .nlargest(10, 'rating')[['title', 'rating', 'duration_mins']]
                         .reset_index(drop=True))
            top_rated.index = top_rated.index + 1
            top_rated['duration_mins'] = top_rated['duration_mins'].apply(
                lambda x: f"{int(x//60)}h {int(x%60)}m"
            )
            top_rated.columns = ['Title', 'Rating', 'Duration']
            st.dataframe(top_rated, use_container_width=True)

    with tab3:
        st.markdown("**🔎 Search and browse all movies**")
        search_term = st.text_input("Search by title:", placeholder="e.g. Dune")
        if search_term:
            filtered = df[df['title'].str.contains(search_term, case=False, na=False)]
        else:
            filtered = df

        show_cols = ['title', 'rating', 'duration_mins', 'vote_count']
        # Only show columns that exist
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(
            filtered[show_cols].reset_index(drop=True),
            use_container_width=True,
            height=350
        )
        st.caption(f"Showing {len(filtered):,} movies")
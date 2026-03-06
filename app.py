
import streamlit as st
import pandas as pd
import numpy as np
from difflib import get_close_matches
from sklearn.metrics.pairwise import cosine_similarity


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Song Recommender",
    page_icon="🎵",
    layout="wide",
)

# ── Custom CSS (Black Theme) ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: #000000;
        min-height: 100vh;
        color: #ffffff;
    }

    /* Center the main content */
    .block-container {
        max-width: 700px !important;
        margin: 0 auto;
        padding-top: 3rem !important;
    }

    /* Song result card */
    .song-card {
        background: #0a0a0a;
        border: 1px solid #1f1f1f;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.7rem;
        transition: all 0.2s ease;
        cursor: default;
    }
    .song-card:hover {
        border-color: #333333;
        background: #111111;
        transform: translateX(4px);
    }
    .song-rank {
        font-size: 1.1rem;
        font-weight: 700;
        color: #666666;
        min-width: 2rem;
        display: inline-block;
    }
    .song-name {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
    }
    .song-artist {
        font-size: 0.82rem;
        color: #888888;
        margin-top: 0.15rem;
    }

    /* Header */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
        text-align: center;
    }
    .hero-sub {
        color: #888888;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }



    /* Container generic styling */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background: #0a0a0a;
        border: 1px solid #1f1f1f;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Buttons */
    .stButton > button {
        background: #ffffff;
        color: #000000;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #dddddd;
        color: #000000;
        transform: translateY(-1px);
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background: #0a0a0a !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        color: white !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    .badge {
        display: inline-block;
        background: #111111;
        border: 1px solid #333333;
        color: #aaaaaa;
        font-size: 0.72rem;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        margin-bottom: 1.2rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-center {
        text-align: center;
    }

    /* Center the Get Recommendations button */
    .stButton {
        display: flex;
        justify-content: center;
    }
    .stButton > button {
        width: auto;
        min-width: 250px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load dataset (same logic as notebook) ─────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    df = pd.read_csv("spotify_tracks.csv")
    df["track_name"] = df["track_name"].str.lower().str.strip()
    df["album_name"] = df["album_name"].str.lower().str.strip()
    df["artist_name"] = df["artist_name"].str.lower().str.strip()
    df["language"] = df["language"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["track_id"])
    return df

# ── Features (same as notebook) ──────────────────────────────────────────────
features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

df = load_dataset()

# ── Session state defaults ───────────────────────────────────────────────────
if "picked_suggestion" not in st.session_state:
    st.session_state["picked_suggestion"] = None
if "fuzzy_suggestions" not in st.session_state:
    st.session_state["fuzzy_suggestions"] = None

# ── Helper: show recommendations ─────────────────────────────────────────────
def show_recommendations(song_name_in_df):
    """Given an exact track_name from the dataframe, display recommendations."""
    matched = df[df["track_name"] == song_name_in_df]
    if matched.empty:
        st.warning("⚠️ Song not in dataset. Try a different song.")
        return

    chosen_track_name = matched.iloc[0]["track_name"]

    st.success(
        f"✅ Found: **{matched.iloc[0]['track_name'].title()}** — "
        f"{matched.iloc[0]['artist_name'].title()}"
    )

    # ── Recommend (same as notebook) ─────────────────────────────
    song_vector = matched.iloc[[0]][features].values
    song_language = matched.iloc[0]["language"]

    # Filter by same language
    filtered_df = df[df["language"] == song_language]

    # Exclude the chosen song by track_id and any name variants
    input_base_name = matched.iloc[0]["track_name"].split("(")[0].strip()
    other_songs = filtered_df[
        (filtered_df["track_id"] != matched.iloc[0]["track_id"]) &
        (~filtered_df["track_name"].str.startswith(input_base_name, na=False))
    ].copy()

    other_features = other_songs[features].values
    sim_scores = cosine_similarity(song_vector, other_features)[0]

    top_idx = np.argsort(sim_scores)[::-1][:30]

    candidates = other_songs.iloc[top_idx][
        ["track_name", "artist_name", "album_name"]
    ].copy()
    candidates["_base_name"] = candidates["track_name"].str.split("(").str[0].str.strip()
    recommendations = candidates.drop_duplicates(subset=["_base_name"]).drop(columns=["_base_name"]).head(10).reset_index(drop=True)

    st.markdown(
        f"#### 🎧 Top 10 songs similar to "
        f"'{chosen_track_name.title()}' ({song_language})"
    )
    st.divider()

    for i, row in recommendations.iterrows():
        st.markdown(f"""
        <div class="song-card">
            <span class="song-rank">{i+1}</span>&nbsp;&nbsp;
            <span class="song-name">{row['track_name'].title()}</span>
            <div class="song-artist" style="padding-left:2.5rem">
                {row['artist_name'].title()} &nbsp;·&nbsp; {row['album_name'].title()}
            </div>
        </div>""", unsafe_allow_html=True)


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="badge-center"><span class="badge">ML-Powered</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Song Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Discover music that matches your vibe using cosine similarity on Spotify audio features.</div>', unsafe_allow_html=True)

# ───────────────── SINGLE SONG SEARCH ─────────────────
with st.container():
    st.markdown("#### 🎵 Search for a Song")
    song_input = st.text_input(
        "Song name",
        placeholder="e.g. kun faya kun",
        label_visibility="collapsed"
    )
    artist_input = st.text_input(
        "Artist name (optional)",
        placeholder="Artist name (optional, press Enter to skip)",
        label_visibility="collapsed"
    )

    get_rec_click = st.button("Get Recommendations")

# ── When user clicks "Get Recommendations" ────────────────────────────────────
if get_rec_click and song_input:
    # Clear previous state for a fresh search
    st.session_state["picked_suggestion"] = None
    st.session_state["fuzzy_suggestions"] = None

    song_name = song_input.strip().lower()
    artist_name = artist_input.strip().lower() if artist_input else ""

    # ── Search logic (same as notebook) ───────────────────────────────────
    if artist_name:
        song_match = df[
            (df["track_name"] == song_name) &
            (df["artist_name"] == artist_name)
        ]
    else:
        song_match = df[df["track_name"] == song_name]

    if song_match.empty:
        # Fallback: try partial match
        song_match = df[df["track_name"].str.contains(song_name, na=False)]

    if song_match.empty:
        # ── "Did you mean?" fuzzy suggestions ─────────────────────────
        all_names = df["track_name"].unique().tolist()
        suggestions = get_close_matches(song_name, all_names, n=5, cutoff=0.5)
        if suggestions:
            st.session_state["fuzzy_suggestions"] = suggestions
        else:
            st.error("❌ Song not found in dataset. Try a different name.")
    else:
        # Direct match found → show recommendations immediately
        st.session_state["picked_suggestion"] = song_match.iloc[0]["track_name"]
        st.rerun()

# ── Show fuzzy suggestion buttons (persists across reruns) ────────────────────
if st.session_state["fuzzy_suggestions"] and not st.session_state["picked_suggestion"]:
    st.warning("⚠️ Song not found. Did you mean one of these?")
    for idx, s in enumerate(st.session_state["fuzzy_suggestions"]):
        artist = df[df["track_name"] == s].iloc[0]["artist_name"]
        if st.button(f"🎵 {s.title()} — {artist.title()}", key=f"suggest_{idx}"):
            st.session_state["picked_suggestion"] = s
            st.session_state["fuzzy_suggestions"] = None
            st.rerun()

# ── Show recommendations for picked song ─────────────────────────────────────
if st.session_state["picked_suggestion"]:
    show_recommendations(st.session_state["picked_suggestion"])

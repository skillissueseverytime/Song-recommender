
import streamlit as st
import pandas as pd
import numpy as np
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
    }
    .hero-sub {
        color: #888888;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 1.5rem;
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

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="badge">ML-Powered</div>', unsafe_allow_html=True)
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

if get_rec_click and song_input:
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
        # Fallback: try partial match (same as notebook cell "single_song_pick")
        song_match = df[df["track_name"].str.contains(song_name, na=False)]

    if song_match.empty:
        st.error("❌ Song not found in dataset. Try a different song.")
    else:
        chosen_song = song_match.iloc[0]
        chosen_track_id = chosen_song["track_id"]
        chosen_track_name = chosen_song["track_name"]

        # ── Match by track_id (same as notebook) ─────────────────────────
        matched = df[df["track_id"] == chosen_track_id]

        if matched.empty:
            matched = df[df["track_name"].str.lower() == chosen_track_name.lower()]

        if matched.empty:
            st.warning("⚠️ Song not in dataset. Try a different song.")
        else:
            st.success(
                f"✅ Found: **{matched.iloc[0]['track_name'].title()}** — "
                f"{matched.iloc[0]['artist_name'].title()}"
            )

            # ── Recommend (same as notebook) ─────────────────────────────
            song_vector = matched.iloc[[0]][features].values
            song_language = matched.iloc[0]["language"]

            # Filter by same language
            filtered_df = df[df["language"] == song_language]

            # Exclude the chosen song itself
            other_songs = filtered_df[
                filtered_df["track_name"] != matched.iloc[0]["track_name"]
            ].copy()

            other_features = other_songs[features].values

            sim_scores = cosine_similarity(song_vector, other_features)[0]

            top_idx = np.argsort(sim_scores)[::-1][1:11]

            recommendations = other_songs.iloc[top_idx][
                ["track_name", "artist_name", "album_name"]
            ].drop_duplicates(subset=["track_name"]).head(10).reset_index(drop=True)

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

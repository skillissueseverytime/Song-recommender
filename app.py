import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Song Recommender",
    page_icon="🎵",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* Main card */
    .main-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }

    /* Song result card */
    .song-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(138,43,226,0.3);
        border-radius: 14px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.7rem;
        transition: all 0.2s ease;
        cursor: default;
    }
    .song-card:hover {
        border-color: rgba(138,43,226,0.8);
        background: rgba(138,43,226,0.1);
        transform: translateX(4px);
    }
    .song-rank {
        font-size: 1.1rem;
        font-weight: 700;
        color: #9b59b6;
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
        color: rgba(255,255,255,0.55);
        margin-top: 0.15rem;
    }

    /* Header */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a855f7, #6366f1, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    .hero-sub {
        color: rgba(255,255,255,0.55);
        font-size: 1.05rem;
        margin-top: 0.4rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
        color: rgba(255,255,255,0.6);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #a855f7, #6366f1) !important;
        color: white !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #a855f7, #6366f1);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(138,43,226,0.4);
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: white !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    .badge {
        display: inline-block;
        background: rgba(138,43,226,0.2);
        border: 1px solid rgba(138,43,226,0.5);
        color: #c084fc;
        font-size: 0.72rem;
        padding: 2px 10px;
        border-radius: 20px;
        font-weight: 600;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load credentials & Spotify client ────────────────────────────────────────
load_dotenv()

@st.cache_resource
def get_spotify():
    client_id     = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        st.error("❌ SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET not found in .env")
        st.stop()
    auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=auth)

@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    df = pd.read_csv("dataset.csv")
    df["track_name"] = df["track_name"].str.lower()
    df["album_name"] = df["album_name"].str.lower()
    df["artists"]    = df["artists"].str.lower()
    df = df.drop_duplicates(subset=["track_id"])
    return df

FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

sp = get_spotify()
df = load_dataset()


# ── Helper functions ──────────────────────────────────────────────────────────
def recommend_from_vector(seed_vector, exclude_ids, top_n=10):
    """Given a feature vector (1, 9), return top_n similar songs excluding seed."""
    pool = df[~df["track_id"].isin(exclude_ids)].copy()
    sims = cosine_similarity(seed_vector, pool[FEATURES].values)[0]
    top_idx = np.argsort(sims)[::-1][:top_n]
    results = pool.iloc[top_idx][["track_name", "artists", "album_name"]].copy()
    results["similarity"] = sims[top_idx]
    return results.reset_index(drop=True)


def fetch_playlist_songs(playlist_id):
    """Fetch tracks from a Spotify playlist and match them to the dataset."""
    results = sp.playlist_items(playlist_id)
    songs = []
    for item in results["items"]:
        track = item.get("item")
        if track:
            artists = ", ".join([a["name"] for a in track["artists"]])
            songs.append({
                "track_name": track["name"].lower(),
                "album_name": track["album"]["name"].lower(),
                "artists":    artists.lower(),
                "track_id":   track["id"],
            })
    playlist_df = pd.DataFrame(songs)

    # Filter: only tracks present in dataset
    matched = df[
        df.set_index(["track_name", "album_name", "artists"]).index.isin(
            playlist_df.set_index(["track_name", "album_name", "artists"]).index
        )
    ].drop_duplicates(subset=["track_id"])
    return playlist_df, matched


def search_song(name, artist=""):
    query = f"track:{name}"
    if artist:
        query += f" artist:{artist}"
    res = sp.search(q=query, type="track", limit=5)
    return res["tracks"]["items"]


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="badge">🎛️ ML-Powered</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Song Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Discover music that matches your vibe using cosine similarity on Spotify audio features</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎵 Single Song", "📋 Playlist"])

# ───────────────── TAB 1: SINGLE SONG ─────────────────
with tab1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("#### Enter a song to get similar recommendations")

    col1, col2 = st.columns([2, 1])
    with col1:
        song_name = st.text_input("🎤 Song Name", placeholder="e.g. Brown Munde")
    with col2:
        artist_name = st.text_input("👤 Artist (optional)", placeholder="e.g. AP Dhillon")

    search_clicked = st.button("🔍 Search Song", key="search_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    if search_clicked and song_name:
        with st.spinner("Searching Spotify..."):
            tracks = search_song(song_name, artist_name)

        if not tracks:
            st.error("❌ No song found. Try a different name.")
        else:
            st.markdown("**Select the correct song:**")
            options = {
                f"{t['name']} — {', '.join(a['name'] for a in t['artists'])}": t
                for t in tracks
            }
            chosen_label = st.selectbox("", list(options.keys()), label_visibility="collapsed")
            chosen = options[chosen_label]
            chosen_id = chosen["id"]
            chosen_name = chosen["name"]

            if st.button("🎧 Get Recommendations", key="rec_btn"):
                with st.spinner("Finding similar songs..."):
                    matched = df[df["track_id"] == chosen_id]
                    if matched.empty:
                        matched = df[df["track_name"] == chosen_name.lower()]

                    if matched.empty:
                        st.warning("⚠️ This song isn't in the local dataset. Try another.")
                    else:
                        seed_vector = matched.iloc[[0]][FEATURES].values
                        recs = recommend_from_vector(seed_vector, [chosen_id])

                        st.success(f"Top 10 songs similar to **{chosen_name}**")
                        st.divider()
                        for i, row in recs.iterrows():
                            st.markdown(f"""
                            <div class="song-card">
                                <span class="song-rank">#{i+1}</span>&nbsp;&nbsp;
                                <span class="song-name">{row['track_name'].title()}</span>
                                <div class="song-artist" style="padding-left:2.5rem">
                                    {row['artists'].title()} &nbsp;·&nbsp; {row['album_name'].title()}
                                </div>
                            </div>""", unsafe_allow_html=True)


# ───────────────── TAB 2: PLAYLIST ─────────────────
with tab2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("#### Paste your Spotify Playlist ID")
    st.markdown(
        '<small style="color:rgba(255,255,255,0.4)">Open playlist on Spotify → Share → Copy link → '
        'grab the ID after <code>/playlist/</code></small>',
        unsafe_allow_html=True
    )

    playlist_id = st.text_input("🔗 Playlist ID", placeholder="e.g. 7zXhVAENGtOlG6Mnwk7bwv", label_visibility="collapsed")
    playlist_btn = st.button("🎶 Recommend from Playlist", key="playlist_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    if playlist_btn and playlist_id:
        with st.spinner("Fetching playlist & computing recommendations..."):
            try:
                playlist_df, matched = fetch_playlist_songs(playlist_id.strip())
                if matched.empty:
                    st.warning("⚠️ No songs from your playlist were found in the local dataset.")
                else:
                    playlist_vector = matched[FEATURES].mean().values.reshape(1, -1)
                    recs = recommend_from_vector(playlist_vector, matched["track_id"].tolist())

                    st.success(f"Found **{len(matched)}** matched songs from your playlist → Top 10 recommendations:")
                    st.divider()
                    for i, row in recs.iterrows():
                        st.markdown(f"""
                        <div class="song-card">
                            <span class="song-rank">#{i+1}</span>&nbsp;&nbsp;
                            <span class="song-name">{row['track_name'].title()}</span>
                            <div class="song-artist" style="padding-left:2.5rem">
                                {row['artists'].title()} &nbsp;·&nbsp; {row['album_name'].title()}
                            </div>
                        </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error: {e}")

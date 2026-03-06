import os
import re
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

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #0a0a0a;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
        color: #888888;
        font-weight: 500;
        border: none !important;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #000000 !important;
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


# ── Load credentials & Spotify client ────────────────────────────────────────
load_dotenv()

@st.cache_resource
def get_spotify():
    # Try fetching from os.environ first, then fallback to Streamlit secrets
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    if not client_id and "SPOTIFY_CLIENT_ID" in st.secrets:
        client_id = st.secrets["SPOTIFY_CLIENT_ID"]

    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_secret and "SPOTIFY_CLIENT_SECRET" in st.secrets:
        client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]

    if not client_id or not client_secret:
        st.error("❌ API Keys missing! Please add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to Streamlit Secrets.")
        st.stop()
        
    client_id = str(client_id).strip()
    client_secret = str(client_secret).strip()

    try:
        auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(auth_manager=auth)
        # Try a dummy request to force authentication right now rather than later
        sp.search(q="test", limit=1)
        return sp
    except Exception as e:
        import traceback
        st.error(f"❌ Failed to connect to Spotify. Are your keys valid?")
        st.code(traceback.format_exc(), language="python")
        st.stop()

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

# ── Session State initialization for Single Song Search ──
if "search_results" not in st.session_state:
    st.session_state.search_results = None


# ── Helper functions ──────────────────────────────────────────────────────────
def extract_id_from_url(url, type_of_id="playlist"):
    """
    Returns the exact ID the user pasted, trimming whitespace only.
    """
    return url.strip()


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
    try:
        results = sp.playlist_items(playlist_id)
    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 404:
            raise ValueError("Playlist not found! Make sure the playlist is PUBLIC, not private.")
        else:
            raise ValueError(f"Spotify API Error: {e.msg}")
            
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
    
    if not songs:
        return pd.DataFrame(), pd.DataFrame()
        
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
st.markdown('<div class="badge">ML-Powered</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Song Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Discover music that matches your vibe using cosine similarity on Spotify audio features.</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Song", "Spotify Playlist"])

# ───────────────── TAB 1: SINGLE SONG ─────────────────
with tab1:
    with st.container():
        st.markdown("#### Search or paste a Track Link")
        song_input = st.text_input("Song Name or Spotify URL", placeholder="e.g. Brown Munde OR https://open.spotify.com/track/...", label_visibility="collapsed")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            get_rec_click = st.button("Get Recommendations")

    # If the user pasted a Spotify URL, bypass search and go straight to recommending
    if get_rec_click and song_input:
        if "spotify.com/track/" in song_input:
            track_id = extract_id_from_url(song_input, "track")
            with st.spinner("Finding similar songs..."):
                try:
                    track_info = sp.track(track_id)
                    chosen_name = track_info["name"]
                    
                    matched = df[df["track_id"] == track_id]
                    if matched.empty:
                        matched = df[df["track_name"] == chosen_name.lower()]

                    if matched.empty:
                        st.warning(f"⚠️ **{chosen_name}** isn't in the local dataset. Try searching by name or pasting another song.")
                    else:
                        seed_vector = matched.iloc[[0]][FEATURES].values
                        recs = recommend_from_vector(seed_vector, [track_id])
                        st.success(f"Top 10 songs similar to **{chosen_name}**")
                        for i, row in recs.iterrows():
                            st.markdown(f"""
                            <div class="song-card">
                                <span class="song-rank">{i+1}</span>&nbsp;&nbsp;
                                <span class="song-name">{row['track_name'].title()}</span>
                                <div class="song-artist" style="padding-left:2.5rem">
                                    {row['artists'].title()} &nbsp;·&nbsp; {row['album_name'].title()}
                                </div>
                            </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error("Invalid Spotify Track URL.")
                    
        # Otherwise, they entered a search querying text (e.g. "Brown Munde")
        else:
            with st.spinner("Searching Spotify..."):
                # Clear previous search results so we do a fresh search
                st.session_state.search_results = search_song(song_input)
                
    # If we have search results stored in session state, display them so the user can pick
    if st.session_state.search_results is not None and "spotify.com/track/" not in str(song_input):
        if len(st.session_state.search_results) == 0:
            st.error("No song found. Try a different name.")
        else:
            st.markdown("##### 🔍 Select the correct match:")
            options = {
                f"{t['name']} — {', '.join(a['name'] for a in t['artists'])}": t
                for t in st.session_state.search_results
            }
            
            chosen_label = st.selectbox("Select match", list(options.keys()), label_visibility="collapsed")
            chosen = options[chosen_label]
            chosen_id = chosen["id"]
            chosen_name = chosen["name"]

            if st.button("Confirm & Recommend", type="primary"):
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
                                <span class="song-rank">{i+1}</span>&nbsp;&nbsp;
                                <span class="song-name">{row['track_name'].title()}</span>
                                <div class="song-artist" style="padding-left:2.5rem">
                                    {row['artists'].title()} &nbsp;·&nbsp; {row['album_name'].title()}
                                </div>
                            </div>""", unsafe_allow_html=True)


# ───────────────── TAB 2: PLAYLIST ─────────────────
with tab2:
    with st.container():
        st.markdown("#### Paste your Spotify Playlist Link")
        st.markdown(
            '<small style="color:#888888">Open playlist on Spotify → Share → Copy link</small>',
            unsafe_allow_html=True
        )

        playlist_input = st.text_input("Playlist URL", placeholder="https://open.spotify.com/playlist/...", label_visibility="collapsed")
        playlist_btn = st.button("Recommend from Playlist")

    if playlist_btn and playlist_input:
        playlist_id = extract_id_from_url(playlist_input, "playlist")
        with st.spinner("Fetching playlist & computing recommendations..."):
            try:
                playlist_df, matched = fetch_playlist_songs(playlist_id)
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
                            <span class="song-rank">{i+1}</span>&nbsp;&nbsp;
                            <span class="song-name">{row['track_name'].title()}</span>
                            <div class="song-artist" style="padding-left:2.5rem">
                                {row['artists'].title()} &nbsp;·&nbsp; {row['album_name'].title()}
                            </div>
                        </div>""", unsafe_allow_html=True)

            except Exception as e:
                import traceback
                st.error("Invalid Playlist URL or Spotify API error.")

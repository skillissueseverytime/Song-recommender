# 🎵 Song Recommender

A machine learning-based song recommendation system that suggests similar songs based on audio features using **cosine similarity**.

🔗 **Live Demo:** [recommend-me-some.streamlit.app](https://recommend-me-some.streamlit.app)

## How It Works

1. Enter a song name (and optionally an artist name)
2. The app finds the song in the dataset
3. Filters songs by the **same language**
4. Computes **cosine similarity** on 9 audio features
5. Returns the **top 10 most similar songs**

### 🔍 Fuzzy Search

Misspelled the song name? No problem — the app suggests close matches using fuzzy matching so you can pick the right one.

## 🔧 Audio Features Used

| Feature | Description |
|---------|-------------|
| Danceability | How suitable a track is for dancing |
| Energy | Intensity and activity level |
| Loudness | Overall loudness in dB |
| Speechiness | Presence of spoken words |
| Acousticness | Whether the track is acoustic |
| Instrumentalness | Whether the track has no vocals |
| Liveness | Presence of a live audience |
| Valence | Musical positiveness (happy vs sad) |
| Tempo | Speed of the track in BPM |

## 📂 Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app |
| `pre.ipynb` | Jupyter notebook with the recommendation pipeline |
| `spotify_tracks.csv` | Song dataset with audio features (~62K tracks) |
| `requirements.txt` | Python dependencies |

## ⚙️ Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/skillissueseverytime/Song-recommender.git
   cd Song-recommender
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset

The dataset (`spotify_tracks.csv`) contains ~62K songs with pre-computed Spotify audio features including:

`track_id`, `track_name`, `artist_name`, `album_name`, `language`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`

## 🛠️ Tech Stack

- **Python** — Core language
- **Streamlit** — Web UI
- **pandas** — Data processing
- **scikit-learn** — Cosine similarity
- **difflib** — Fuzzy matching for search suggestions

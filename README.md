# Song Recommendation System using ML

A machine learning-based song recommendation system that analyzes your Spotify playlist and recommends songs with similar audio features from a large dataset.

## 🎵 How It Works

1. **Fetches your Spotify playlist** using the Spotify Web API (`spotipy`)
2. **Matches songs** from the playlist to a dataset with audio features
3. **Computes cosine similarity** between your playlist's average audio profile and every song in the dataset
4. **Recommends the top 10 songs** most similar to your taste

## 🔧 Audio Features Used

- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo

## 📂 Files

| File | Description |
|------|-------------|
| `pre.ipynb` | Main Jupyter notebook with the recommendation pipeline |
| `dataset.csv` | Large song dataset with audio features *(not included in repo — too large)* |

## ⚙️ Setup

1. Install dependencies:
   ```
   pip install pandas scikit-learn spotipy numpy
   ```

2. Set your Spotify API credentials in the notebook:
   ```python
   client_id = "YOUR_CLIENT_ID"
   client_secret = "YOUR_CLIENT_SECRET"
   ```

3. Set your playlist ID:
   ```python
   playlist_id = "YOUR_PLAYLIST_ID"
   ```

4. Download the dataset and place it as `dataset.csv` in the same directory.

5. Run all cells in `pre.ipynb`.

## 📊 Dataset

The dataset (`dataset.csv`) contains ~100K+ songs with pre-computed Spotify audio features. It is not included in this repository due to its size (~20MB). You can use any Spotify track dataset with the following columns:
`track_id`, `track_name`, `artists`, `album_name`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`

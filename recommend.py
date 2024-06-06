import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
import librosa
warnings.filterwarnings("ignore")
client_id = 'e42da97cd842452a8e2b8a62023e9a55'
client_secret = 'd881b55843c0432da08786e313735804'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
def audio_recc():
    df = pd.read_csv('main_data.csv')
    file_path = 'audio.wav'
    y, sr = librosa.load(file_path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env)
    energy = tempogram.mean()
    loudness = librosa.feature.rms(y=y)
    loudness = max(loudness[0])
    valence = librosa.effects.harmonic(y=y)
    valence = max(valence)
    acousticness = librosa.feature.spectral_centroid(y=y)
    acousticness = (acousticness[0].mean() - min(acousticness[0])) / (max(acousticness[0]) - min(acousticness[0]))
    danceability = librosa.feature.spectral_bandwidth(y=y)
    danceability = (danceability[0].mean() - min(danceability[0])) / (max(danceability[0]) - min(danceability[0]))
    fea=[danceability, loudness, acousticness, valence, energy]
    df['all'] = df[['danceability', 'loudness', 'acousticness', 'valence', 'energy']].values.tolist()
    x = np.array(df['all'].tolist())
    cosine_similarities = cosine_similarity([fea], x)
    similar_indices = np.argsort(cosine_similarities[0])[::-1][:6]
    res=[]
    for i in range(6):
        res.append(classify(df['track_name'].iloc[similar_indices[i]].upper()))
    return res
def get_val(name):
    results = sp.search(q=name, type='track', limit=1)
    track_uri = results['tracks']['items'][0]['uri']
    audio_features = sp.audio_features([track_uri])
    danceability = audio_features[0]['danceability']
    loudness = audio_features[0]['loudness']
    acousticness = audio_features[0]['acousticness']
    valence = audio_features[0]['valence']
    energy = audio_features[0]['energy']
    return [danceability,loudness,acousticness,valence,energy]
def classify(name):
    results = sp.search(q=name, type='track',limit=1)
    artist_name = results['tracks']['items'][0]['album']['artists'][0]['name']
    image = results['tracks']['items'][0]['album']['images'][0]['url']
    prev_link = results['tracks']['items'][0]['external_urls']['spotify']
    return [name.upper(),artist_name.upper(),image,prev_link]
def recommendation(name):
    fea=get_val(name.upper())
    res=[]
    res.append(classify(name))
    df=pd.read_csv('main_data.csv')
    df=df.drop(['Unnamed: 0',"Unnamed: 9",'Unnamed: 10'],axis=1)
    df['all'] = df[['danceability', 'loudness', 'acousticness', 'valence', 'energy']].values.tolist()
    x = np.array(df['all'].tolist())
    cosine_similarities = cosine_similarity([fea], x)
    similar_indices = np.argsort(cosine_similarities[0])[::-1][:5]
    for i in range(5):
        res.append(classify(df['track_name'].iloc[similar_indices[i]].upper()))
    return res
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import warnings
import pickle
warnings.filterwarnings("ignore")
label_encoder=LabelEncoder()
df=pd.read_csv('tcc_ceds_music.csv')
df=df[['track_name','danceability','acousticness','instrumentalness','valence','energy','genre']]
df=df.dropna()
df['genre']= label_encoder.fit_transform(df['genre'])
with open('test_model.pkl', 'rb') as model_file:
    model=pickle.load(model_file)
def classify(name):
    client_id = 'e42da97cd842452a8e2b8a62023e9a55'
    client_secret = 'd881b55843c0432da08786e313735804'
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    results = sp.search(q=name, type='track', limit=1)
    track_uri = results['tracks']['items'][0]['uri']
    audio_features = sp.audio_features([track_uri])
    danceability = audio_features[0]['danceability']
    instrumentalness = audio_features[0]['instrumentalness']
    acousticness = audio_features[0]['acousticness']
    valence = audio_features[0]['valence']
    energy = audio_features[0]['energy']
    res=model.predict([[danceability,acousticness,instrumentalness,valence,energy]])
    genre=label_encoder.inverse_transform(res)[0]
    return genre
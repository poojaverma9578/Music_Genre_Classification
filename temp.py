import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.preprocessing import LabelEncoder
import warnings
encoder=LabelEncoder()
warnings.filterwarnings("ignore")
df=pd.read_csv('tcc_ceds_music.csv')
df=df[['track_name','danceability','acousticness','instrumentalness','valence','energy','genre']]
df.genre=encoder.fit_transform(df['genre'])
df=df[df.genre.isin([0,2,3])==False]
x=df[['danceability','acousticness','instrumentalness','valence','energy']]
y=df.genre
model=KNeighborsClassifier(n_neighbors=1)
model.fit(x,y)
with open ('test_model.pkl','wb') as file:
    pickle.dump(model,file)

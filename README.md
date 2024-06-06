The model uses 2 datasets. Firstly, tcc_music_ceds dataset which can be downloaded from kaggle and secondly it is the main_data.csv file uploaded in the git itself.
The model can be explained using 2 files, that is, main_knn.ipynb and music_classification_sudiofiles.ipynb. 
The first one explains the first modeule of the project where the audio name is given to the model and using spotify API it extracts features from that audio file and then the model generates its genre and recommendations. 
The other model is ANN which takes in the audio file itself and classifies it after extracting features using MFCC.
The frontend uses a server.js file to interact with the model integrated with a flask server at the backend.

import keras
import numpy as np
import librosa
actual_labels=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
def mfcc_process():
    data,sample_rate=librosa.load('audio.wav')
    mfcc=librosa.feature.mfcc(y=data,sr=sample_rate,n_mfcc=40)
    mfcc_processed=np.mean(mfcc.T,axis=0)
    model = keras.models.load_model('yoyo.keras')
    testing = mfcc_processed
    testing = np.array(testing.tolist())
    testing_reshaped = np.reshape(testing, (1, 40))
    predictions = np.argmax(model.predict(testing_reshaped))
    return (actual_labels[predictions])
print(mfcc_process())
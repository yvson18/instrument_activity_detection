import os
import json
import librosa
import numpy as np
from sklearn import preprocessing

def amplitude_envelope(signal, frame_size= 1024, hop_length = 512):
    amplitude_envelope = []
    #calculate AE for each frame
    for i in range(0, len(signal), hop_length):
        current_frame_amplitude_envelope = max(signal[i:i+frame_size])
        amplitude_envelope.append(current_frame_amplitude_envelope)

    return np.array(amplitude_envelope)

def create_normalized_envelope(signal):
    return preprocessing.MinMaxScaler().fit_transform(np.array(signal).reshape(-1,1)).reshape(1,-1)[0]

def stretch_sample(array, new_length):
    original_array = array
    old_indices = np.arange(len(original_array))
    new_indices = np.linspace(0, len(original_array) - 1, new_length)
    expanded_array = np.interp(new_indices, old_indices, original_array)
    return expanded_array

def ae_per_time_windows(amplitude_envelope, sr, time_window = 1):
    activations = []
    for i in range(0, len(amplitude_envelope), int(sr * time_window)):
        activations.append(max(amplitude_envelope[i: int(i + sr * time_window)]))
    return activations

def process_activations(signal, sr, time_window = 1):
    
    ae = amplitude_envelope(signal)
    normalized_envelope = create_normalized_envelope(ae)
    stretched_envelope = stretch_sample(normalized_envelope, len(signal))
    activations = ae_per_time_windows(stretched_envelope, sr, time_window)

    return activations

def calculate_mfccs(signal, sr, n_mfcc = 13, n_fft = 2048, hop_length = 512):
    mfccs = librosa.feature.mfcc(signal, sr, n_fft = n_fft, hop_length = hop_length, n_mfcc = n_mfcc)
    return mfccs.T


def process_mfccs(signal, sr, time_window = 1):
    mfccs = []
    for i in range(0, len(signal), int(sr * time_window)):
        mfccs.append(calculate_mfccs(signal[i: int(i + sr * time_window)], sr).tolist())
    return mfccs



def main(output_file, mixture_path, drums_path, bass_path, rest_path, vocals_path):

    song_names = [arquivo for arquivo in os.listdir(mixture_path) if os.path.isfile(os.path.join(mixture_path, arquivo))]; song_names.sort()
    
    drums_activation = []
    bass_activation = []
    rest_activation = []
    vocals_activation = []

    mfccs = []
    i = 0
    for song_path in song_names:
        print(f"Processing song {i + 1} of {len(song_names)}")
        
        # calculate activations for each stem
        drums_signal, sr = librosa.load(drums_path + song_path)
        drums_activation += list(process_activations(drums_signal, sr))

        bass_signal, _ = librosa.load(bass_path + song_path)
        bass_activation += list(process_activations(bass_signal, sr))
        
        rest_signal, _ = librosa.load(rest_path + song_path)
        rest_activation += list(process_activations(rest_signal, sr))

        vocals_signal, _ = librosa.load(vocals_path + song_path)
        vocals_activation += list(process_activations(vocals_signal, sr))

        # calculate mfccs for each frame
        mixture_signal, sr = librosa.load(mixture_path + song_path)
        mfccs.append(process_mfccs(mixture_signal ,sr))
        i+=1

    activations = np.array([drums_activation, bass_activation, rest_activation, vocals_activation]).T.tolist()
   
    mfccs_agrregated = []
    
    for m in mfccs:
         mfccs_agrregated+= m

    data = {
        "targets": ["drums", "bass", "rest", "vocals"],
        "activations" : activations,
        "mfccs": mfccs_agrregated
    }

    with open(output_file, 'w') as fp:
        json.dump(data, fp, indent=4)

if(__name__ == "__main__"):
    type_ = "validation"

    main(
        f"data_{type_}.json",
        f"full_dataset_wav/{type_}/mixture/",
        f"full_dataset_wav/{type_}/drums/",
        f"full_dataset_wav/{type_}/bass/" ,
        f"full_dataset_wav/{type_}/rest/",
        f"full_dataset_wav/{type_}/vocals/"
    )


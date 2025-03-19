import os
import numpy as np
import librosa
import openl3
import pandas as pd
import torch
from src.training import get_song_features
from src.lab import read_chord_file


def extract_features(audio_path, lab_path, sr=44100):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    # Separate harmonic and percussive components
    y_harmonic, _ = librosa.effects.hpss(y)

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Chroma feature extraction (beat-synchronous)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.mean).T

    # Extract OpenL3 embeddings
    embeddings, ts = openl3.get_audio_embedding(y, sr, embedding_size=512, content_type='music')

    # Align embeddings to beats
    feature_vectors = []
    for i in range(len(beat_times)-1):
        # Get normalized chroma for this beat
        normalized_chroma = chroma_sync[i] / np.linalg.norm(chroma_sync[i])
        
        # Find embeddings that fall within this beat interval
        idx = np.where((ts >= beat_times[i]) & (ts < beat_times[i+1]))[0]
        if len(idx) > 0:
            emb_mean = np.mean(embeddings[idx], axis=0)
        else:
            emb_mean = np.zeros(embeddings.shape[1])
        
        # Create combined feature vector
        combined_feature = np.concatenate((normalized_chroma, emb_mean))
        feature_vectors.append(combined_feature)

    # Convert to numpy array
    feature_vectors = np.array(feature_vectors)

    # Use the read_chord_file function from src.lab
    labels_df = read_chord_file(lab_path)
    
    # Convert the start and end columns to float if they aren't already
    if not pd.api.types.is_numeric_dtype(labels_df['start']):
        labels_df['start'] = pd.to_numeric(labels_df['start'], errors='coerce')
    if not pd.api.types.is_numeric_dtype(labels_df['end']):
        labels_df['end'] = pd.to_numeric(labels_df['end'], errors='coerce')

    # Assign chord labels to beats with error handling
    chord_labels = []
    default_chord = labels_df['chord'].iloc[0]  # Use first chord as default
    
    for beat_time in beat_times[:-1]:  # Use one fewer beat time to match feature_vectors length
        matching_rows = labels_df[(labels_df['start'] <= beat_time) & (labels_df['end'] > beat_time)]
        if not matching_rows.empty:
            chord_labels.append(matching_rows.iloc[0]['chord'])
        else:
            # Find the closest chord based on time if no direct match
            closest_idx = (labels_df['start'] - beat_time).abs().argmin()
            chord_labels.append(labels_df.iloc[closest_idx]['chord'])
    
    return feature_vectors, chord_labels


def create_dataset(audio_dir, labels_dir, output_path):
    all_features = []
    all_labels = []

    for file in os.listdir(audio_dir):
        if file.endswith('.mp3'):
            audio_path = os.path.join(audio_dir, file)
            lab_file = file.replace('.mp3', '.lab')
            lab_path = os.path.join(labels_dir, lab_file)

            features, labels = extract_features(audio_path, lab_path)
            all_features.append(features)
            all_labels.extend(labels)

    # Combine into arrays
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)

    # Create a mapping from chord names to indices
    unique_chords = np.unique(all_labels)
    chord_to_idx = {chord: idx for idx, chord in enumerate(unique_chords)}
    
    # Convert string labels to integer indices
    label_indices = np.array([chord_to_idx[chord] for chord in all_labels])
    
    # Save dataset with both string labels and numeric indices
    np.savez(output_path, 
             features=all_features, 
             labels=all_labels,
             label_indices=label_indices,
             chord_mapping=np.array(list(chord_to_idx.items())))
    
    # Create PyTorch dataset with numeric indices
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(all_features, dtype=torch.float32),
        torch.tensor(label_indices, dtype=torch.long)
    )
    
    return dataset, chord_to_idx


# Usage example:
audio_dir = 'lab_and_audio_files_small'
labels_dir = 'lab_and_audio_files_small'
output_path = 'chord_dataset_small.npz'

create_dataset(audio_dir, labels_dir, output_path)

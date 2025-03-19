import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import openl3
from src.utils import load_embeddings, save_embeddings
from src.consts import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_NORM,
    DEFAULT_TUNING,
    COL_NAMES_NOTES,
)

# Global cache for the OpenL3 model
MODEL_OPENL3 = None


def get_openl3_model():
    """
    Loads the OpenL3 model once and reuses it.

    Returns:
        OpenL3 model instance.
    """
    global MODEL_OPENL3
    if MODEL_OPENL3 is None:
        print("🔄 Loading OpenL3 model...")
        MODEL_OPENL3 = openl3.models.load_audio_embedding_model(
            content_type="music", input_repr="mel256", embedding_size=512
        )
        print("✅ OpenL3 model loaded and cached!")
    return MODEL_OPENL3


get_openl3_model()


def get_chromagram_from_file(
    filename: str, remove_percussive: bool = False
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Extracts chromagram features from an audio file.

    Args:
        filename (str): Path to the audio file.

    Returns:
        tuple[np.ndarray, float]: Chromagram and frames per second.
    """
    x, Fs = librosa.load(filename)

    _, beat_frames = librosa.beat.beat_track(y=x, sr=Fs)
    beat_times = librosa.frames_to_time(beat_frames, sr=Fs)

    if remove_percussive:
        x, _ = librosa.effects.hpss(x)

    chromagram = librosa.feature.chroma_stft(
        y=x,
        sr=Fs,
        tuning=DEFAULT_TUNING,
        norm=DEFAULT_NORM,
        hop_length=DEFAULT_HOP_LENGTH,
        n_fft=DEFAULT_N_FFT,
    )

    # Compute frames per second (ensuring it's in seconds, NOT milliseconds)
    frames_per_sec = chromagram.shape[1] / (len(x) / Fs)

    return chromagram, frames_per_sec, beat_times


def convert_chromagram_to_dataframe(
    chromagram: np.ndarray,
    frames_per_sec: float,
    beat_times: np.ndarray,
    pool_to_beats: bool = False,
) -> pd.DataFrame:
    """
    Converts chromagram data into a DataFrame with time-aligned frames.
    If `pool_to_beats` is True, it aggregates chroma features and predicted chords over beats.

    Args:
        chromagram (np.ndarray): Chromagram array.
        frames_per_sec (float): Number of frames per second.
        beat_times (np.ndarray): Array of detected beat times (seconds).
        pool_to_beats (bool): Whether to aggregate chroma features and predictions over beats.

    Returns:
        pd.DataFrame: DataFrame containing chroma features with time windows.
    """
    frame_duration_sec = 1.0 / frames_per_sec  # Compute frame duration in SECONDS

    chromagram_df = pd.DataFrame(chromagram.T, columns=COL_NAMES_NOTES)

    # Compute 'start' and 'end' times in SECONDS
    chromagram_df["start"] = chromagram_df.index * frame_duration_sec
    chromagram_df["end"] = chromagram_df["start"] + frame_duration_sec

    # Initialize predicted column (it should already exist after HMM predictions)
    if "predicted" not in chromagram_df:
        chromagram_df["predicted"] = None  # Placeholder for later predictions

    if pool_to_beats:
        # Backup original predictions
        chromagram_df["predicted_original"] = chromagram_df["predicted"].copy()

        # Assign each frame to the closest beat
        chromagram_df["beat_cluster"] = np.digitize(chromagram_df["start"], beat_times)

        # Aggregate chroma features within each beat cluster
        beat_pooled_chromagram = (
            chromagram_df.groupby("beat_cluster")[COL_NAMES_NOTES].mean().reset_index()
        )

        # Override 'predicted' column with the most common prediction per beat
        mode_cluster = chromagram_df.groupby("beat_cluster")["predicted"].agg(
            lambda x: x.value_counts().index[0]
        )
        chromagram_df["predicted"] = mode_cluster.loc[
            chromagram_df["beat_cluster"]
        ].values

        # Replace start & end times with beat-aligned ones
        beat_pooled_chromagram["start"] = beat_times[
            beat_pooled_chromagram["beat_cluster"] - 1
        ]
        beat_pooled_chromagram["end"] = np.append(
            beat_pooled_chromagram["start"][1:].values, chromagram_df["end"].iloc[-1]
        )

        return chromagram_df

    return chromagram_df


def detect_beats(audio_path: str, sr=16000):
    """
    Detects beat times in an audio file.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Sample rate.

    Returns:
        np.ndarray: Beat times in seconds.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    return beat_times


def extract_openl3_embeddings(wav_file: str, embedding_size=512):
    """
    Extracts OpenL3 embeddings with a customizable window size and hop size.

    Args:
        wav_file (str): Path to the WAV file.
        window_size (float): Window size in seconds (default 0.5s).
        hop_size (float): Hop size in seconds (default 0.1s).
        sr (int): Sample rate (default 16kHz).
        embedding_size (int): Size of embeddings (512 or 6144).

    Returns:
        np.ndarray: OpenL3 embeddings (n_frames, embedding_size).
    """
    embeddings = load_embeddings(f"{wav_file}_openl3.joblib")
    if embeddings is None:
        audio_file_path = f"{wav_file}.wav"
        # Load audio
        y, Fs = librosa.load(audio_file_path, mono=True)

        # Extract OpenL3 embeddings
        embeddings, _ = openl3.get_audio_embedding(
            y,
            Fs,
            content_type="music",
            embedding_size=embedding_size,
            hop_size=DEFAULT_HOP_LENGTH / Fs,
            model=get_openl3_model(),
        )
        save_embeddings(embeddings, f"{wav_file}_openl3.joblib")

    return embeddings


def synchronize_features(
    chromagram: pd.DataFrame, embeddings: np.ndarray
) -> pd.DataFrame:
    """
    Synchronizes chroma and openl3 embeddings by truncating to the shortest frame length
    while preserving chromagram column names.

    Args:
        chromagram (pd.DataFrame): Chroma feature DataFrame (n_frames, 12).
        embeddings (np.ndarray): openl3 embeddings (n_frames, 256).

    Returns:
        pd.DataFrame: DataFrame containing combined features with column names preserved.
    """
    # Ensure both features have the same time dimension
    min_frames = min(len(chromagram), embeddings.shape[0])

    # Truncate to the same number of frames
    chroma_resized = chromagram.iloc[:min_frames].copy()  # Retains DataFrame structure
    embeddings_resized = embeddings[:min_frames]  # NumPy array

    # Convert embeddings into a DataFrame with column names
    embedding_columns = [f"openL3_{i}" for i in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(embeddings_resized, columns=embedding_columns)

    # Concatenate chroma DataFrame with embeddings DataFrame
    combined_df = pd.concat(
        [chroma_resized.reset_index(drop=True), embeddings_df], axis=1
    )

    return combined_df

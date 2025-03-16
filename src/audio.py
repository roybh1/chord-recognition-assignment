import librosa
import numpy as np
import pandas as pd
from src.consts import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_NORM,
    DEFAULT_TUNING,
    COL_NAMES_NOTES,
)

# pitch class profile


def get_chromagram_from_file(filename: str) -> tuple[np.ndarray, float]:
    """
    Extracts chromagram features from an audio file.

    Args:
        filename (str): Path to the audio file.

    Returns:
        tuple[np.ndarray, float]: Chromagram and frames per second.
    """
    x, Fs = librosa.load(filename)
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

    return chromagram, frames_per_sec


def convert_chromagram_to_dataframe(
    chromagram: np.ndarray, frames_per_sec: float
) -> pd.DataFrame:
    """
    Converts chromagram data into a DataFrame with time-aligned frames.

    Args:
        chromagram (np.ndarray): Chromagram array.
        frames_per_sec (float): Number of frames per second.

    Returns:
        pd.DataFrame: DataFrame containing chroma features and corresponding time windows.
    """
    frame_duration_sec = 1.0 / frames_per_sec  # Compute frame duration in SECONDS

    chromagram_df = pd.DataFrame(chromagram.T, columns=COL_NAMES_NOTES)

    # Ensure 'start' and 'end' times are in SECONDS, NOT MILLISECONDS
    chromagram_df["start"] = (
        chromagram_df.index * frame_duration_sec
    )  # Start time of each frame
    chromagram_df["end"] = (
        chromagram_df["start"] + frame_duration_sec
    )  # End time of each frame

    return chromagram_df

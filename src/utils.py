import numpy as np
import pandas as pd
import joblib  # Faster than pickle
import os


def annotate_chord_sequence(
    chromagram: pd.DataFrame, chords_annotation: pd.DataFrame
) -> pd.DataFrame:
    """
    Assigns chord labels to each time frame in the chromagram using chord annotations.

    Args:
        chromagram (pd.DataFrame): Dataframe with chroma features and time frames.
        chords_annotation (pd.DataFrame): Dataframe with 'start' times and corresponding 'chord' labels.

    Returns:
        pd.DataFrame: Chromagram with an added 'chord' column, maintaining the correct order.
    """

    # Ensure chords_annotation is sorted by time (important for correct ordering)
    chords_annotation = chords_annotation.sort_values(by="start").reset_index(drop=True)

    # Use searchsorted to find the correct chord index for each time frame in chromagram
    chord_indices = (
        np.searchsorted(
            chords_annotation["start"].values, chromagram["start"].values, side="right"
        )
        - 1
    )

    # Ensure indices stay within bounds
    chord_indices = np.clip(chord_indices, 0, len(chords_annotation) - 1)

    # Assign chords based on these indices
    chromagram["chord"] = chords_annotation.iloc[chord_indices]["chord"].values

    # Assign <START> and <END> at the correct positions
    chromagram.loc[chromagram.index[0], "chord"] = "<START>"
    chromagram.loc[chromagram.index[-1], "chord"] = "<END>"

    return chromagram


def save_embeddings(embeddings: np.ndarray, filename: str):
    """
    Saves OpenL3 embeddings to a file.

    Args:
        embeddings (np.ndarray): OpenL3 embeddings.
        filename (str): Path to save the file.
    """
    joblib.dump(embeddings, filename)
    print(f"✅ Embeddings saved to {filename}")


def load_embeddings(filename: str):
    """
    Loads OpenL3 embeddings from a file.

    Args:
        filename (str): Path to the file.

    Returns:
        np.ndarray: Loaded OpenL3 embeddings.
    """
    if os.path.exists(filename):
        print(f"✅ Loading cached embeddings from {filename}")
        return joblib.load(filename)
    return None  # If the file doesn’t exist, return None

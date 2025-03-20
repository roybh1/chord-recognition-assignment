import pandas as pd
from hmmlearn import hmm
from src.lab import read_chord_file
from src.audio import (
    convert_chromagram_to_dataframe,
    get_chromagram_from_file,
    extract_openl3_embeddings,
    synchronize_features,
)
from src.utils import annotate_chord_sequence, save_embeddings
from src.consts import COL_NAMES_NOTES

from src.hmm import (
    build_transition_probability_matrix,
    compute_initial_state_probabilities,
    filter_and_normalize_initial_probabilities,
    build_gaussian_hmm,
    extract_mean_and_covariance,
    get_hmm_predictions,
    build_pomegranate_hmm,
)

DEST_FOLDER = "lab_and_audio_files"
AUDIO_SUFFIX = ".mp3"


def get_song_chromagram(
    song_name: str, remove_percussive: bool = False, pool_to_beats: bool = False
) -> pd.DataFrame:
    song_path = f"{DEST_FOLDER}/{song_name}"
    lab_file_path = f"{song_path}.lab"
    audio_file_path = f"{song_path}{AUDIO_SUFFIX}"

    chords_annotation = read_chord_file(lab_file_path)
    chromagram = convert_chromagram_to_dataframe(
        *get_chromagram_from_file(audio_file_path, remove_percussive=remove_percussive),
        pool_to_beats=pool_to_beats,
    )
    annotate_chord_sequence(chromagram, chords_annotation)
    return chromagram


def get_song_features(
    song_name: str,
    remove_percussive: bool = False,
    pool_to_beats: bool = False,
    add_openl3: bool = False,
) -> pd.DataFrame:
    chromagram = get_song_chromagram(
        song_name, remove_percussive=remove_percussive, pool_to_beats=pool_to_beats
    )
    if not add_openl3:
        return chromagram

    song_path = f"{DEST_FOLDER}/{song_name}"
    openl3_embeddings = extract_openl3_embeddings(song_path)
    merged = synchronize_features(chromagram, openl3_embeddings)
    merged.fillna(0, inplace=True)
    return merged


def run_training_pipeline(
    songs_names: list[str], remove_percussive: bool = False, add_openl3: bool = False
) -> tuple[hmm.GaussianHMM, dict]:
    """
    train hmm over labeled audio files
    """
    chromagram = pd.concat(
        [
            get_song_features(
                song_name, remove_percussive=remove_percussive, add_openl3=add_openl3
            )
            for song_name in songs_names
        ],
        ignore_index=True,
    )

    # Select only numerical features for training
    feature_columns = [col for col in chromagram.columns if col not in ["predicted"]]

    # Ensure all training and prediction use the same features
    feature_matrix = chromagram[feature_columns]
    print(
        f"Training Feature Matrix Shape: {feature_matrix.shape}"
    )  # Expect (n_frames, n_features)

    mu_array, states_cov_matrices = extract_mean_and_covariance(feature_matrix)
    transition_matrix = build_transition_probability_matrix(feature_matrix)
    initial_state_probs = filter_and_normalize_initial_probabilities(
        compute_initial_state_probabilities(), transition_matrix
    )
    chord_numbers = range(len(mu_array.index.values))
    chords = mu_array.index.values
    ix_2_chord = {ix_: chord_str for ix_, chord_str in zip(chord_numbers, chords)}

    return (
        build_pomegranate_hmm(
            initial_state_probs, transition_matrix, mu_array, states_cov_matrices
        ),
        ix_2_chord,
    )


def get_predictions(
    h_markov_model: hmm.GaussianHMM,
    chord_dict: dict,
    song_name: str,
    remove_percussive: bool = False,
    pool_to_beats: bool = False,
    add_openl3: bool = False,
) -> pd.DataFrame:
    """
    Generates chord predictions using the trained HMM.

    Args:
        h_markov_model (hmm.GaussianHMM): Trained HMM model.
        chord_dict (dict): Mapping of state indices to chord labels.
        song_name (str): Name of the song to process.
        remove_percussive (bool): Whether to remove percussive elements.
        pool_to_beats (bool): Whether to pool features over beats.
        add_openl3 (bool): Whether to include OpenL3 features.

    Returns:
        pd.DataFrame: DataFrame with start, end, true chord, and predicted chord.
    """
    # Extract all features
    chromagram = get_song_features(
        song_name,
        remove_percussive=remove_percussive,
        pool_to_beats=pool_to_beats,
        add_openl3=add_openl3,
    )

    # 🔹 Select only the feature columns (exclude non-feature columns)
    feature_columns = [
        col
        for col in chromagram.columns
        if col not in ["start", "end", "chord", "predicted"]
    ]

    # 🔹 Use all features (Chroma + OpenL3)
    feature_matrix = chromagram[feature_columns]

    print(
        f"Start Probabilities Shape: {h_markov_model.startprob_.shape}"
    )  # Expect (n_states,)
    print(
        f"Transition Matrix Shape: {h_markov_model.transmat_.shape}"
    )  # Expect (n_states, n_states)
    print(
        f"Feature Matrix Shape (X): {feature_matrix.shape}"
    )  # Expect (n_samples, n_features)
    print(
        f"Mean Vectors Shape (mu_array): {h_markov_model.means_.shape}"
    )  # Expect (n_states, n_features)

    # 🔹 Predict chord indices
    chord_ix_predictions = h_markov_model.predict(feature_matrix)

    # 🔹 Convert indices to chord names
    chord_str_predictions = get_hmm_predictions(chord_ix_predictions, chord_dict)

    # 🔹 Store predictions
    chromagram["predicted"] = chord_str_predictions

    # Return only the relevant columns
    return chromagram[["start", "end", "chord", "predicted"]]

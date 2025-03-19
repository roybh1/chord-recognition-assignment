import pandas as pd
from hmmlearn import hmm
from src.lab import read_chord_file
from src.audio import convert_chromagram_to_dataframe, get_chromagram_from_file
from src.utils import annotate_chord_sequence
from src.consts import COL_NAMES_NOTES

from src.hmm import (
    build_transition_probability_matrix,
    compute_initial_state_probabilities,
    filter_and_normalize_initial_probabilities,
    build_gaussian_hmm,
    extract_mean_and_covariance,
    get_hmm_predictions,
)

DEST_FOLDER = "lab_and_audio_files"
AUDIO_SUFFIX = ".mp3"


def get_song_chromagram(song_name: str) -> pd.DataFrame:
    """
    get the chromagram of a song
    the chromagram is a matrix of size 12xT, where T is the number of frames in the song

    """
    song_path = f"{DEST_FOLDER}/{song_name}"
    lab_file_path = f"{song_path}.lab"
    audio_file_path = f"{song_path}{AUDIO_SUFFIX}"

    chords_annotation = read_chord_file(lab_file_path)
    chromagram = convert_chromagram_to_dataframe(
        *get_chromagram_from_file(audio_file_path)
    )
    annotate_chord_sequence(chromagram, chords_annotation)
    return chromagram


def run_training_pipeline(songs_names: list[str]):
    """
    train hmm over labeled audio files
    """
    chromagram = pd.concat(
        [get_song_chromagram(song_name) for song_name in songs_names], ignore_index=True
    )

    mu_array, states_cov_matrices = extract_mean_and_covariance(chromagram)
    transition_matrix = build_transition_probability_matrix(chromagram)
    initial_state_probs = filter_and_normalize_initial_probabilities(
        compute_initial_state_probabilities(), transition_matrix
    )
    chord_numbers = range(len(mu_array.index.values))
    chords = mu_array.index.values
    ix_2_chord = {ix_: chord_str for ix_, chord_str in zip(chord_numbers, chords)}

    return (
        build_gaussian_hmm(
            initial_state_probs, transition_matrix, mu_array, states_cov_matrices
        ),
        ix_2_chord,
    )


def get_predictions(
    h_markov_model: hmm.GaussianHMM, chord_dict: dict, song_name: str
) -> pd.DataFrame:
    chromagram = get_song_chromagram(song_name)

    chord_ix_predictions = h_markov_model.predict(chromagram[COL_NAMES_NOTES])
    chord_str_predictions = get_hmm_predictions(chord_ix_predictions, chord_dict)
    chromagram["predicted"] = chord_str_predictions

    return chromagram

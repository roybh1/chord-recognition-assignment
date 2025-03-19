import pandas as pd
from hmmlearn import hmm
from src.lab import read_chord_file
from src.audio import (
    convert_chromagram_to_dataframe,
    get_chromagram_from_file,
    extract_openl3_embeddings,
    synchronize_features,
)
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
    add_vggish: bool = False,
) -> pd.DataFrame:
    chromagram = get_song_chromagram(
        song_name, remove_percussive=remove_percussive, pool_to_beats=pool_to_beats
    )
    if not add_vggish:
        return chromagram

    song_path = f"{DEST_FOLDER}/{song_name}"
    audio_file_path = f"{song_path}.mp3"
    openl3_embeddings = extract_openl3_embeddings(audio_file_path)
    return synchronize_features(chromagram, openl3_embeddings)


def run_training_pipeline(
    songs_names: list[str], remove_percussive: bool = False, add_vggish: bool = False
) -> tuple[hmm.GaussianHMM, dict]:
    """
    train hmm over labeled audio files
    """
    chromagram = pd.concat(
        [
            get_song_features(
                song_name, remove_percussive=remove_percussive, add_vggish=add_vggish
            )
            for song_name in songs_names
        ],
        ignore_index=True,
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
    h_markov_model: hmm.GaussianHMM,
    chord_dict: dict,
    song_name: str,
    remove_percussive: bool = False,
    pool_to_beats: bool = False,
    add_vggish: bool = False,
) -> pd.DataFrame:
    chromagram = get_song_features(
        song_name,
        remove_percussive=remove_percussive,
        pool_to_beats=pool_to_beats,
        add_vggish=add_vggish,
    )

    chord_ix_predictions = h_markov_model.predict(chromagram[COL_NAMES_NOTES])
    chord_str_predictions = get_hmm_predictions(chord_ix_predictions, chord_dict)
    chromagram["predicted"] = chord_str_predictions

    # return only the relevant columns: start, end, chord, predicted
    return chromagram[["start", "end", "chord", "predicted"]]

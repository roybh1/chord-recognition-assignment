import pandas as pd
import numpy as np
from src.lab import read_chord_file
from hmmlearn import hmm
import os
from src.consts import FULL_CHORDS, DATA_DIR, COL_NAMES_NOTES


def compute_transition_matrix(chord_transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the transition probability matrix for chord changes.

    Args:
        chord_transitions (pd.DataFrame): A dataframe containing transitions
                                          from one chord to the next.

    Returns:
        pd.DataFrame: A transition probability matrix where each entry (i, j)
                      represents the probability of transitioning from chord 'i' to 'j'.
    """
    # Count occurrences of each chord transition
    transition_counts = (
        chord_transitions.groupby(["current_chord", "next_chord"])
        .size()
        .reset_index(name="count")
    )

    # Compute probabilities correctly using .transform() instead of .apply()
    transition_counts["transition_prob"] = transition_counts[
        "count"
    ] / transition_counts.groupby("current_chord")["count"].transform("sum")

    # Convert to pivot table (transition matrix)
    transition_matrix = transition_counts.pivot(
        index="current_chord", columns="next_chord", values="transition_prob"
    )

    return transition_matrix.fillna(0)  # Fill missing values with 0


def build_transition_probability_matrix(chromagram: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a chord transition probability matrix from a chromagram dataframe.

    Args:
        chromagram (pd.DataFrame): A dataframe containing chords for each time frame.

    Returns:
        pd.DataFrame: The transition probability matrix including <START> and <END> states.
    """
    # Create a dataframe of chord transitions (excluding the last frame)
    chord_transitions = pd.DataFrame(
        {
            "current_chord": chromagram["chord"].iloc[:-1].values,  # Current chord
            "next_chord": chromagram["chord"].iloc[1:].values,  # Next chord
        }
    )

    # Compute the transition matrix
    transition_matrix = compute_transition_matrix(chord_transitions)

    # Add <END> state (no outgoing transitions)
    transition_matrix.loc["<END>"] = 0.0

    # Add <START> state (no incoming transitions)
    transition_matrix["<START>"] = 0.0

    # Ensure <END> always transitions to itself with probability 1
    transition_matrix.loc["<END>", "<END>"] = 1.0

    return transition_matrix


def compute_initial_state_probabilities(annotations_path: str = DATA_DIR) -> pd.Series:
    """
    Computes the initial state probability distribution from chord annotations.

    Args:
        ignore_silence (bool): If True, ignores silence when determining the first chord of a song.
        annotations_path (str): Path to the folder containing chord annotation (.lab) files.

    Returns:
        pd.Series: A probability distribution for starting chords, including <START> and <END> states.
    """
    first_chords: list[str] = []  # Stores the first chord of each song

    # Iterate through all chord annotation files in the specified directory
    for filename in os.listdir(annotations_path):
        if filename.endswith(".lab"):  # Process only .lab files
            chords_data: pd.DataFrame = read_chord_file(
                f"{annotations_path}/{filename}"
            )
            first_chords.append(
                chords_data["chord"].values[0]
            )  # Extract the first chord

    if not first_chords:  # Handle the case where no chords are found
        raise ValueError(
            "No valid first chords found in the dataset. Check your annotations."
        )

    # Count occurrences of each first chord
    unique_chords, chord_counts = np.unique(first_chords, return_counts=True)

    # Convert counts to probabilities by normalizing
    initial_chord_probs: pd.Series = pd.Series(
        chord_counts / chord_counts.sum(), index=unique_chords
    )

    # Ensure the probability distribution includes special states
    initial_chord_probs = pd.concat(
        [initial_chord_probs, pd.Series([0.0], index=["<START>"])]
    )  # No probability for <START>
    initial_chord_probs = pd.concat(
        [initial_chord_probs, pd.Series([1.0], index=["<END>"])]
    )  # <END> always transitions to itself

    # Ensure all known chords are represented in the probability distribution, even if they haven't appeared
    for chord in FULL_CHORDS:
        if chord not in initial_chord_probs.index:
            initial_chord_probs = pd.concat(
                [initial_chord_probs, pd.Series([0.0], index=[chord])]
            )

    return initial_chord_probs


def filter_and_normalize_initial_probabilities(
    initial_probs: pd.Series, transition_matrix: pd.DataFrame
) -> pd.Series:
    """
    Filters and normalizes the initial state probability distribution
    to only include chords present in the transition matrix.

    Args:
        initial_probs (pd.Series): The computed initial state probabilities.
        transition_matrix (pd.DataFrame): The transition probability matrix.

    Returns:
        pd.Series: A filtered and renormalized initial probability distribution.
    """
    # Ensure the provided probability distribution includes all transition matrix columns
    filtered_probs: pd.Series = initial_probs.reindex(
        transition_matrix.columns, fill_value=0.0
    )

    # Normalize the probabilities so they sum to 1
    total_prob: float = filtered_probs.sum()
    normalized_probs: pd.Series = (
        filtered_probs / total_prob if total_prob > 0 else filtered_probs
    )

    return normalized_probs


def build_gaussian_hmm(
    initial_state_prob: np.ndarray,
    transition_matrix: pd.DataFrame,
    mu_array: np.ndarray,
    states_cov_matrices: np.ndarray,
    epsilon: float = 1e-6,
) -> hmm.GaussianHMM:
    """
    Builds a Gaussian HMM with corrected covariance matrices.

    Args:
        initial_state_prob (np.ndarray): Initial state probability vector.
        transition_matrix (pd.DataFrame): Transition probability matrix.
        mu_array (np.ndarray): Mean vectors for each state.
        states_cov_matrices (np.ndarray): Covariance matrices for each state.
        epsilon (float): Small regularization factor for covariance matrices.

    Returns:
        hmm.GaussianHMM: The trained Gaussian Hidden Markov Model.
    """
    # Ensure all covariance matrices are valid
    for i in range(states_cov_matrices.shape[0]):
        cov_matrix = states_cov_matrices[i]
        # Ensure matrix is symmetric
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        # Regularize if non-positive-definite
        try:
            np.linalg.cholesky(cov_matrix)  # Check if positive definite
        except np.linalg.LinAlgError:
            cov_matrix += epsilon * np.eye(cov_matrix.shape[0])  # Regularization
        states_cov_matrices[i] = cov_matrix

    # Continuous emission model
    h_markov_model = hmm.GaussianHMM(
        n_components=transition_matrix.shape[0], covariance_type="full"
    )

    # Assign HMM parameters
    h_markov_model.startprob_ = initial_state_prob
    h_markov_model.transmat_ = transition_matrix.values
    h_markov_model.means_ = mu_array
    h_markov_model.covars_ = states_cov_matrices

    return h_markov_model


def compute_mean_note_vector(chromagram: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the mean chroma vector for each chord.

    Args:
        chromagram (pd.DataFrame): DataFrame containing chroma features and chord labels.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a chord and columns contain the mean chroma values.
    """
    return chromagram.groupby("chord")[COL_NAMES_NOTES].mean()


def compute_covariance_matrices(chromagram: pd.DataFrame) -> np.ndarray:
    """
    Computes the covariance matrix for each chord's chroma features.

    Args:
        chromagram (pd.DataFrame): DataFrame containing chroma features and chord labels.

    Returns:
        np.ndarray: An array of covariance matrices, one for each unique chord.
    """
    states_cov_matrices = []
    for _, group in chromagram.groupby("chord"):  # alphabetic order
        states_cov_matrices.append(group[COL_NAMES_NOTES].cov().values)
    states_cov_matrices = np.array(states_cov_matrices)
    states_cov_matrices[0:2] = 0 # turn cov matrices of <START> and <END> to from nan to 0
    return states_cov_matrices


def extract_mean_and_covariance(
    chromagram: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Extracts mean chroma vectors and covariance matrices for each chord.

    Args:
        chromagram (pd.DataFrame): DataFrame containing chroma features and chord labels.

    Returns:
        tuple: A tuple containing:
            - A DataFrame of mean chroma vectors per chord.
            - An array of covariance matrices per chord.
    """
    mean_vectors = compute_mean_note_vector(chromagram)
    covariance_matrices = compute_covariance_matrices(chromagram)
    return mean_vectors, covariance_matrices


def get_hmm_predictions(chord_ix_predictions, ix_2_chord):
    return np.array([ix_2_chord[chord_ix] for chord_ix in chord_ix_predictions])

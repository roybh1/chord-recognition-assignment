import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import seaborn as sns




def evaluate_all_songs(results: dict) -> dict:
    """
    Computes accuracy and weighted F1-score for each song and overall.

    Args:
        results (dict): A dictionary where keys are song names and values are chromagram DataFrames
                        with 'chord' (true labels) and 'predicted' columns.

    Returns:
        dict: A dictionary with individual song metrics and overall metrics.
    """
    all_y_true = []
    all_y_pred = []
    song_metrics = {}

    print("\n🔹 **Chord Prediction Evaluation Across All Songs** 🔹\n")

    for song_name, chromagram in results.items():
        # Remove <START> and <END> from evaluation
        filtered_chromagram = chromagram[
            (chromagram["chord"] != "<START>") & (chromagram["chord"] != "<END>")
        ]

        # Extract true and predicted labels
        y_true = filtered_chromagram["chord"]
        y_pred = filtered_chromagram["predicted"]

        # Compute accuracy and F1-score
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")

        # Store per-song metrics
        song_metrics[song_name] = {"accuracy": accuracy, "f1_score": f1}

        # Append to global lists for overall calculation
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        print(f"🎵 {song_name}: Accuracy = {accuracy:.4f}, F1-score = {f1:.4f}")

    # Compute overall accuracy and F1-score across all songs
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, average="weighted")
    overall_precision = precision_score(all_y_true, all_y_pred, average="weighted")
    overall_recall = recall_score(all_y_true, all_y_pred, average="weighted")


    # Store overall metrics
    song_metrics["overall"] = {
        "accuracy": overall_accuracy,
        "f1_score": overall_f1,
        "precision": overall_precision,
        "recall": overall_recall,
    }

    print("\n🔹 **Overall Performance Across All Songs** 🔹")
    print(f"🎶 Overall Accuracy = {overall_accuracy:.4f}")
    print(f"🎶 Overall F1-score = {overall_f1:.4f}")
    print(f"🎶 Overall Precision = {overall_precision:.4f}")
    print(f"🎶 Overall Recall = {overall_recall:.4f}\n")

    # print(f"🎶 Average difference of chord prediction counts = {avg_differente_of_chord_prediction_counts(results):.4f}\n")

    return song_metrics

def avg_differente_of_chord_prediction_counts(results: dict) -> float:
    """
    Computes the average difference in chord prediction counts between true and predicted chords.

    Args:
        results (dict): A dictionary where keys are song names and values are chromagram DataFrames
    
    Returns:
        float: The average difference in chord prediction counts between true and predicted chords.
    """
    true_counts = []
    predicted_counts = []

    for song_name, chromagram in results.items():
        # Remove <START> and <END>
        filtered_chromagram = chromagram[
            (chromagram["chord"] != "<START>") & (chromagram["chord"] != "<END>")
        ]

        true_counts.append(filtered_chromagram["chord"].value_counts())
        predicted_counts.append(filtered_chromagram["predicted"].value_counts())

    true_counts = pd.concat(true_counts)
    predicted_counts = pd.concat(predicted_counts)

    return np.mean(np.abs(true_counts - predicted_counts))
    

def plot_grouped_chord_timeline(chromagram: pd.DataFrame, song_name: str):
    """
    Plots a vertical timeline comparing true vs. predicted chords over time,
    grouping consecutive identical chords together and displaying transition times.
    Also computes and displays Accuracy and F1-score.

    Args:
        chromagram (pd.DataFrame): DataFrame with 'start', 'chord' (true), and 'predicted' columns.
        song_name (str): The name of the song being evaluated.
    """
    # Filter out <START> and <END>
    filtered_chromagram = chromagram[
        (chromagram["chord"] != "<START>") & (chromagram["chord"] != "<END>")
    ]

    # Compute accuracy and F1-score
    y_true = filtered_chromagram["chord"]
    y_pred = filtered_chromagram["predicted"]

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    # Print metrics
    print(f"🔹 **Accuracy**: {accuracy:.4f}")
    print(f"🔹 **F1-score**: {f1:.4f}")

    # Group consecutive identical chords
    grouped_times = []
    grouped_true_chords = []
    grouped_pred_chords = []

    prev_true = None
    prev_pred = None

    for _, row in filtered_chromagram.iterrows():
        true_chord = row["chord"]
        pred_chord = row["predicted"]
        start_time = row["start"]

        if true_chord != prev_true or pred_chord != prev_pred:
            grouped_times.append(start_time)
            grouped_true_chords.append(true_chord)
            grouped_pred_chords.append(pred_chord)

        prev_true, prev_pred = true_chord, pred_chord

    fig, ax = plt.subplots(
        figsize=(5, len(grouped_times) // 3)
    )  # Adjust height dynamically

    for i, (t, true_chord, pred_chord) in enumerate(
        zip(grouped_times, grouped_true_chords, grouped_pred_chords)
    ):
        color = "green" if true_chord == pred_chord else "red"

        # Show transition time
        ax.text(-1, -i, f"{t:.2f}s", va="center", ha="right", fontsize=9, color="gray")

        # True chord on the left
        ax.text(
            -0.5, -i, true_chord, va="center", ha="right", fontsize=10, color="black"
        )

        # Connecting line
        ax.plot([0, 1], [-i, -i], color=color, linewidth=2)

        # Predicted chord on the right
        ax.text(1.5, -i, pred_chord, va="center", ha="left", fontsize=10, color=color)

    # Formatting
    ax.set_xlim(-2, 2)
    ax.set_ylim(-len(grouped_times), 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"'{song_name}': Grouped Chord Prediction Timeline\n"
        f"(Accuracy: {accuracy:.2f}, F1-score: {f1:.2f})\n"
        "Green = Correct, Red = Incorrect\nTimes Indicate Chord Transitions"
    )

    plt.show()


def plot_chord_transition_counts(results: dict):
    """
    Plots a bar chart comparing the true number of chord transitions vs.
    the predicted number of chord transitions for each song.

    Args:
        results (dict): A dictionary where keys are song names and values are chromagram DataFrames
                        with 'chord' (true labels) and 'predicted' columns.
    """
    song_names = []
    true_transitions = []
    predicted_transitions = []

    for song_name, chromagram in results.items():
        # Remove <START> and <END>
        filtered_chromagram = chromagram[
            (chromagram["chord"] != "<START>") & (chromagram["chord"] != "<END>")
        ]

        # Count true chord transitions (number of times the chord changes)
        true_chord_changes = (
            filtered_chromagram["chord"] != filtered_chromagram["chord"].shift()
        ).sum()

        # Count predicted chord transitions
        predicted_chord_changes = (
            filtered_chromagram["predicted"] != filtered_chromagram["predicted"].shift()
        ).sum()

        # Store data
        song_names.append(song_name)
        true_transitions.append(true_chord_changes)
        predicted_transitions.append(predicted_chord_changes)

    # Plot bar chart
    x = np.arange(len(song_names))  # Song positions
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width / 2,
        true_transitions,
        width,
        label="True Chord Transitions",
        color="blue",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        predicted_transitions,
        width,
        label="Predicted Chord Transitions",
        color="orange",
        alpha=0.7,
    )

    # Formatting
    ax.set_xlabel("Songs")
    ax.set_ylabel("Number of Chord Transitions")
    ax.set_title("True vs. Predicted Chord Transition Counts per Song")
    ax.set_xticks(x)
    ax.set_xticklabels(song_names, rotation=45, ha="right")
    ax.legend()

    plt.show()

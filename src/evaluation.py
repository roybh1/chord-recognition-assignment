import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
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

    # Store overall metrics
    song_metrics["overall"] = {"accuracy": overall_accuracy, "f1_score": overall_f1}

    print("\n🔹 **Overall Performance Across All Songs** 🔹")
    print(f"🎶 Overall Accuracy = {overall_accuracy:.4f}")
    print(f"🎶 Overall F1-score = {overall_f1:.4f}\n")

    return song_metrics


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

    # Set seaborn aesthetics
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    
    height = max(6, len(grouped_times) // 3)
    fig, ax = plt.subplots(figsize=(10, height))

    # Define custom colors with seaborn palettes
    correct_color = sns.color_palette("viridis", 1)[0]    # Vibrant green-blue
    incorrect_color = sns.color_palette("rocket", 1)[0]   # Deep red-purple
    time_color = sns.color_palette("gray", 5)[2]         # Medium gray

    for i, (t, true_chord, pred_chord) in enumerate(
        zip(grouped_times, grouped_true_chords, grouped_pred_chords)
    ):
        is_correct = true_chord == pred_chord
        color = correct_color if is_correct else incorrect_color
        alpha = 0.9

        # Show transition time with better styling
        ax.text(-1, -i, f"{t:.2f}s", va="center", ha="right", fontsize=10, 
                color=time_color, fontweight='light', style='italic')

        # True chord on the left with better styling
        ax.text(-0.5, -i, true_chord, va="center", ha="right", fontsize=11, 
                color="black", fontweight='bold')

        # Connecting line with better styling
        ax.plot([0, 1], [-i, -i], color=color, linewidth=2.5, alpha=alpha)

        # Predicted chord on the right with better styling
        ax.text(1.5, -i, pred_chord, va="center", ha="left", fontsize=11, 
                color=color, fontweight='bold')

    # Improved formatting
    ax.set_xlim(-2, 2)
    ax.set_ylim(-len(grouped_times), 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Better title with seaborn styling
    plt.suptitle(f"Chord Prediction Timeline: '{song_name}'", 
               fontsize=16, y=0.98, fontweight='bold')
    plt.title(f"Accuracy: {accuracy:.2f} | F1-score: {f1:.2f}", 
            fontsize=12, style='italic', pad=10)
    
    # Add legend manually
    legend_elements = [
        plt.Line2D([0], [0], color=correct_color, lw=2.5, label='Correct Prediction'),
        plt.Line2D([0], [0], color=incorrect_color, lw=2.5, label='Incorrect Prediction')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), frameon=True, ncol=2)

    plt.tight_layout()
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

    # Set seaborn aesthetics
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)
    
    # Create DataFrame for better seaborn plotting
    plot_data = pd.DataFrame({
        'Song': [name for name in song_names for _ in range(2)],
        'Type': ['True Transitions'] * len(song_names) + ['Predicted Transitions'] * len(song_names),
        'Count': true_transitions + predicted_transitions
    })
    
    # Create plot with seaborn
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        x='Song', 
        y='Count', 
        hue='Type', 
        data=plot_data,
        palette="viridis",
        alpha=0.8
    )
    
    # Improve formatting
    plt.title('True vs. Predicted Chord Transition Counts per Song', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Songs', fontsize=12, labelpad=10)
    plt.ylabel('Number of Chord Transitions', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    
    # Add count labels on the bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width()/2., 
            height + 1, 
            int(height),
            ha="center", fontsize=9, fontweight='bold'
        )
    
    plt.legend(title='', loc='upper right', frameon=True)
    plt.tight_layout()
    plt.show()

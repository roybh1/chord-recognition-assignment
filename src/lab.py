import re
import pandas as pd


def _simplify_chords(chords_df: pd.DataFrame) -> list[str]:
    chords_processed = chords_df["chord"].str.split(":maj")
    chords_processed = [
        elem[0] for elem in chords_processed
    ]  # further process step above to return 1 array
    chords_processed = [elem.split("/")[0] for elem in chords_processed]
    chords_processed = [elem.split("aug")[0] for elem in chords_processed]
    chords_processed = [elem.split(":(")[0] for elem in chords_processed]
    chords_processed = [elem.split("(")[0] for elem in chords_processed]
    chords_processed = [elem.split(":sus")[0] for elem in chords_processed]
    chords_processed = [re.split(":?\d", elem)[0] for elem in chords_processed]
    chords_processed = [elem.replace("dim", "min") for elem in chords_processed]
    chords_processed = [elem.replace("hmin", "min") for elem in chords_processed]
    chords_processed = [re.split(":$", elem)[0] for elem in chords_processed]
    return chords_processed


def read_chord_file(music_file_path: str) -> pd.DataFrame:
    """
    Read a chord file and return a dataframe with the start and end time of each chord and the chord itself.
    """
    chords_annotation = pd.read_csv(music_file_path, sep=" ", header=None)
    if chords_annotation.shape[1] != 3:
        chords_annotation = pd.read_csv(music_file_path, sep="\t", header=None)
    chords_annotation.columns = ["start", "end", "chord"]
    chords_annotation["chord"] = _simplify_chords(chords_annotation)
    # replace silence by probable tonal end
    chords_annotation.loc[chords_annotation["chord"] == "N", "chord"] = (
        chords_annotation["chord"].mode()[0]
    )
    return chords_annotation

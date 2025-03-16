COL_NAMES_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FULL_CHORDS = COL_NAMES_NOTES.copy()
FULL_CHORDS.extend([f"{note}{suffix}" for note in COL_NAMES_NOTES for suffix in ["m"]])

DATA_DIR = "./lab_and_audio_files/"

DEFAULT_TUNING = 0
DEFAULT_NORM = 2
DEFAULT_HOP_LENGTH = 1024
DEFAULT_N_FFT = 4096

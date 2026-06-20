import os
import numpy as np
import pandas as pd
import mne

# =========================
# CONFIG
# =========================
ROOT_DATASET = r"E:\TUH_EEG_DATA\v2.0.3\edf\train"
ROOT_SAVE = r"E:\PreprocessedTrainingSet"

WINDOW_SEC = 10
OVERLAP = 0.5
TARGET_FS = 250

# limit preprocessing
MAX_FILES = 150
processed_count = 0

# =========================
# SAFETY CHECK
# =========================
ROOT_DATASET = os.path.abspath(ROOT_DATASET)
ROOT_SAVE = os.path.abspath(ROOT_SAVE)

if ROOT_SAVE.startswith(ROOT_DATASET):
    raise ValueError("ROOT_SAVE is inside ROOT_DATASET — unsafe configuration.")

# =========================
# TCP pairs
# =========================
tcp_pairs_ar = [
    ("FP1","F7"),("F7","T3"),("T3","T5"),("T5","O1"),
    ("FP2","F8"),("F8","T4"),("T4","T6"),("T6","O2"),
    ("A1","T3"),("T3","C3"),("C3","CZ"),("CZ","C4"),
    ("C4","T4"),("T4","A2"),("FP1","F3"),("F3","C3"),
    ("C3","P3"),("P3","O1"),("FP2","F4"),("F4","C4"),
    ("C4","P4"),("P4","O2"),
]

tcp_pairs_le = [
    ('EEG FP1-LE','EEG F7-LE'),('EEG F7-LE','EEG T3-LE'),
    ('EEG T3-LE','EEG T5-LE'),('EEG T5-LE','EEG O1-LE'),
    ('EEG FP2-LE','EEG F8-LE'),('EEG F8-LE','EEG T4-LE'),
    ('EEG T4-LE','EEG T6-LE'),('EEG T6-LE','EEG O2-LE'),
    ('EEG A1-LE','EEG T3-LE'),('EEG T3-LE','EEG C3-LE'),
    ('EEG C3-LE','EEG CZ-LE'),('EEG CZ-LE','EEG C4-LE'),
    ('EEG C4-LE','EEG T4-LE'),('EEG T4-LE','EEG A2-LE'),
    ('EEG FP1-LE','EEG F3-LE'),('EEG F3-LE','EEG C3-LE'),
    ('EEG C3-LE','EEG P3-LE'),('EEG P3-LE','EEG O1-LE'),
    ('EEG FP2-LE','EEG F4-LE'),('EEG F4-LE','EEG C4-LE'),
    ('EEG C4-LE','EEG P4-LE'),('EEG P4-LE','EEG O2-LE'),
]

tcp_names = [
    'FP1-F7','F7-T3','T3-T5','T5-O1',
    'FP2-F8','F8-T4','T4-T6','T6-O2',
    'A1-T3','T3-C3','C3-CZ','CZ-C4','C4-T4','T4-A2',
    'FP1-F3','F3-C3','C3-P3','P3-O1',
    'FP2-F4','F4-C4','C4-P4','P4-O2'
]

# =========================
# WINDOWING
# =========================
def create_windows(raw):
    sfreq = int(raw.info['sfreq'])
    win_samples = WINDOW_SEC * sfreq
    step = int(win_samples * (1 - OVERLAP))

    data = raw.get_data()
    windows = []
    start = 0
    while start + win_samples <= data.shape[1]:
        windows.append(data[:, start:start+win_samples])
        start += step
    return np.array(windows), step, sfreq

# =========================
# LABELING
# =========================
def create_labels(csv_path, windows, step, sfreq):
    annotations = pd.read_csv(csv_path, comment='#')
    seiz = annotations[annotations['label'] == 'seiz']
    seiz_events = seiz[['start_time','stop_time']].values

    labels = []
    for i in range(windows.shape[0]):
        ws = (i * step) / sfreq
        we = ws + WINDOW_SEC
        label = 0
        for ss,se in seiz_events:
            if (ws < se) and (we > ss):
                label = 1
                break
        labels.append(label)
    return np.array(labels)

# =========================
# NORMALIZATION
# =========================
def normalize(windows):
    return (windows - np.mean(windows, axis=(1,2), keepdims=True)) / \
           (np.std(windows, axis=(1,2), keepdims=True) + 1e-8)

# =========================
# AR preprocessing
# =========================
def preprocess_ar(edf):
    raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
    raw.rename_channels(lambda x: x.replace("EEG ","").replace("-REF",""))

    tcp_data = []
    for a,b in tcp_pairs_ar:
        if a in raw.ch_names and b in raw.ch_names:
            tcp_data.append(raw.get_data(picks=a)-raw.get_data(picks=b))

    tcp_data = np.vstack(tcp_data)
    info = mne.create_info(tcp_names, raw.info['sfreq'], ch_types='eeg')
    raw_tcp = mne.io.RawArray(tcp_data, info, verbose=False)

    if raw_tcp.info['sfreq'] != TARGET_FS:
        raw_tcp.resample(TARGET_FS)

    raw_tcp.filter(0.5,40,verbose=False)
    raw_tcp.notch_filter(50,verbose=False)

    return create_windows(raw_tcp)

# =========================
# LE preprocessing
# =========================
def preprocess_le(edf):
    raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)

    tcp_raw = mne.set_bipolar_reference(
        raw,
        anode=[p[0] for p in tcp_pairs_le],
        cathode=[p[1] for p in tcp_pairs_le],
        ch_name=tcp_names,
        drop_refs=False,
        verbose=False
    )
    tcp_raw.pick_channels(tcp_names)

    if tcp_raw.info['sfreq'] != TARGET_FS:
        tcp_raw.resample(TARGET_FS)

    tcp_raw.filter(0.5,40,verbose=False)
    tcp_raw.notch_filter(50,verbose=False)

    return create_windows(tcp_raw)

# =========================
# MAIN LOOP
# =========================
for root, dirs, files in os.walk(ROOT_DATASET):

    folder = os.path.basename(root)

    if folder == "01_tcp_ar":
        montage = "AR"
    elif folder == "02_tcp_le":
        montage = "LE"
    else:
        continue

    edfs = [f for f in files if f.endswith(".edf")]

    for edf in edfs:
        if processed_count >= MAX_FILES:
            break

        base = os.path.splitext(edf)[0]
        csv_path = os.path.join(root, base + ".csv_bi")
        if not os.path.exists(csv_path):
            continue

        edf_path = os.path.join(root, edf)

        rel = os.path.relpath(root, ROOT_DATASET)
        save_dir = os.path.join(ROOT_SAVE, rel)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, base + ".npz")
        if os.path.exists(save_path):
            continue

        try:
            if montage == "AR":
                windows, step, sfreq = preprocess_ar(edf_path)
            else:
                windows, step, sfreq = preprocess_le(edf_path)

            labels = create_labels(csv_path, windows, step, sfreq)
            windows = normalize(windows)

            np.savez_compressed(
                save_path,
                windows=windows.astype(np.float32),
                labels=labels.astype(np.int8)
            )

            processed_count += 1

        except Exception:
            continue

    if processed_count >= MAX_FILES:
        break
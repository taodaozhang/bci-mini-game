"""Read the same eligible trials as the old benchmark, with 22 shared channels."""
import json
import math
from pathlib import Path
import numpy as np
import pyedflib
from scipy.io import loadmat
from scipy.signal import resample_poly
from model_api import ROOT, EXTERNAL, CHANNELS


def prepare(name):
    destination = ROOT / 'data' / f'{name}_22channels.npz'
    if destination.exists():
        return dict(np.load(destination, allow_pickle=False))
    original = dict(np.load(EXTERNAL / 'data/prepared' / name / 'all_epochs.npz', allow_pickle=False))
    index = {uid: i for i, uid in enumerate(original['trial_id'])}
    X = np.empty((len(index), 22, 1000), dtype=np.float32)
    filled = np.zeros(len(index), dtype=bool)
    if name == 'PhysioNet':
        for subject in range(1, 110):
            for run in [4, 8, 12]:
                file = EXTERNAL / f'data/raw/physionet/S{subject:03d}/S{subject:03d}R{run:02d}.edf'
                with pyedflib.EdfReader(str(file)) as reader:
                    labels = [c.replace('.', '').strip().upper() for c in reader.getSignalLabels()]
                    indices = [labels.index(c.upper()) for c in CHANNELS]
                    frequencies = [reader.getSampleFrequency(i) for i in indices]
                    assert len(set(frequencies)) == 1
                    fs = int(frequencies[0])
                    signals = []
                    for i in indices:
                        unit = reader.getPhysicalDimension(i).strip().replace('µ', 'u').replace('μ', 'u').lower()
                        factor = {'uv': 1, 'mv': 1000, 'v': 1e6}[unit]
                        signals.append(reader.readSignal(i)*factor)
                    signals = np.stack(signals)
                    onsets, _, annotations = reader.readAnnotations()
                for event, (onset, annotation) in enumerate(zip(onsets, annotations)):
                    uid = f'S{subject:03d}_R{run:02d}_E{event:03d}'
                    if uid not in index:
                        continue
                    i = index[uid]
                    assert original['y'][i] == (0 if annotation == 'T1' else 1)
                    start = int(round(float(onset)*fs))
                    epoch = signals[:, start:start+4*fs]
                    if fs != 250:
                        divisor = math.gcd(fs, 250)
                        epoch = resample_poly(epoch, 250//divisor, fs//divisor, axis=-1)
                    X[i] = epoch
                    filled[i] = True
            if subject % 30 == 0:
                print(f'Preparing PhysioNet 22 channels: {subject}/109 people', flush=True)
    else:
        for subject in range(1, 10):
            for part in 'TE':
                file = EXTERNAL / f'data/raw/bnci001/A{subject:02d}{part}.mat'
                runs = np.atleast_1d(loadmat(file, struct_as_record=False, squeeze_me=True)['data'])
                for run_idx, run in enumerate(runs):
                    for event, onset in enumerate(np.asarray(run.trial, dtype=np.int64).ravel()):
                        uid = f'A{subject:02d}_{part}_R{run_idx:02d}_E{event:03d}'
                        if uid not in index:
                            continue
                        i = index[uid]
                        assert original['y'][i] == int(np.asarray(run.y).ravel()[event])-1
                        start = int(onset)-1+500
                        X[i] = run.X[start:start+1000, :22].T
                        filled[i] = True
    assert filled.all() and np.isfinite(X).all()
    original['X'] = X
    original['channels'] = np.asarray(CHANNELS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **original)
    print(f'{name}: saved {X.shape}, identical trial IDs/labels to old benchmark', flush=True)
    return original

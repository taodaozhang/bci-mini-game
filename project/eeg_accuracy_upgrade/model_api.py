"""Portable EEG inference. 22 named channels, 250 Hz, microvolts, 4 seconds."""
from pathlib import Path
import sys
import joblib
import numpy as np
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT.parent / 'external_eeg_tests'
sys.path.insert(0, str(ROOT / 'legacy_helpers'))
import test_external_eeg as old

CHANNELS = list(old.BNCI_CHANNELS)
NINE = list(old.CHANNELS)
NINE_INDICES = [CHANNELS.index(c) for c in NINE]
BANDS = [(8, 12), (12, 20), (20, 30), (8, 30)]


def check_input(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[None]
    if X.ndim != 3 or X.shape[1:] != (22, 1000) or not np.isfinite(X).all():
        raise ValueError('Input must be finite (n,22,1000) or (22,1000), 250Hz, microvolts. Channel order is model_api.CHANNELS.')
    return X


def covariance_bank(X):
    X = check_input(X)
    output = np.empty((len(X), len(BANDS), 22, 22))
    for start in range(0, len(X), 128):
        batch = X[start:start+128].copy()
        batch -= batch.mean(axis=1, keepdims=True)
        for b, (lo, hi) in enumerate(BANDS):
            sos = butter(4, [lo, hi], btype='bandpass', fs=250, output='sos')
            z = sosfilt(sos, batch, axis=-1)[..., 125:]
            z -= z.mean(axis=-1, keepdims=True)
            output[start:start+len(batch), b] = np.einsum('nct,ndt->ncd', z, z)/(z.shape[-1]-1)
    return output


def logcov_features(covs, channels):
    if channels == 9:
        covs = covs[:, :, NINE_INDICES, :][:, :, :, NINE_INDICES]
        projection = np.eye(9)-np.ones((9, 9))/9
        covs = np.einsum('ij,nbjk,kl->nbil', projection, covs, projection)
    n = covs.shape[-1]
    trace = np.maximum(np.trace(covs, axis1=-2, axis2=-1), 1e-12)
    normalized = covs/trace[..., None, None]
    # Fixed 5% shrinkage makes average-reference covariance positive definite.
    normalized = 0.95*normalized+0.05*np.eye(n)/n
    values, vectors = np.linalg.eigh(normalized)
    logs = np.einsum('...ik,...k,...jk->...ij', vectors, np.log(np.maximum(values, 1e-12)), vectors)
    row, col = np.triu_indices(n)
    features = logs[..., row, col]
    features[..., row != col] *= np.sqrt(2)
    return features.reshape(len(covs), -1)


def predict_with_cache(model, X, covs=None, feature_cache=None, baseline_covs=None):
    kind = model['kind']
    if kind == 'aligned_logcov':
        raise ValueError('Batch-aligned model requires predict_aligned with participant IDs.')
    if kind == 'vote':
        predictions = [predict_with_cache(member, X, covs, feature_cache, baseline_covs) for member in model['members']]
        return (np.stack(predictions).sum(axis=0) >= 2).astype(np.int64)
    if kind == 'baseline':
        if baseline_covs is None:
            baseline_covs = old.compute_covs(X[:, NINE_INDICES])
        legacy = model['legacy']
        return legacy['classifier'].predict(old.base.features(baseline_covs, legacy['spatial_filters']))
    count = model['feature_channels']
    if feature_cache is not None and count in feature_cache:
        features = feature_cache[count]
    else:
        covs = covariance_bank(X) if covs is None else covs
        features = logcov_features(covs, count)
    return model['classifier'].predict(model['scaler'].transform(features))


def predict_eeg(eeg, model_path, sampling_rate=250, channels=None, unit='microvolt'):
    """Return 0=left hand, 1=right hand. No calibrated confidence output."""
    model = joblib.load(Path(model_path))
    if sampling_rate != 250 or unit != 'microvolt':
        raise ValueError('Expected 250 Hz and microvolt units.')
    X = check_input(eeg)
    if channels is not None:
        names = list(channels)
        if len(names) != 22 or set(names) != set(CHANNELS):
            raise ValueError('The file must contain exactly the 22 required EEG channels.')
        X = X[:, [names.index(c) for c in CHANNELS]]
    return predict_with_cache(model, X)


def predict_aligned(eeg, subject_ids, model_path, channels=None, sampling_rate=250, unit='microvolt'):
    """Offline batch adaptation uses unlabeled EEG from each target person."""
    from train_alignment import alignment_features
    model = joblib.load(model_path)
    if model['kind'] != 'aligned_logcov':
        raise ValueError('This API is only for an aligned batch model.')
    if sampling_rate != 250 or unit != 'microvolt':
        raise ValueError('Expected 250 Hz and microvolts.')
    X = check_input(eeg)
    people = np.asarray(subject_ids)
    if people.shape != (len(X),):
        raise ValueError('One subject ID is required for each epoch.')
    if min(np.unique(people, return_counts=True)[1]) < 5:
        raise ValueError('Batch adaptation requires at least five epochs per participant; use the ordinary model for single epochs.')
    if channels is not None:
        names = list(channels)
        if len(names) != 22 or set(names) != set(CHANNELS):
            raise ValueError('Exactly the 22 required channels are needed.')
        X = X[:, [names.index(c) for c in CHANNELS]]
    features = alignment_features(covariance_bank(X), people, model['feature_channels'])
    return model['classifier'].predict(model['scaler'].transform(features))

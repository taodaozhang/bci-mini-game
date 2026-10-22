"""BNCI 2014-004: download, session split, train and evaluate left/right EEG.

Run: python train_left_right_eeg.py
Predict: python train_left_right_eeg.py --predict data/splits/test.npz
No deep-learning/GPU dependency. Existing files are never deleted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from scipy.io import loadmat
from scipy.linalg import eigh
from scipy.signal import butter, sosfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PAGE = 'https://bnci-horizon-2020.eu/database/data-sets'
BASE = 'https://lampx.tugraz.at/~bci/database/004-2014/'
FS = 250
CHANNELS = ['C3', 'Cz', 'C4']
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40), (8, 30)]


def json_save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def download(root, subjects):
    raw = root / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = []
    for subject in subjects:
        for part in 'TE':
            name = f'B{subject:02d}{part}.mat'
            path = raw / name
            if not path.exists():
                for attempt in range(4):
                    try:
                        print(f'Downloading {name}', flush=True)
                        with urllib.request.urlopen(BASE + name, timeout=120) as src, open(path.with_suffix('.part'), 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                        path.with_suffix('.part').replace(path)
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(2)
            # Parse on every run, so a truncated cached download fails visibly.
            data = loadmat(path, struct_as_record=False, squeeze_me=True)['data']
            if len(np.atleast_1d(data)) != (3 if part == 'T' else 2):
                raise ValueError(f'Unexpected number of sessions in {name}')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.append({'file': name, 'url': BASE + name, 'bytes': path.stat().st_size, 'sha256': digest})
    json_save(root / 'data' / 'sources.json', {'page': PAGE, 'license': 'CC BY-ND 4.0', 'files': manifest})


def prepare_splits(root, subjects):
    """MAT trial index is 1-based. Extract [trial+3 s, trial+7 s).

    This fixed window is inside MOABB's [3,7.5] interval. Each epoch is
    a complete trial; no sliding windows or random trial splitting.
    """
    groups = {name: [] for name in ['train', 'validation', 'test']}
    for subject in subjects:
        for part in 'TE':
            file = f'B{subject:02d}{part}.mat'
            sessions = np.atleast_1d(loadmat(root / 'data' / 'raw' / file, struct_as_record=False, squeeze_me=True)['data'])
            for idx, session in enumerate(sessions):
                session_id = idx + (1 if part == 'T' else 4)
                group = 'train' if session_id <= 3 else ('validation' if session_id == 4 else 'test')
                if int(session.fs) != FS:
                    raise ValueError(f'{file}: expected {FS} Hz')
                if list(session.classes) != ['left hand', 'right hand']:
                    raise ValueError(f'{file}: unexpected class mapping')
                trials = np.asarray(session.trial, dtype=np.int64).ravel() - 1
                labels = np.asarray(session.y, dtype=np.int64).ravel() - 1
                artifacts = np.asarray(session.artifacts, dtype=np.int64).ravel()
                if not (len(trials) == len(labels) == len(artifacts)) or not np.isin(labels, [0, 1]).all():
                    raise ValueError(f'{file}: invalid labels/artifact flags')
                for j, (onset, label, artifact) in enumerate(zip(trials, labels, artifacts)):
                    start, end = onset + 3 * FS, onset + 7 * FS
                    epoch = np.asarray(session.X[start:end, :3].T, dtype=np.float32)
                    if epoch.shape != (3, 1000) or not np.isfinite(epoch).all():
                        raise ValueError(f'{file}: invalid epoch {j}')
                    uid = f'B{subject:02d}_S{session_id}_trial{j+1:03d}'
                    groups[group].append((epoch, int(label), subject, session_id, uid, int(artifact), file, int(onset)))
    splits = {}
    directory = root / 'data' / 'splits'
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / 'trial_manifest.csv', 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['split', 'trial_id', 'subject', 'session', 'label', 'artifact', 'source_file', 'trial_onset_zero_based'])
        for name, rows in groups.items():
            values = list(zip(*rows))
            split = dict(X=np.stack(values[0]), y=np.array(values[1], dtype=np.int64), subject=np.array(values[2]), session=np.array(values[3]), trial_id=np.array(values[4]), artifact=np.array(values[5]), fs=np.array(FS), channels=np.array(CHANNELS), unit=np.array('microvolt'))
            np.savez_compressed(directory / f'{name}.npz', **split)
            splits[name] = split
            for row in rows:
                writer.writerow([name, row[4], row[2], row[3], row[1], row[5], row[6], row[7]])
            print(f'{name}: {len(rows)} trials; {sum(split["artifact"] == 0)} clean', flush=True)
    ids = [set(s['trial_id']) for s in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    sessions = [{(int(p), int(q)) for p, q in zip(s['subject'], s['session'])} for s in splits.values()]
    assert not (sessions[0] & sessions[1] or sessions[0] & sessions[2] or sessions[1] & sessions[2])
    return splits


def epoch_covariances(X, bands=BANDS):
    """Causal filtering within each 4 s epoch, reset to zero each time.

    Discard first 0.5 s to reduce filter initialization transient. No future
    epoch samples or cross-trial filter state are used in training/prediction.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3 or X.shape[1:] != (3, 1000) or not np.isfinite(X).all():
        raise ValueError('Expected finite EEG shape (n_trials, 3, 1000), C3/Cz/C4, 250 Hz, microvolts.')
    if np.any(np.ptp(X, axis=-1) < 1e-8):
        raise ValueError('Flat EEG channel detected; check electrodes and input units.')
    output = []
    for lo, hi in bands:
        sos = butter(4, [lo, hi], btype='bandpass', fs=FS, output='sos')
        z = sosfilt(sos, X, axis=-1)[..., 125:]
        z -= z.mean(axis=-1, keepdims=True)
        cov = np.einsum('nct,ndt->ncd', z, z) / (z.shape[-1] - 1)
        output.append(cov)
    return np.stack(output, axis=1)


def fit_spatial(covs, y, mode):
    if mode == 'bandpower':
        return np.repeat(np.eye(3)[None], len(BANDS), axis=0)
    filters = []
    for cov in covs.transpose(1, 0, 2, 3):
        trace = np.trace(cov, axis1=1, axis2=2)
        normalized = cov / np.maximum(trace[:, None, None], 1e-15)
        a, b = [normalized[y == c].mean(axis=0) for c in [0, 1]]
        a = 0.95 * a + 0.05 * np.eye(3) * np.trace(a) / 3
        b = 0.95 * b + 0.05 * np.eye(3) * np.trace(b) / 3
        _, vecs = eigh(a, a + b)
        filters.append(vecs.T)
    return np.stack(filters)


def features(covs, filters):
    power = np.einsum('bkc,nbcd,bkd->nbk', filters, covs, filters)
    power = np.maximum(power, 1e-15)
    relative = power / power.sum(axis=2, keepdims=True)
    return np.log(relative).reshape(len(covs), -1)


def new_classifier(name):
    if name == 'lda':
        classifier = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    elif name == 'linear_svm':
        classifier = SVC(C=1, kernel='linear', class_weight='balanced')
    else:
        classifier = SVC(C=1 if name == 'rbf_svm_c1' else 10, kernel='rbf', class_weight='balanced')
    return make_pipeline(StandardScaler(), classifier)


def select_model(train_covs, train_y, val_covs, val_y):
    """Only training labels fit parameters; validation chooses among 8 candidates."""
    best_score = -1
    best = None
    candidates = []
    for mode in ['bandpower', 'fbcsp']:
        spatial = fit_spatial(train_covs, train_y, mode)
        train_f, val_f = features(train_covs, spatial), features(val_covs, spatial)
        for classifier_name in ['lda', 'linear_svm', 'rbf_svm_c1', 'rbf_svm_c10']:
            classifier = new_classifier(classifier_name)
            classifier.fit(train_f, train_y)
            score = float(balanced_accuracy_score(val_y, classifier.predict(val_f)))
            candidate = {'features': mode, 'classifier': classifier_name, 'validation_balanced_accuracy': score}
            candidates.append(candidate)
            if score > best_score:
                best_score = score
                best = {'spatial_filters': spatial, 'classifier': classifier, 'selection': candidate}
    best.update({'bands': BANDS, 'fs': FS, 'channels': CHANNELS, 'epoch_samples': 1000, 'unit': 'microvolt', 'labels': {0: 'left_hand', 1: 'right_hand'}, 'epoch_interval_from_trial_seconds': [3, 7], 'filter': '4th order causal Butterworth, zero state per epoch, discard initial 0.5s', 'training_rule': 'Clean trials from sessions 1-3 only. Validation session 4 selects parameters; no refit.', 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__}})
    return best, candidates


def wilson(y, pred):
    n = len(y)
    p = np.mean(y == pred)
    z = 1.96
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z * np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half), float(center+half)]


def metrics(y, pred):
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'kappa': float(cohen_kappa_score(y, pred)), 'confusion_matrix_left_right': confusion_matrix(y, pred, labels=[0, 1]).tolist(), 'accuracy_wilson_95_percent': wilson(y, pred)}


def evaluate(root, name, model, test_covs, test, mask):
    prediction = model['classifier'].predict(features(test_covs[mask], model['spatial_filters']))
    truth = test['y'][mask]
    clean = test['artifact'][mask] == 0
    result = {'name': name, 'selection': model['selection'], 'test_all_trials': metrics(truth, prediction), 'test_clean_trials': metrics(truth[clean], prediction[clean])}
    with open(root / 'reports' / f'{name}_test_predictions.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['trial_id', 'subject', 'artifact', 'true_label', 'predicted_label'])
        w.writerows(zip(test['trial_id'][mask], test['subject'][mask], test['artifact'][mask], truth, prediction))
    return result


def predict_eeg(eeg, model_path=None, sampling_rate=250, channels=('C3', 'Cz', 'C4'), unit='microvolt'):
    """Game interface: 4 s epoch(s) -> 0 left, 1 right. Supports 3x1000 input.

    Caller must supply the same bipolar electrode montage/reference used for
    training. Output is a forced two-class decision, not rest detection.
    """
    model = joblib.load(model_path or ROOT / 'models' / 'left_right_global.joblib')
    if sampling_rate != model['fs'] or list(channels) != model['channels'] or unit != model['unit']:
        raise ValueError('Sampling rate/channel order/units differ from trained model.')
    X = np.asarray(eeg)
    if X.ndim == 2:
        X = X[None]
    covs = epoch_covariances(X, model['bands'])
    return model['classifier'].predict(features(covs, model['spatial_filters']))


def run_training(root, subjects):
    for directory in ['models', 'reports']:
        (root / directory).mkdir(parents=True, exist_ok=True)
    download(root, subjects)
    splits = prepare_splits(root, subjects)
    covs = {name: epoch_covariances(data['X']) for name, data in splits.items()}
    tr, va, te = [splits[name] for name in ['train', 'validation', 'test']]
    report = {'dataset': 'BNCI2014-004 / BCI Competition IV 2b', 'source': PAGE, 'split': 'Within-subject cross-session: sessions 1-3 train, 4 validation, 5 test. Subjects overlap across splits.', 'labels': {'0': 'left_hand', '1': 'right_hand'}, 'preprocessing': '3 EEG channels C3/Cz/C4, 250Hz; trial+3 to +7 seconds; causal filter bank; 0.5s warmup discarded; no EOG features.', 'artifact_policy': 'Save all trials and flags. Fit/select using clean train/validation trials. Report all test trials as primary and clean-only as secondary.', 'counts': {name: {'all': len(data['y']), 'clean': int(sum(data['artifact'] == 0)), 'left': int(sum(data['y'] == 0)), 'right': int(sum(data['y'] == 1))} for name, data in splits.items()}, 'models': []}
    for subject in [None, *subjects]:
        name = 'left_right_global' if subject is None else f'left_right_B{subject:02d}'
        train_mask = (tr['artifact'] == 0) & (True if subject is None else tr['subject'] == subject)
        val_mask = (va['artifact'] == 0) & (True if subject is None else va['subject'] == subject)
        test_mask = np.ones(len(te['y']), dtype=bool) if subject is None else te['subject'] == subject
        model, candidates = select_model(covs['train'][train_mask], tr['y'][train_mask], covs['validation'][val_mask], va['y'][val_mask])
        model['subject'] = subject
        joblib.dump(model, root / 'models' / f'{name}.joblib')
        json_save(root / 'reports' / f'{name}_validation_candidates.json', candidates)
        # Test data enter here only after selection. No model is chosen by test results.
        result = evaluate(root, name, model, covs['test'], te, test_mask)
        result['training_n'] = int(sum(train_mask))
        result['validation_n'] = int(sum(val_mask))
        report['models'].append(result)
        print(f'{name}: validation={model["selection"]["validation_balanced_accuracy"]:.3f}, test={result["test_all_trials"]["accuracy"]:.3f}', flush=True)
    individual = report['models'][1:]
    report['personal_models_weighted_test_accuracy'] = sum(x['test_all_trials']['accuracy'] * x['test_all_trials']['n'] for x in individual) / sum(x['test_all_trials']['n'] for x in individual)
    report['versions'] = model['versions']
    json_save(root / 'reports' / 'metrics.json', report)
    make_report(root, report)
    return report


def make_report(root, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = report['models']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    axes[0].bar(['Global'] + [f'B{i:02d}' for i in range(1, len(rows))], [r['test_all_trials']['accuracy']*100 for r in rows], color=['#5668b7'] + ['#269f92']*(len(rows)-1))
    axes[0].axhline(50, linestyle='--', color='gray', label='50% chance baseline')
    axes[0].set(ylim=(0, 100), ylabel='Accuracy (%)', title='Held-out session 5, all trials')
    axes[0].legend(loc='lower right')
    cm = np.array(rows[0]['test_all_trials']['confusion_matrix_left_right'])
    axes[1].imshow(cm, cmap='Blues')
    for (i, j), value in np.ndenumerate(cm):
        axes[1].text(j, i, str(value), ha='center', va='center', color='white' if value > cm.max()/2 else 'black')
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Left', 'Right'], yticklabels=['Left', 'Right'], xlabel='Predicted', ylabel='Actual', title='Global model confusion matrix')
    fig.savefig(root / 'reports' / 'test_results.png', dpi=160)
    plt.close(fig)
    lines = ['# 左右手 EEG 实际训练报告', '', '数据：BNCI2014-004，9 位受试者；按采集会话划分，禁止随机打散相邻片段。', '', '| 集合 | 会话 | 全部试次 | 无伪迹试次 |', '|---|---|---:|---:|']
    for name, session in [('train', '1–3'), ('validation', '4'), ('test', '5')]:
        n = report['counts'][name]
        lines.append(f'| {name} | {session} | {n["all"]} | {n["clean"]} |')
    lines += ['', '训练和验证只使用原始标记无伪迹的试次；测试主指标保留全部试次，另列无伪迹结果。三组含相同受试者，衡量的是已见受试者的跨会话表现，不能当作新用户准确率。', '', '| 模型 | 验证平衡准确率 | 测试准确率（全部） | 测试准确率（无伪迹） | 测试 Kappa |', '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["name"]} | {row["selection"]["validation_balanced_accuracy"]:.2%} | {row["test_all_trials"]["accuracy"]:.2%} | {row["test_clean_trials"]["accuracy"]:.2%} | {row["test_all_trials"]["kappa"]:.3f} |')
    lines += ['', f'9 个个人模型各用于对应受试者，按试次数加权的测试准确率：{report["personal_models_weighted_test_accuracy"]:.2%}。这不是挑选测试分数最高的模型用于新用户。', '', '参数选择：频带功率或 FBCSP + 标准化 + LDA/线性 SVM/RBF SVM，共 8 个预先定义候选，仅比较验证集。CSP、标准化及分类器仅拟合训练集；不把验证集或测试集合并回训练。', '', '少量 EEG 数据适合先训练传统特征模型，并不一定适合从零训练大型神经网络。公开数据可用于建立流程；用户本人仍需采集带左右手标签的校准数据，并匹配电极、参考方式与单位。', '', '当前模型输入为 250 Hz、C3/Cz/C4 三路双极 EEG、微伏单位、4 秒窗口；输出 0=左手、1=右手。它没有学习静息或其他动作，连续输入时始终会强制输出左右之一；暂不应把每个输出直接当作可靠游戏指令。', '', 'Wilson 95% 区间见 metrics.json；这是逐试次近似区间，同会话相关性会影响区间解释。', '', '来源：https://bnci-horizon-2020.eu/database/data-sets', '官方说明：https://bnci-horizon-2020.eu/database/data-sets/004-2014/description.pdf', '引用：Leeb et al., 2007, IEEE TNSRE, DOI 10.1109/TNSRE.2007.906956；Tangermann et al., 2012, DOI 10.3389/fnins.2012.00055。', '原始数据许可：CC BY-ND 4.0；保留原始文件及来源。本地预处理划分用于模型实验，分享原始/处理后数据时应核对相应许可。', '解析参考：https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bnci/base.py', '窗口参考：https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bnci/bnci_2014.py', '']
    (root / 'reports' / '训练报告.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Project output directory')
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 10)))
    parser.add_argument('--predict', type=Path, help='NPZ containing X, or NPY epoch data')
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    if args.predict:
        data = np.load(args.predict, allow_pickle=False)
        X = data['X'] if isinstance(data, np.lib.npyio.NpzFile) else data
        prediction = predict_eeg(X, args.model or args.root / 'models' / 'left_right_global.joblib')
        print(json.dumps({'labels': prediction.tolist(), 'mapping': {'0': 'left_hand', '1': 'right_hand'}}))
    else:
        if not args.subjects or not all(1 <= s <= 9 for s in args.subjects) or len(set(args.subjects)) != len(args.subjects):
            parser.error('subjects must be distinct integers from 1 to 9')
        with threadpool_limits(limits=2):
            run_training(args.root.resolve(), args.subjects)


if __name__ == '__main__':
    raise SystemExit('Helper for the frozen old model. Run test_external_eeg.py for the external benchmark.')

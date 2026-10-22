"""External EEG benchmark. Labels are always 0=left imagery, 1=right imagery.

python test_external_eeg.py             # prepare original data and run tests
python test_external_eeg.py --evaluate # reuse prepared epochs
python test_external_eeg.py --prepare  # prepare without fitting/testing

Download first with download_external.py. No original model is modified.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pyedflib
import scipy
import sklearn
from scipy.io import loadmat
from scipy.linalg import eigh
from scipy.signal import butter, resample_poly, sosfilt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from threadpoolctl import threadpool_limits

import eeg_baseline as base

ROOT = Path(__file__).resolve().parent
CHANNELS = ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4']
BNCI_CHANNELS = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
DATASETS = ['PhysioNet', 'BNCI2014_001']
SEED = 20261015


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def protocol():
    return {'seed': SEED, 'labels': {'0': 'left_hand_imagery', '1': 'right_hand_imagery'}, 'data_selection': {'PhysioNet': 'All 109 subjects, runs 4,8,12 only. T1=left, T2=right. First 4 seconds of each annotated imagery event. Reject events shorter than 4 seconds, nonfinite/flat windows, or missing required channels. Never stretch a short event.', 'BNCI2014_001': 'All 9 subjects, T and E sessions. Original classes 1,2 only; trial start +2 to +6 seconds. Keep expert artifact flags.'}, 'raw_epoch_channels': CHANNELS, 'epoch_samples': 1000, 'fs': 250, 'resampling': 'EDF physical units -> microvolt; per-epoch scipy resample_poly to 250Hz; offline evaluation only.', 'old_model': 'Frozen original BNCI2014-004 model, using native-reference C3/Cz/C4 channels. No fitting, threshold tuning, label flipping or target normalization. Reference/montage differs from original bipolar training; exploratory transfer with acquisition mismatch.', 'new_models': 'Nine common motor channels, average reference over these nine within each epoch. Fourth-order causal Butterworth bank; zero initial state per epoch; discard first 0.5s. Fit CSP/scaler/classifier only on clean source training trials.', 'candidate_grid': {'features': ['bandpower', 'fbcsp'], 'classifiers': ['lda', 'linear_svm', 'rbf_svm_c1', 'rbf_svm_c10'], 'selection_metric': 'validation subject-macro balanced accuracy'}, 'cross_subject': {'PhysioNet': '5 fixed subject folds; in each round test one fold, validate next fold, train remaining three. Each eligible person tested exactly once.', 'BNCI2014_001': '9 rounds; hold one person for test and next person for validation; train remaining seven. Each person tested exactly once.'}, 'cross_dataset': 'Fixed FBCSP + shrinkage LDA, with no target data or labels used for fit/selection. Train all clean source trials; test all target trials. Two directions. Hyperparameters fixed before evaluation.', 'reporting': 'Report all eligible test trials as primary, clean-only secondary where flags exist, pooled/macro accuracy, per-person scores, confusion matrices, predictions. Subject bootstrap CI resamples people, not correlated epochs. No score-based exclusions.'}


def valid_epoch(X):
    return X.shape == (9, 1000) and np.isfinite(X).all() and np.all(np.ptp(X, axis=1) > 1e-8)


def save_prepared(name, rows, excluded, files):
    if not rows:
        raise ValueError(f'{name}: no usable epochs')
    X, y, subjects, sessions, ids, artifacts = zip(*rows)
    data = {'X': np.stack(X).astype(np.float32), 'y': np.asarray(y, dtype=np.int64), 'subject': np.asarray(subjects), 'session': np.asarray(sessions), 'trial_id': np.asarray(ids), 'artifact': np.asarray(artifacts), 'channels': np.asarray(CHANNELS), 'fs': np.asarray(250), 'unit': np.asarray('microvolt')}
    directory = ROOT / 'data/prepared' / name
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / 'all_epochs.npz', **data)
    manifest = [{'trial_id': uid, 'subject': int(s), 'session_or_run': int(r), 'label': int(label), 'artifact': int(a)} for uid, s, r, label, a in zip(ids, subjects, sessions, y, artifacts)]
    write_csv(directory / 'trials.csv', manifest)
    save_json(directory / 'quality_audit.json', {'n_files': files, 'n_epochs': len(y), 'eligible_subjects': np.unique(subjects).tolist(), 'counts_by_subject': {str(s): int(sum(data['subject'] == s)) for s in np.unique(subjects)}, 'excluded': excluded})
    print(f'{name}: prepared {len(y)} trials / {len(np.unique(subjects))} eligible people; {len(excluded)} audit exclusions', flush=True)
    return data


def prepare_physionet():
    rows, excluded = [], []
    files = 0
    for subject in range(1, 110):
        for run in [4, 8, 12]:
            file = ROOT / f'data/raw/physionet/S{subject:03d}/S{subject:03d}R{run:02d}.edf'
            if not file.exists():
                raise FileNotFoundError(file)
            with pyedflib.EdfReader(str(file)) as reader:
                labels = [n.replace('.', '').strip().upper() for n in reader.getSignalLabels()]
                if any(c.upper() not in labels for c in CHANNELS):
                    excluded.append({'file': file.name, 'event': -1, 'reason': 'missing_required_channels'})
                    continue
                indices = [labels.index(c.upper()) for c in CHANNELS]
                rates = [reader.getSampleFrequency(c) for c in indices]
                if len(set(rates)) != 1 or not float(rates[0]).is_integer():
                    raise ValueError(f'{file}: unsupported rates {rates}')
                fs = int(rates[0])
                signals = []
                for idx in indices:
                    unit = reader.getPhysicalDimension(idx).strip().replace('µ', 'u').replace('μ', 'u').lower()
                    factor = {'uv': 1, 'mv': 1000, 'v': 1e6}.get(unit)
                    if factor is None:
                        raise ValueError(f'{file}: unsupported physical unit {unit}')
                    signals.append(reader.readSignal(idx) * factor)
                signals = np.stack(signals)
                onsets, durations, annotations = reader.readAnnotations()
            files += 1
            for idx, (onset, duration, annotation) in enumerate(zip(onsets, durations, annotations)):
                if annotation not in ['T1', 'T2']:
                    continue
                next_event = float(onsets[idx+1]) if idx+1 < len(onsets) else signals.shape[1]/fs
                available = min(float(duration), next_event-float(onset), signals.shape[1]/fs-float(onset))
                if available < 4 - 1e-6:
                    excluded.append({'file': file.name, 'event': idx, 'reason': f'imagery_event_shorter_than_4s:{available:.4f}'})
                    continue
                start = int(round(float(onset)*fs))
                epoch = signals[:, start:start+4*fs]
                if fs != 250:
                    divisor = math.gcd(fs, 250)
                    epoch = resample_poly(epoch, 250//divisor, fs//divisor, axis=-1)
                epoch = epoch.astype(np.float32)
                if not valid_epoch(epoch):
                    excluded.append({'file': file.name, 'event': idx, 'reason': 'invalid_or_flat_epoch'})
                    continue
                uid = f'S{subject:03d}_R{run:02d}_E{idx:03d}'
                rows.append((epoch, 0 if annotation == 'T1' else 1, subject, run, uid, -1))
        if subject % 20 == 0:
            print(f'PhysioNet parsing person {subject}/109', flush=True)
    return save_prepared('PhysioNet', rows, excluded, files)


def prepare_bnci():
    rows, excluded = [], []
    indices = [BNCI_CHANNELS.index(c) for c in CHANNELS]
    files = 0
    for subject in range(1, 10):
        for session, part in enumerate('TE', 1):
            file = ROOT / f'data/raw/bnci001/A{subject:02d}{part}.mat'
            runs = np.atleast_1d(loadmat(file, struct_as_record=False, squeeze_me=True)['data'])
            files += 1
            for run_idx, run in enumerate(runs):
                trials = np.asarray(run.trial, dtype=np.int64).ravel()
                if not len(trials):
                    continue
                if int(run.fs) != 250 or run.X.shape[1] != 25:
                    raise ValueError(f'{file}: unexpected shape/rate')
                labels = np.asarray(run.y, dtype=np.int64).ravel()
                flags = np.asarray(run.artifacts, dtype=np.int64).ravel()
                assert len(trials) == len(labels) == len(flags)
                assert list(run.classes)[:2] == ['left hand', 'right hand']
                for idx, (onset, label, artifact) in enumerate(zip(trials, labels, flags)):
                    if label not in [1, 2]:
                        continue
                    start = int(onset) - 1 + 2*250
                    epoch = run.X[start:start+1000, indices].T.astype(np.float32)
                    uid = f'A{subject:02d}_{part}_R{run_idx:02d}_E{idx:03d}'
                    if not valid_epoch(epoch):
                        excluded.append({'file': file.name, 'event': uid, 'reason': 'invalid_or_flat_epoch'})
                        continue
                    rows.append((epoch, int(label)-1, subject, session, uid, int(artifact)))
    return save_prepared('BNCI2014_001', rows, excluded, files)


def compute_covs(X):
    output = np.empty((len(X), len(base.BANDS), 9, 9), dtype=np.float64)
    for start in range(0, len(X), 128):
        batch = X[start:start+128].astype(np.float64)
        batch -= batch.mean(axis=1, keepdims=True)
        for b, (lo, hi) in enumerate(base.BANDS):
            sos = butter(4, [lo, hi], btype='bandpass', fs=250, output='sos')
            z = sosfilt(sos, batch, axis=-1)[..., 125:]
            z -= z.mean(axis=-1, keepdims=True)
            output[start:start+len(batch), b] = np.einsum('nct,ndt->ncd', z, z)/(z.shape[-1]-1)
    return output


def fit_spatial(covs, y, mode):
    if mode == 'bandpower':
        return np.repeat(np.eye(9)[None], len(base.BANDS), axis=0)
    result = []
    for cov in covs.transpose(1, 0, 2, 3):
        trace = np.maximum(np.trace(cov, axis1=1, axis2=2), 1e-15)
        normalized = cov/trace[:, None, None]
        a, b = [normalized[y == c].mean(axis=0) for c in [0, 1]]
        a = 0.95*a+0.05*np.eye(9)*np.trace(a)/9
        b = 0.95*b+0.05*np.eye(9)*np.trace(b)/9
        _, vectors = eigh(a, a+b)
        # Three largest/smallest generalized eigenvectors, fit on source train.
        result.append(vectors[:, [0, 1, 2, 6, 7, 8]].T)
    return np.stack(result)


def subject_score(y, pred, subject):
    return float(np.mean([balanced_accuracy_score(y[subject == s], pred[subject == s]) for s in np.unique(subject)]))


def select_model(covs, data, train, validation):
    best_score, best = -1, None
    candidates = []
    for mode in ['bandpower', 'fbcsp']:
        filters = fit_spatial(covs[train], data['y'][train], mode)
        train_f = base.features(covs[train], filters)
        val_f = base.features(covs[validation], filters)
        for name in ['lda', 'linear_svm', 'rbf_svm_c1', 'rbf_svm_c10']:
            classifier = base.new_classifier(name)
            classifier.fit(train_f, data['y'][train])
            pred = classifier.predict(val_f)
            score = subject_score(data['y'][validation], pred, data['subject'][validation])
            entry = {'features': mode, 'classifier': name, 'validation_subject_macro_balanced_accuracy': score}
            candidates.append(entry)
            if score > best_score:
                best_score = score
                best = {'spatial_filters': filters, 'classifier': classifier, 'selection': entry}
    best.update({'channels': CHANNELS, 'reference': 'mean of these nine channels', 'fs': 250, 'epoch_samples': 1000, 'unit': 'microvolt', 'bands': base.BANDS, 'labels': {0: 'left_hand', 1: 'right_hand'}, 'filter': '4th order causal Butterworth, zero state per epoch, first 0.5s discarded'})
    return best, candidates


def predict_saved(eeg, model_path):
    model = joblib.load(model_path)
    X = np.asarray(eeg)
    if X.ndim == 2:
        X = X[None]
    if X.ndim != 3 or X.shape[1:] != (9, 1000) or not np.isfinite(X).all():
        raise ValueError('Expected shape (n,9,1000), channels FC3/FCz/FC4/C3/Cz/C4/CP3/CPz/CP4, microvolts, 250Hz.')
    return model['classifier'].predict(base.features(compute_covs(X), model['spatial_filters']))


def summarize(data, pred):
    truth = data['y']
    result = base.metrics(truth, pred)
    # Replace epoch-independent Wilson CI with a participant bootstrap interval.
    result.pop('accuracy_wilson_95_percent')
    people = []
    for s in np.unique(data['subject']):
        mask = data['subject'] == s
        people.append({'subject': int(s), 'n': int(sum(mask)), 'accuracy': float(accuracy_score(truth[mask], pred[mask])), 'balanced_accuracy': float(balanced_accuracy_score(truth[mask], pred[mask]))})
    result['n_subjects'] = len(people)
    result['subject_macro_accuracy'] = float(np.mean([r['accuracy'] for r in people]))
    result['subject_macro_balanced_accuracy'] = float(np.mean([r['balanced_accuracy'] for r in people]))
    accuracies = np.array([r['accuracy'] for r in people])
    rng = np.random.default_rng(SEED)
    bootstrap = rng.choice(accuracies, size=(2000, len(people)), replace=True).mean(axis=1)
    result['subject_macro_accuracy_bootstrap_95_percent'] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    clean = data['artifact'] == 0
    if clean.any():
        result['clean_test_accuracy'] = float(accuracy_score(truth[clean], pred[clean]))
        result['clean_test_n'] = int(sum(clean))
    return result, people


def record_result(name, data, pred, extra):
    result, people = summarize(data, pred)
    result.update(extra)
    result['experiment'] = name
    write_csv(ROOT / 'reports' / f'{name}_per_subject.csv', people)
    write_csv(ROOT / 'reports' / f'{name}_predictions.csv', [{'trial_id': str(uid), 'subject': int(s), 'artifact': int(a), 'true_label': int(y), 'predicted_label': int(p)} for uid, s, a, y, p in zip(data['trial_id'], data['subject'], data['artifact'], data['y'], pred)])
    print(f'{name}: accuracy={result["accuracy"]:.2%}; macro={result["subject_macro_accuracy"]:.2%}', flush=True)
    return result


def subject_folds(name, subjects):
    values = np.random.default_rng(SEED).permutation(np.unique(subjects))
    folds = np.array_split(values, 5 if name == 'PhysioNet' else len(values))
    output = []
    for idx, test in enumerate(folds):
        validation = folds[(idx+1) % len(folds)]
        train = np.array([s for s in values if s not in test and s not in validation])
        assert not (set(train)&set(test) or set(train)&set(validation) or set(test)&set(validation))
        output.append({'fold': idx, 'train_subjects': train.tolist(), 'validation_subjects': validation.tolist(), 'test_subjects': test.tolist()})
    return output


def cross_subject(name, data, covs):
    folds = subject_folds(name, data['subject'])
    save_json(ROOT / 'data/prepared' / name / 'subject_folds.json', folds)
    preds = np.full(len(data['y']), -1, dtype=np.int64)
    fold_results = []
    for fold in folds:
        train = np.isin(data['subject'], fold['train_subjects']) & (data['artifact'] != 1)
        val = np.isin(data['subject'], fold['validation_subjects']) & (data['artifact'] != 1)
        test = np.isin(data['subject'], fold['test_subjects'])
        if fold['fold'] == 0:
            directory = ROOT / 'data/prepared' / name / 'first_fold_splits'
            directory.mkdir(parents=True, exist_ok=True)
            # Retain all trials/flags in split artifacts; fit/select masks above reject flagged trials.
            for split, people in [('train', fold['train_subjects']), ('validation', fold['validation_subjects']), ('test', fold['test_subjects'])]:
                mask = np.isin(data['subject'], people)
                subset = {k: v[mask] if k in ['X', 'y', 'subject', 'session', 'trial_id', 'artifact'] else v for k, v in data.items()}
                np.savez_compressed(directory / f'{split}.npz', **subset)
        model, candidates = select_model(covs, data, train, val)
        model['split'] = fold
        path = ROOT / 'models' / f'{name}_unseen_subject_fold{fold["fold"]:02d}.joblib'
        joblib.dump(model, path)
        pred = model['classifier'].predict(base.features(covs[test], model['spatial_filters']))
        assert np.all(preds[test] == -1)
        preds[test] = pred
        result = dict(fold, training_trials=int(sum(train)), validation_trials=int(sum(val)), test_trials=int(sum(test)), selection=model['selection'], test_accuracy=float(accuracy_score(data['y'][test], pred)))
        fold_results.append(result)
        save_json(ROOT / 'reports' / f'{name}_fold{fold["fold"]:02d}_validation.json', candidates)
        print(f'{name} unseen-person fold {fold["fold"]+1}/{len(folds)}: test {result["test_accuracy"]:.2%}', flush=True)
    assert np.isin(preds, [0, 1]).all()
    save_json(ROOT / 'reports' / f'{name}_fold_results.json', fold_results)
    return record_result(f'{name}_unseen_subjects', data, preds, {'method': 'Nested subject-isolated evaluation; one test fold and one validation fold, remaining people train. Every eligible person tested once.', 'n_folds': len(folds)})


def cross_dataset(source_name, target_name, source, target, source_covs, target_covs):
    train = source['artifact'] != 1
    filters = fit_spatial(source_covs[train], source['y'][train], 'fbcsp')
    classifier = base.new_classifier('lda')
    classifier.fit(base.features(source_covs[train], filters), source['y'][train])
    name = f'{source_name}_to_{target_name}'
    model = {'spatial_filters': filters, 'classifier': classifier, 'channels': CHANNELS, 'reference': 'mean of these nine channels', 'fs': 250, 'epoch_samples': 1000, 'unit': 'microvolt', 'bands': base.BANDS, 'source_dataset': source_name, 'method': 'Fixed FBCSP + automatic-shrinkage LDA; no target fit or parameter selection.'}
    joblib.dump(model, ROOT / 'models' / f'{name}.joblib')
    pred = classifier.predict(base.features(target_covs, filters))
    return record_result(name, target, pred, {'method': model['method'], 'train_dataset': source_name, 'test_dataset': target_name, 'training_trials': int(sum(train))})


def make_report(results, audits):
    lines = ['# 新增左右手 EEG 数据：下载与泛化测试报告', '', '两个新数据源已下载到本项目 data/raw。只分析运动想象，不混入真实动作、双手/双脚、舌头等其他标签。标签统一为 0=左手想象、1=右手想象。', '', '| 新数据 | 原始文件 | 可用人数 | 可用左右手试次 |', '|---|---:|---:|---:|']
    for name, audit in audits.items():
        lines.append(f'| {name} | {audit["n_files"]} | {len(audit["eligible_subjects"])} | {audit["n_epochs"]} |')
    lines += ['', 'PhysioNet 下载全部 109 人的 327 个 EDF+，只选择 R04/R08/R12。部分事件若不足 4 秒或信号无效则客观剔除；具体人/事件、原因和数量见 quality_audit.json，不按测试分数剔除。BCI IV 2a 下载全部 18 个 MAT，保留左右手类别；原文件中的其他类别仍完整保留。', '', '| 测试 | 全部试次准确率 | 按人平均准确率 | 按人平均平衡准确率 | 人数 |', '|---|---:|---:|---:|---:|']
    for result in results:
        lines.append(f'| {result["experiment"]} | {result["accuracy"]:.2%} | {result["subject_macro_accuracy"]:.2%} | {result["subject_macro_balanced_accuracy"]:.2%} | {result["n_subjects"]} |')
    lines += ['', '## 如何理解三种测试', '', '1. frozen_old：冻结之前 BNCI 2014-004 模型，用 C3/Cz/C4 的新数据直接预测。新数据的参考/导联方式不等于原训练数据的双极导联，所以它是包含采集方式不匹配的探索性压力测试；不能把下降全部归因于算法或人的差异。未做目标数据归一化、调参或标签反转。', '', '2. unseen_subjects：在每套新数据内部，训练、验证、测试三组人员完全不同。PhysioNet 分 5 个按人划分的折；每轮 3 折训练、1 折验证、1 折测试。2a 共 9 轮，每轮 7 人训练、1 人验证、1 人测试。每个人只贡献一次最终测试预测。各折单独训练并保存模型，不能称为一个在全部人上训练后又测同一批人的模型。', '', '3. A_to_B：两套新数据相互迁移。预先固定 FBCSP + 收缩 LDA，仅在来源数据的无伪迹试次上训练，不接触目标数据做参数选择。两套数据采用相同九通道及九通道平均参考。它检验换数据源后的表现，依然受设备、任务和预处理差异影响。', '', '## 处理和评估规则', '', '新模型使用 FC3、FCz、FC4、C3、Cz、C4、CP3、CPz、CP4；250 Hz；4 秒窗口。PhysioNet 每个想象事件取前 4 秒并用 resample_poly 重采样；2a 取试次开始后 2–6 秒。每窗口九通道平均参考、4–40 Hz 滤波器组；因果滤波每窗清零，前 0.5 秒不计入特征。EDF 重采样是离线处理，本结果不代表流式延迟或在线控制性能。', '', '重新训练候选：频带功率/FBCSP × LDA、线性 SVM、RBF SVM(C=1/10)，仅按验证组的按人平均平衡准确率选择。CSP、标准化和分类器只拟合训练组。BCI 数据排除训练/验证中的原始专家伪迹试次，测试主指标保留全部试次。PhysioNet 未提供同样的试次伪迹标记，不冒充“已清理干净”的数据。', '', '所有逐人分数和逐试次预测均已保存；results.json 含混淆矩阵、Kappa、按人 bootstrap 区间及可用的无伪迹测试成绩。各折模型的训练/验证/测试人员清单见 subject_folds.json。每套数据 first_fold_splits 提供第一折三组 NPZ；其他折由同一 all_epochs.npz 与人员清单精确重建。', '', '## 结论边界', '', '之前的 76.25% 是已见受试者的跨会话结果；这里的按人隔离和跨数据源结果更严格，不应把不同测试条件的百分比当作同一基准比较。下载更多数据不会自动保证泛化；个人校准和与设备一致的采集方式仍需要单独验证。这里只测左右手，没有建立静息或拒绝输出机制，不能当作可直接上线的游戏控制器。', '', '2a 与旧 2b 来自相同竞赛系列；公开编号不能据此证明两套库之间的自然人身份完全不重合。PhysioNet 为不同采集项目，是更明确的外部数据来源。按人隔离的各新数据内部测试不会发生人员重叠。', '', '## 来源与引用', '', '- PhysioNet：https://physionet.org/content/eegmmidb/1.0.0/ （下载使用官方 archive 镜像）；Schalk 2009, DOI 10.13026/C28G6P；Schalk et al. 2004, BCI2000, DOI 10.1109/TBME.2004.827072。许可：Open Data Commons Attribution License v1.0。', '- BCI IV 2a / BNCI2014-001：https://bnci-horizon-2020.eu/database/data-sets；官方说明 https://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf；Tangermann et al. 2012, DOI 10.3389/fnins.2012.00055。许可：CC BY-ND 4.0。原始文件保持不变，处理划分用于本地训练实验。', '- 其他已找到、此次没有下载：Cho2017（52 人，64 路 EEG，左右手想象）https://gigadb.org/dataset/100295；OpenBMI/Lee2019（54 人，两次采集）https://gigadb.org/dataset/100542。它们不计入本次实测结果。', '', '每个原始文件的真实 URL、字节数、SHA-256 见 data/sources.json。', '']
    (ROOT / 'reports' / '泛化测试报告.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
    names = [r['experiment'].replace('PhysioNet', 'Physio').replace('BNCI2014_001', 'IV-2a').replace('_', ' ') for r in results]
    values = [r['subject_macro_accuracy']*100 for r in results]
    bars = ax.barh(names, values, color=['#b58456']*2+['#367bb5']*2+['#298a77']*2)
    ax.invert_yaxis()
    ax.axvline(50, color='gray', ls='--', label='50% chance baseline')
    ax.set(xlim=(0, 100), xlabel='Subject-macro accuracy (%)', title='Independent data and unseen-person tests')
    ax.bar_label(bars, fmt='%.2f%%', padding=4)
    ax.legend(loc='lower right')
    fig.savefig(ROOT / 'reports' / 'external_results.png', dpi=160)
    plt.close(fig)


def evaluate(data):
    results = []
    for name in DATASETS:
        d = data[name]
        # C3/Cz/C4 at indices 3,4,5; deliberately preserve native reference for old model.
        pred = base.predict_eeg(d['X'][:, [3, 4, 5], :], ROOT / 'models/frozen_bnci004.joblib')
        results.append(record_result(f'{name}_frozen_old', d, pred, {'method': protocol()['old_model']}))
    covs = {name: compute_covs(d['X']) for name, d in data.items()}
    for name in DATASETS:
        results.append(cross_subject(name, data[name], covs[name]))
    for source, target in [('BNCI2014_001', 'PhysioNet'), ('PhysioNet', 'BNCI2014_001')]:
        results.append(cross_dataset(source, target, data[source], data[target], covs[source], covs[target]))
    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__, 'pyedflib': pyedflib.__version__, 'joblib': joblib.__version__}
    save_json(ROOT / 'reports/results.json', {'protocol': protocol(), 'versions': versions, 'experiments': results})
    audits = {name: json.loads((ROOT / 'data/prepared' / name / 'quality_audit.json').read_text(encoding='utf-8')) for name in DATASETS}
    make_report(results, audits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    # Written before any predictions or score inspection.
    save_json(ROOT / 'reports/protocol.json', protocol())
    with threadpool_limits(limits=2):
        if args.evaluate:
            data = {name: dict(np.load(ROOT / 'data/prepared' / name / 'all_epochs.npz', allow_pickle=False)) for name in DATASETS}
        else:
            data = {'PhysioNet': prepare_physionet(), 'BNCI2014_001': prepare_bnci()}
        if not args.prepare:
            evaluate(data)


if __name__ == '__main__':
    main()

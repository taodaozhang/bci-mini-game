"""Compare predefined algorithms on exactly the previous subject-isolated folds."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

import model_api as api
from prepare_22channels import prepare

ROOT = api.ROOT
SEED = 20261019
DATASETS = ['PhysioNet', 'BNCI2014_001']


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def protocol():
    return {'seed': SEED, 'evaluation': 'Same exact trial IDs, labels, artifact flags and held-out subject folds as the prior 9-channel benchmark. Train/validation/test people are disjoint. Previously inspected public benchmark: repeated development evaluation, not a new blind test.', 'channels': api.CHANNELS, 'comparison': 'Previous 9-channel baseline plus 9/22-channel logarithmic covariance features with shrinkage LDA, RBF SVM(C=1), ExtraTrees(300 trees,min_samples_leaf=5), and MLP(64,16). Nine-channel controls distinguish feature/algorithm changes from extra electrodes.', 'bands': api.BANDS, 'features': 'Each epoch independently: common average reference, causal order-4 Butterworth, discard first 0.5s, covariance/trace, fixed 5% diagonal shrinkage, matrix logarithm, weighted symmetric vectorization. No target batch normalization or label-based feature selection.', 'neural': {'method': 'Backpropagation + Adam', 'hidden_layers': [64, 16], 'activation': 'relu', 'learning_rate': 0.0005, 'l2_alpha': 0.1, 'batch_size': 64, 'max_epochs': 100, 'early_stop_patience': 15, 'minimum_epochs': 15, 'criterion': 'validation subject-macro balanced accuracy', 'sample_weight': 'equal total weight per source training person'}, 'selection': 'All scalers and model weights fit training data only. Select family/channel count/checkpoint solely by validation subject-macro balanced accuracy; ties keep earlier candidate. Majority vote combines validation-selected linear/SVM, tree, and neural models. Final winner among baseline, 8 new models and majority vote selected before any outer-test predictions.', 'deployment': 'Save validation-selected model and validation-selected neural model for every fold. Demo uses fixed fold 0, never highest test-scoring fold. Labels 0=left,1=right. No calibrated confidence output.'}


def fit_neural(train_f, y, val_f, val_y, val_subject, train_subject, seed, label):
    classifier = MLPClassifier(hidden_layer_sizes=(64, 16), activation='relu', solver='adam', alpha=0.1, batch_size=64, learning_rate_init=0.0005, max_iter=1, early_stopping=False, random_state=seed)
    people, counts = np.unique(train_subject, return_counts=True)
    weights = np.ones(len(y))
    for person, count in zip(people, counts):
        weights[train_subject == person] = len(y)/(len(people)*count)
    best_score, best, stale = -1, None, 0
    history = []
    for epoch in range(1, 101):
        classifier.partial_fit(train_f, y, classes=np.array([0, 1]), sample_weight=weights)
        score = api.old.subject_score(val_y, classifier.predict(val_f), val_subject)
        history.append({'epoch': epoch, 'training_cross_entropy_with_regularization': float(classifier.loss_), 'validation_subject_macro_balanced_accuracy': float(score)})
        if score > best_score+1e-9:
            best_score, best, stale = score, copy.deepcopy(classifier), 0
        else:
            stale += 1
        if epoch % 20 == 0:
            print(f'  {label}: epoch {epoch}, loss={classifier.loss_:.4f}, best validation={best_score:.3f}', flush=True)
        if epoch >= 15 and stale >= 15:
            break
    return best, history


def fit_candidate(features, data, train, val, algorithm, count, seed, label):
    scaler = StandardScaler().fit(features[train])
    train_f = scaler.transform(features[train]).astype(np.float32)
    val_f = scaler.transform(features[val]).astype(np.float32)
    y = data['y'][train]
    history = []
    if algorithm == 'lda':
        classifier = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    elif algorithm == 'svm':
        classifier = SVC(C=1, kernel='rbf', class_weight='balanced')
    elif algorithm == 'trees':
        classifier = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, max_features='sqrt', class_weight='balanced', n_jobs=4, random_state=seed)
    else:
        classifier, history = fit_neural(train_f, y, val_f, data['y'][val], data['subject'][val], data['subject'][train], seed, label)
    if algorithm != 'mlp':
        classifier.fit(train_f, y)
    model = {'kind': 'logcov', 'feature_channels': count, 'algorithm': algorithm, 'classifier': classifier, 'scaler': scaler}
    predicted = classifier.predict(val_f)
    score = api.old.subject_score(data['y'][val], predicted, data['subject'][val])
    train_accuracy = float(accuracy_score(y, classifier.predict(train_f)))
    return model, float(score), train_accuracy, history


def metadata(model, dataset, fold, name):
    model.update({'input_channels': api.CHANNELS, 'sampling_rate': 250, 'unit': 'microvolt', 'epoch_samples': 1000, 'bands': api.BANDS, 'dataset': dataset, 'split': fold, 'selected_name': name, 'labels': {0: 'left_hand', 1: 'right_hand'}, 'benchmark_note': 'Repeated public-data benchmark; same folds as earlier experiments, not a fresh blind test.'})
    return model


def run_dataset(name):
    started = time.monotonic()
    data = prepare(name)
    old_data = dict(np.load(api.EXTERNAL / 'data/prepared' / name / 'all_epochs.npz', allow_pickle=False))
    assert np.array_equal(data['trial_id'], old_data['trial_id'])
    assert np.array_equal(data['y'], old_data['y'])
    assert np.allclose(data['X'][:, api.NINE_INDICES], old_data['X'], rtol=1e-6, atol=1e-5)
    folds = json.loads((api.EXTERNAL / 'data/prepared' / name / 'subject_folds.json').read_text(encoding='utf-8'))
    save_json(ROOT / 'reports' / f'{name}_subject_folds.json', folds)
    print(f'{name}: computing logarithmic covariance features', flush=True)
    covs = api.covariance_bank(data['X'])
    features = {count: api.logcov_features(covs, count) for count in [9, 22]}
    # Old covariance/features remain exactly the old algorithm and its original 9 channels.
    baseline_covs = api.old.compute_covs(old_data['X'])
    all_predictions = {}
    all_predictions['selected'] = np.full(len(data['y']), -1, dtype=int)
    all_predictions['neural_selected'] = np.full(len(data['y']), -1, dtype=int)
    fold_records = []
    for fold in folds:
        fid = fold['fold']
        train = np.isin(data['subject'], fold['train_subjects']) & (data['artifact'] != 1)
        val = np.isin(data['subject'], fold['validation_subjects']) & (data['artifact'] != 1)
        test = np.isin(data['subject'], fold['test_subjects'])
        assert not (set(fold['train_subjects']) & set(fold['test_subjects']))
        assert not (set(fold['train_subjects']) & set(fold['validation_subjects']))
        assert not (set(fold['validation_subjects']) & set(fold['test_subjects']))
        print(f'{name} fold {fid+1}/{len(folds)}: train {sum(train)}, validation {sum(val)}, test {sum(test)}', flush=True)
        legacy = joblib.load(api.EXTERNAL / 'models' / f'{name}_unseen_subject_fold{fid:02d}.joblib')
        models = {'baseline_9': {'kind': 'baseline', 'legacy': legacy}}
        legacy_val = legacy['classifier'].predict(api.old.base.features(baseline_covs[val], legacy['spatial_filters']))
        scores = {'baseline_9': api.old.subject_score(data['y'][val], legacy_val, data['subject'][val])}
        legacy_train = legacy['classifier'].predict(api.old.base.features(baseline_covs[train], legacy['spatial_filters']))
        train_scores = {'baseline_9': float(accuracy_score(data['y'][train], legacy_train))}
        for count in [9, 22]:
            for algorithm in ['lda', 'svm', 'trees', 'mlp']:
                candidate = f'logcov_{algorithm}_{count}'
                model, score, train_accuracy, history = fit_candidate(features[count], data, train, val, algorithm, count, SEED+fid, f'{name} F{fid} {candidate}')
                models[candidate], scores[candidate], train_scores[candidate] = model, score, train_accuracy
                if history:
                    write_csv(ROOT / 'reports' / f'{name}_fold{fid:02d}_{candidate}_learning.csv', history)
                print(f'  {candidate}: train={train_accuracy:.3f}, validation={score:.3f}', flush=True)
        linear = max([n for n in scores if '_lda_' in n or '_svm_' in n], key=scores.get)
        tree = max([n for n in scores if '_trees_' in n], key=scores.get)
        neural = max([n for n in scores if '_mlp_' in n], key=scores.get)
        ensemble = {'kind': 'vote', 'members': [models[n] for n in [linear, tree, neural]], 'member_names': [linear, tree, neural]}
        models['majority_vote'] = ensemble
        val_features = {c: f[val] for c, f in features.items()}
        val_pred = api.predict_with_cache(ensemble, data['X'][val], covs[val], val_features, baseline_covs[val])
        scores['majority_vote'] = api.old.subject_score(data['y'][val], val_pred, data['subject'][val])
        train_pred = api.predict_with_cache(ensemble, data['X'][train], covs[train], {c: f[train] for c, f in features.items()}, baseline_covs[train])
        train_scores['majority_vote'] = float(accuracy_score(data['y'][train], train_pred))
        winner = max(scores, key=scores.get)
        selected = metadata(models[winner], name, fold, winner)
        neural_model = metadata(models[neural], name, fold, neural)
        joblib.dump(selected, ROOT / 'models' / f'{name}_selected_fold{fid:02d}.joblib', compress=3)
        joblib.dump(neural_model, ROOT / 'models' / f'{name}_neural_fold{fid:02d}.joblib', compress=3)
        # All choices/checkpoints are now frozen. Only now compute held-out predictions.
        test_features = {c: f[test] for c, f in features.items()}
        for candidate, model in models.items():
            pred = api.predict_with_cache(model, data['X'][test], covs[test], test_features, baseline_covs[test])
            if candidate not in all_predictions:
                all_predictions[candidate] = np.full(len(data['y']), -1, dtype=int)
            assert np.all(all_predictions[candidate][test] == -1)
            all_predictions[candidate][test] = pred
        all_predictions['selected'][test] = all_predictions[winner][test]
        all_predictions['neural_selected'][test] = all_predictions[neural][test]
        record = dict(fold, selected=winner, neural_selected=neural, validation_subject_macro_balanced_accuracies=scores, training_accuracies=train_scores, selected_test_accuracy=float(accuracy_score(data['y'][test], all_predictions['selected'][test])))
        fold_records.append(record)
        save_json(ROOT / 'reports' / f'{name}_fold_results.json', fold_records)
        print(f'  SELECTED {winner}; held-out test {record["selected_test_accuracy"]:.2%}', flush=True)
        if fid == 0:
            directory = ROOT / 'data' / name / 'first_fold'
            directory.mkdir(parents=True, exist_ok=True)
            for split, people in [('train', fold['train_subjects']), ('validation', fold['validation_subjects']), ('test', fold['test_subjects'])]:
                mask = np.isin(data['subject'], people)
                subset = {k: v[mask] if k in ['X', 'y', 'subject', 'session', 'trial_id', 'artifact'] else v for k, v in data.items()}
                np.savez_compressed(directory / f'{split}.npz', **subset)
    results = {}
    for candidate, pred in all_predictions.items():
        assert np.isin(pred, [0, 1]).all()
        summary, _ = api.old.summarize(data, pred)
        results[candidate] = summary
    rows = []
    for i in range(len(data['y'])):
        row = {'trial_id': str(data['trial_id'][i]), 'subject': int(data['subject'][i]), 'true_label': int(data['y'][i]), 'artifact': int(data['artifact'][i])}
        row.update({n: int(p[i]) for n, p in all_predictions.items()})
        rows.append(row)
    write_csv(ROOT / 'reports' / f'{name}_all_predictions.csv', rows)
    output = {'dataset': name, 'n_trials': len(data['y']), 'n_people': len(np.unique(data['subject'])), 'seconds': time.monotonic()-started, 'algorithms': results}
    save_json(ROOT / 'reports' / f'{name}_results.json', output)
    print(f'{name} DONE: baseline {results["baseline_9"]["accuracy"]:.2%}, validation-selected {results["selected"]["accuracy"]:.2%}, neural {results["neural_selected"]["accuracy"]:.2%}', flush=True)
    return output


def make_report(results):
    lines = ['# 公开 EEG 数据算法升级实验', '', '本次使用已下载的公开数据离线比较算法，不连接脑电设备。每一个试次 ID、标签、伪迹标记、外层训练/验证/测试人员划分均与上次完全相同。', '', '**评估边界：这些公开测试数据以前已经查看过成绩，本次是重复基准比较，不是全新盲测；不能据此保证未知用户或设备的准确率。**', '', '| 数据 | 原九通道基线 | 验证集选出的新方案 | 变化（百分点） | 验证集选出的神经网络 |', '|---|---:|---:|---:|---:|']
    for dataset in results:
        a = dataset['algorithms']
        old_score, new_score = a['baseline_9']['accuracy'], a['selected']['accuracy']
        lines.append(f'| {dataset["dataset"]} | {old_score:.2%} | {new_score:.2%} | {(new_score-old_score)*100:+.2f} | {a["neural_selected"]["accuracy"]:.2%} |')
    lines += ['', '## 实验方法', '', '预先固定候选：原九通道模型；九/二十二通道的矩阵对数协方差特征 + 收缩 LDA、RBF SVM、ExtraTrees、MLP；以及三个模型的多数投票。所有候选和停止条件在外层测试前写入 protocol.json。', '', '神经网络使用两个隐藏层（64、16），ReLU，反向传播+Adam，L2=0.1；训练人员等权重，标准化只拟合训练试次。最多 100 轮，按独立验证人员的平均平衡准确率选检查点，连续 15 轮无提升且已训练至少 15 轮后停止。没有用 sklearn 内部随机试次切分充当跨人验证。', '', '每折先用验证人员选择算法、通道数、神经网络停止轮次和投票成员，再计算测试人员预测。报告中的 selected 是这套验证选择规则的成绩；不能事后从下表挑外层测试分最高的算法冒充已验证的部署选择。原基线也在候选内，因此验证成绩不会低于旧方案，但测试成绩仍可能下降。', '', '增加电极与更换算法是两个同时考察的因素。九通道新算法是控制组；二十二通道模型利用更多额外空间信息，提升不能全归因于算法。输入窗口仍为 250 Hz、4 秒、微伏。协方差按试次独立归一化，不使用整批测试样本去估计对齐参数。', '', '## 所有预定候选的外层测试结果', '', '| 算法 | PhysioNet | BNCI2014-001 |', '|---|---:|---:|']
    lookup = {d['dataset']: d['algorithms'] for d in results}
    keys = list(results[0]['algorithms'])
    for name in keys:
        values = [f'{lookup[d][name]["accuracy"]:.2%}' if d in lookup else '—' for d in DATASETS]
        lines.append(f'| {name} | {values[0]} | {values[1]} |')
    lines += ['', 'baseline_9=旧方案；logcov=矩阵对数协方差；lda=收缩线性判别；svm=径向核支持向量机；trees=随机化树集成；mlp=反向传播神经网络；后缀 9/22 表示使用的特征通道数。neural_selected 在每折仅由验证集选择九/二十二通道神经网络。', '', '各折训练准确率、验证成绩、最终选择见 *_fold_results.json；神经网络逐轮损失和验证曲线见 *_learning.csv；所有候选的逐试次预测见 *_all_predictions.csv。拟合得更好不代表泛化更好，应同时查看训练与独立测试之间的差距。', '', '## 怎么使用', '', '双击 run_demo.cmd 加载 PhysioNet 第一折的验证集选定模型，预测该折未参加训练/选参的测试人员。双击 run_neural_demo.cmd 可明确使用反向传播训练的神经网络。演示和完整基准成绩有区别：第一折只是所有测试人员中的一部分。', '', '所有新模型的输入接口固定二十二通道；模型内部可能只选九通道特征。model_api.predict_eeg 是统一入口。默认演示使用预先固定的 fold00，没有按测试成绩挑选最好的一折。原项目与外部测试目录需要保持为同级目录，脚本会复用其中的原始数据及旧基线代码。', '', '## 文献与实现', '', '- 协方差表征背景：Barachant et al., 2012, https://pubmed.ncbi.nlm.nih.gov/22010143/ 。本项目采用简单矩阵对数特征，并非声称完整复现该论文的黎曼切空间算法。', '- MLP/反向传播与 Adam 实现：https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html', '- 数据来源：PhysioNet https://physionet.org/content/eegmmidb/1.0.0/ ；BNCI https://bnci-horizon-2020.eu/database/data-sets 。原始文件、许可及 SHA-256 沿用 external_eeg_tests/data/sources.json。', '']
    (ROOT / 'reports' / '算法升级报告.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    x = np.arange(len(results))
    for offset, key, label, color in [(-.24, 'baseline_9', 'Previous baseline', '#8c929a'), (0, 'selected', 'Validation-selected method', '#2878b5'), (.24, 'neural_selected', 'Backprop neural network', '#309f84')]:
        bars = ax.bar(x+offset, [r['algorithms'][key]['accuracy']*100 for r in results], width=.24, label=label, color=color)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=[d['dataset'] for d in results], ylim=(0, 100), ylabel='Held-out accuracy (%)', title='Same trials and same subject-isolated folds')
    ax.axhline(50, color='gray', ls='--', lw=1)
    ax.legend(loc='lower right')
    fig.savefig(ROOT / 'reports' / 'comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    for folder in ['models', 'reports', 'data']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    save_json(ROOT / 'reports/protocol.json', protocol())
    results = []
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        for name in args.datasets:
            results.append(run_dataset(name))
    save_json(ROOT / 'reports/results.json', {'protocol': protocol(), 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__}, 'datasets': results})
    make_report(results)


if __name__ == '__main__':
    main()

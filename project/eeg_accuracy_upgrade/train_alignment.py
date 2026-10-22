"""Separate offline adaptation experiment; target EEG signals, never target labels, align batches."""
import argparse
import json
import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
import model_api as api
from prepare_22channels import prepare
from train_compare import save_json, write_csv, metadata

ROOT = api.ROOT


def alignment_features(covs, subjects, count):
    if count == 9:
        covs = covs[:, :, api.NINE_INDICES, :][:, :, :, api.NINE_INDICES]
        p = np.eye(9)-np.ones((9, 9))/9
        covs = np.einsum('ij,nbjk,kl->nbil', p, covs, p)
    trace = np.maximum(np.trace(covs, axis1=-2, axis2=-1), 1e-12)
    covs = .95*covs/trace[..., None, None]+.05*np.eye(count)/count
    aligned = np.empty_like(covs)
    for subject in np.unique(subjects):
        mask = subjects == subject
        reference = covs[mask].mean(axis=0)
        values, vectors = np.linalg.eigh(reference)
        inverse = np.einsum('bik,bk,bjk->bij', vectors, 1/np.sqrt(np.maximum(values, 1e-12)), vectors)
        aligned[mask] = np.einsum('bij,nbjk,bkl->nbil', inverse, covs[mask], inverse)
    aligned /= np.maximum(np.trace(aligned, axis1=-2, axis2=-1)[..., None, None], 1e-12)
    values, vectors = np.linalg.eigh(aligned)
    logs = np.einsum('...ik,...k,...jk->...ij', vectors, np.log(np.maximum(values, 1e-12)), vectors)
    row, col = np.triu_indices(count)
    features = logs[..., row, col]
    features[..., row != col] *= np.sqrt(2)
    return features.reshape(len(covs), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['PhysioNet', 'BNCI2014_001'])
    args = parser.parse_args()
    protocol = {'condition': 'Offline transductive adaptation, separate from strict no-target-fit evaluation. Unlabeled target epochs are used to estimate each held-out participant covariance reference. No target labels fit weights or select candidates.', 'selection': 'Same people-isolated folds. Four fixed candidates: 9/22-channel per-band covariance whitening + log covariance + shrinkage LDA/RBF SVM(C=1). Choose by validation subject-macro balanced accuracy only.', 'signal_processing': 'Trace-normalized covariance + fixed 5% shrinkage. Each person and frequency band centered with mean covariance inverse square root. All available epochs for that person estimate their reference; predictions therefore depend on batch composition.', 'warning': 'Cannot directly call this zero-calibration single-trial or streaming accuracy. Previously inspected public benchmark, not fresh blind evaluation.', 'reference': 'Inspired by Euclidean alignment: He and Wu, https://arxiv.org/abs/1808.05464 . Here whitening is applied to per-band regularized covariances; not an exact paper reproduction.'}
    save_json(ROOT / 'reports/alignment_protocol.json', protocol)
    outputs = []
    with threadpool_limits(limits=2):
        for dataset in args.datasets:
            data = prepare(dataset)
            covs = api.covariance_bank(data['X'])
            # No label argument is passed to the alignment transform.
            features = {count: alignment_features(covs, data['subject'], count) for count in [9, 22]}
            folds = json.loads((api.EXTERNAL / 'data/prepared' / dataset / 'subject_folds.json').read_text(encoding='utf-8'))
            selected = np.full(len(data['y']), -1, dtype=int)
            records = []
            for fold in folds:
                train = np.isin(data['subject'], fold['train_subjects']) & (data['artifact'] != 1)
                val = np.isin(data['subject'], fold['validation_subjects']) & (data['artifact'] != 1)
                test = np.isin(data['subject'], fold['test_subjects'])
                candidates, scores = {}, {}
                for count in [9, 22]:
                    scaler = StandardScaler().fit(features[count][train])
                    train_f = scaler.transform(features[count][train])
                    val_f = scaler.transform(features[count][val])
                    for name in ['lda', 'svm']:
                        classifier = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto') if name == 'lda' else SVC(C=1, class_weight='balanced', kernel='rbf')
                        classifier.fit(train_f, data['y'][train])
                        candidate = f'aligned_{name}_{count}'
                        candidates[candidate] = {'kind': 'aligned_logcov', 'classifier': classifier, 'scaler': scaler, 'feature_channels': count}
                        scores[candidate] = api.old.subject_score(data['y'][val], classifier.predict(val_f), data['subject'][val])
                winner = max(scores, key=scores.get)
                model = metadata(candidates[winner], dataset, fold, winner)
                model['adaptation_condition'] = protocol['condition']
                joblib.dump(model, ROOT / 'models' / f'{dataset}_aligned_fold{fold["fold"]:02d}.joblib', compress=3)
                selected[test] = model['classifier'].predict(model['scaler'].transform(features[model['feature_channels']][test]))
                record = dict(fold, selected=winner, validation_subject_macro_balanced_accuracies=scores, test_accuracy=float(np.mean(selected[test] == data['y'][test])))
                records.append(record)
                print(f'{dataset} batch-aligned fold {fold["fold"]+1}/{len(folds)}: {winner}, test {record["test_accuracy"]:.2%}', flush=True)
            assert np.isin(selected, [0, 1]).all()
            summary, _ = api.old.summarize(data, selected)
            outputs.append({'dataset': dataset, 'summary': summary, 'folds': records})
            write_csv(ROOT / 'reports' / f'{dataset}_aligned_predictions.csv', [{'trial_id': str(uid), 'subject': int(s), 'true_label': int(y), 'predicted_label': int(p)} for uid, s, y, p in zip(data['trial_id'], data['subject'], data['y'], selected)])
            save_json(ROOT / 'reports' / f'{dataset}_aligned_results.json', outputs[-1])
            print(f'{dataset}: offline unlabeled-batch adaptation {summary["accuracy"]:.2%}', flush=True)
    save_json(ROOT / 'reports/alignment_results.json', {'protocol': protocol, 'datasets': outputs})
    lines = ['# 离线无标签批量对齐实验', '', '**这是一个单独的评估条件：使用了测试人员的未标注 EEG 波形估计对齐参数，没有使用他们的左右手标签训练或选参数。因此不能当作完全不接触目标数据的单试次泛化成绩。**', '', '| 数据集 | 无标签批量对齐准确率 |', '|---|---:|']
    for result in outputs:
        lines.append(f'| {result["dataset"]} | {result["summary"]["accuracy"]:.2%} |')
    lines += ['', '方法：逐人、逐频带估计平均协方差，以逆平方根做空间白化，再计算矩阵对数特征，使用收缩 LDA 或 RBF SVM 分类。九/二十二通道与分类器只由验证人员选定。采用与主实验完全相同的试次、人员隔离和伪迹规则。', '', '选择此模式是因为当前需求是完整公开文件的离线实验。它需要同一个人的一批试次，不能用一个窗口估计可靠的个人参考；接口至少要求每人五次试验。改变输入批次可能改变预测。报告中的成绩采用完整测试人员批次，不应推断为少数窗口或实时准确率。', '', '双击 run_aligned_demo.cmd 演示第一折。Python 调用 model_api.predict_aligned(X, subject_ids, model_path)，其中 X 为 (试次数,22,1000)，subject_ids 与试次一一对应。标签 y 不传入预测函数。', '', '灵感来源：He & Wu, Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach, https://arxiv.org/abs/1808.05464 。这里是逐频带正则化协方差白化的变体，并非原论文完整复现。', '', '本数据以前已经用于测试，当前结果仍是重复基准比较，不是新的盲测。', '']
    (ROOT / 'reports/离线批量对齐报告.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

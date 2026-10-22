"""Audit downloaded files, subject isolation, saved models and reported scores."""
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

import test_external_eeg as e


def read_predictions(name):
    with open(e.ROOT / 'reports' / f'{name}_predictions.csv', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main():
    manifest = json.loads((e.ROOT / 'data/sources.json').read_text(encoding='utf-8'))
    assert not manifest['failures']
    assert len(manifest['files']) == 345
    for item in manifest['files']:
        path = e.ROOT / item['relative_file']
        assert path.stat().st_size == item['bytes']
        with open(path, 'rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == item['sha256']
    report = json.loads((e.ROOT / 'reports/results.json').read_text(encoding='utf-8'))
    assert len(report['experiments']) == 6
    data = {name: dict(np.load(e.ROOT / 'data/prepared' / name / 'all_epochs.npz', allow_pickle=False)) for name in e.DATASETS}
    with threadpool_limits(limits=2):
        covs = {name: e.compute_covs(d['X']) for name, d in data.items()}
        for result in report['experiments']:
            name = result['experiment']
            target = result.get('test_dataset') or ('PhysioNet' if name.startswith('PhysioNet') else 'BNCI2014_001')
            d = data[target]
            assert d['X'].shape[1:] == (9, 1000)
            assert len(set(d['trial_id'])) == len(d['y'])
            assert list(d['channels']) == e.CHANNELS
            rows = read_predictions(name)
            assert np.array_equal(d['trial_id'], [r['trial_id'] for r in rows])
            truth = np.array([int(r['true_label']) for r in rows])
            predicted = np.array([int(r['predicted_label']) for r in rows])
            assert np.array_equal(truth, d['y'])
            assert np.isclose(accuracy_score(truth, predicted), result['accuracy'])
            recomputed, _ = e.summarize(d, predicted)
            assert np.isclose(recomputed['subject_macro_accuracy'], result['subject_macro_accuracy'])
            if name.endswith('frozen_old'):
                fresh = e.base.predict_eeg(d['X'][:, [3, 4, 5]], e.ROOT / 'models/frozen_bnci004.joblib')
            elif name.endswith('unseen_subjects'):
                folds = json.loads((e.ROOT / 'data/prepared' / target / 'subject_folds.json').read_text(encoding='utf-8'))
                fresh = np.full(len(truth), -1)
                held_people = []
                for fold in folds:
                    groups = [set(fold[k+'_subjects']) for k in ['train', 'validation', 'test']]
                    assert all(not (groups[i] & groups[j]) for i in range(3) for j in range(i))
                    assert set.union(*groups) == set(d['subject'])
                    held_people.extend(fold['test_subjects'])
                    model_path = e.ROOT / 'models' / f'{target}_unseen_subject_fold{fold["fold"]:02d}.joblib'
                    model = joblib.load(model_path)
                    assert model['split'] == fold
                    mask = np.isin(d['subject'], fold['test_subjects'])
                    assert np.all(fresh[mask] == -1)
                    fresh[mask] = model['classifier'].predict(e.base.features(covs[target][mask], model['spatial_filters']))
                    index = np.flatnonzero(mask)[0]
                    single = e.predict_saved(d['X'][index], model_path)
                    assert single[0] == fresh[index]
                assert len(held_people) == len(set(held_people)) == len(set(d['subject']))
                split_dir = e.ROOT / 'data/prepared' / target / 'first_fold_splits'
                subsets = [dict(np.load(split_dir / f'{s}.npz', allow_pickle=False)) for s in ['train', 'validation', 'test']]
                people = [set(s['subject']) for s in subsets]
                assert all(not (people[i] & people[j]) for i in range(3) for j in range(i))
                assert sum(len(s['y']) for s in subsets) == len(d['y'])
            else:
                model = joblib.load(e.ROOT / 'models' / f'{name}.joblib')
                assert model['source_dataset'] != target
                fresh = model['classifier'].predict(e.base.features(covs[target], model['spatial_filters']))
            assert np.array_equal(fresh, predicted)
            print(f'{name}: saved-model predictions and reported metrics verified', flush=True)
    output = {'passed': True, 'original_files_hash_verified': len(manifest['files']), 'experiments_verified': len(report['experiments']), 'checks': ['raw file sizes and SHA-256 match download manifest', 'correct trial identities and labels in each prediction table', 'all reported pooled and macro accuracies recomputed', 'train/validation/test subjects disjoint in every fold', 'each eligible person independently tested exactly once', 'first-fold three-way NPZ partitions disjoint and complete', 'reloaded model predictions exactly match every saved prediction', 'single-epoch public prediction interface agrees with batch inference']}
    e.save_json(e.ROOT / 'reports/verification.json', output)
    print(output)


if __name__ == '__main__':
    main()

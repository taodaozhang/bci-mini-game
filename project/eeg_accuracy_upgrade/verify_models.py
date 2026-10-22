"""Verify identical benchmark trials, isolated people, selection rules and saved inference."""
import csv
import json
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
import model_api as api
from train_compare import save_json


def csv_rows(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    results = json.loads((api.ROOT / 'reports/results.json').read_text(encoding='utf-8'))
    aligned_results = json.loads((api.ROOT / 'reports/alignment_results.json').read_text(encoding='utf-8'))
    verified = []
    with threadpool_limits(limits=2):
        for dataset in results['datasets']:
            name = dataset['dataset']
            data = dict(np.load(api.ROOT / 'data' / f'{name}_22channels.npz', allow_pickle=False))
            previous = dict(np.load(api.EXTERNAL / 'data/prepared' / name / 'all_epochs.npz', allow_pickle=False))
            assert np.array_equal(data['trial_id'], previous['trial_id'])
            assert np.array_equal(data['y'], previous['y'])
            assert np.array_equal(data['artifact'], previous['artifact'])
            assert np.array_equal(data['subject'], previous['subject'])
            assert np.allclose(data['X'][:, api.NINE_INDICES], previous['X'], rtol=1e-6, atol=1e-5)
            rows = csv_rows(api.ROOT / 'reports' / f'{name}_all_predictions.csv')
            assert np.array_equal(data['trial_id'], [r['trial_id'] for r in rows])
            old_rows = csv_rows(api.EXTERNAL / 'reports' / f'{name}_unseen_subjects_predictions.csv')
            assert [int(r['baseline_9']) for r in rows] == [int(r['predicted_label']) for r in old_rows]
            for method, summary in dataset['algorithms'].items():
                pred = np.array([int(r[method]) for r in rows])
                fresh, _ = api.old.summarize(data, pred)
                assert fresh['accuracy'] == summary['accuracy']
                assert fresh['subject_macro_accuracy'] == summary['subject_macro_accuracy']
            fold_results = json.loads((api.ROOT / 'reports' / f'{name}_fold_results.json').read_text(encoding='utf-8'))
            all_test_people = []
            for fold in fold_results:
                groups = [set(fold[s+'_subjects']) for s in ['train', 'validation', 'test']]
                assert all(not(groups[i]&groups[j]) for i in range(3) for j in range(i))
                assert fold['selected'] == max(fold['validation_subject_macro_balanced_accuracies'], key=fold['validation_subject_macro_balanced_accuracies'].get)
                mask = np.isin(data['subject'], fold['test_subjects'])
                all_test_people.extend(fold['test_subjects'])
                for family, column in [('selected', 'selected'), ('neural', 'neural_selected')]:
                    path = api.ROOT / 'models' / f'{name}_{family}_fold{fold["fold"]:02d}.joblib'
                    model = joblib.load(path)
                    assert model['split']['test_subjects'] == fold['test_subjects']
                    if family == 'neural':
                        assert isinstance(model['classifier'], MLPClassifier)
                        assert model['classifier'].solver == 'adam'
                    pred = api.predict_eeg(data['X'][mask], path, channels=data['channels'])
                    expected = np.array([int(r[column]) for r in rows])[mask]
                    assert np.array_equal(pred, expected), f'{name} {family} fold {fold["fold"]}: prediction mismatch'
                print(f'{name} fold {fold["fold"]}: selected/neural reloaded predictions verified', flush=True)
            assert len(all_test_people) == len(set(all_test_people)) == len(set(data['subject']))
            aligned = next(r for r in aligned_results['datasets'] if r['dataset'] == name)
            aligned_rows = csv_rows(api.ROOT / 'reports' / f'{name}_aligned_predictions.csv')
            expected_aligned = np.array([int(r['predicted_label']) for r in aligned_rows])
            assert np.mean(expected_aligned == data['y']) == aligned['summary']['accuracy']
            for fold in aligned['folds']:
                assert fold['selected'] == max(fold['validation_subject_macro_balanced_accuracies'], key=fold['validation_subject_macro_balanced_accuracies'].get)
                mask = np.isin(data['subject'], fold['test_subjects'])
                path = api.ROOT / 'models' / f'{name}_aligned_fold{fold["fold"]:02d}.joblib'
                pred = api.predict_aligned(data['X'][mask], data['subject'][mask], path, channels=data['channels'])
                assert np.array_equal(pred, expected_aligned[mask]), f'{name} aligned fold {fold["fold"]}: mismatch'
            verified.append(name)
    output = {'passed': True, 'datasets': verified, 'checks': ['all original trial IDs/labels/flags/people preserved', 'nine common channels match old waveforms', 'old baseline predictions exactly reproduced', 'all reported method scores recomputed', 'all train/validation/test people disjoint', 'every eligible person tested exactly once', 'final choices equal validation-score winner', 'all saved selected models reproduce held-out predictions', 'all saved neural models are actual Adam-trained MLPs and reproduce predictions', 'all offline-aligned models reproduce predictions using held-out signals and subject IDs only']}
    save_json(api.ROOT / 'reports/verification.json', output)
    print(output)


if __name__ == '__main__':
    main()

"""Run a saved model on public held-out EEG. No training and no EEG hardware."""
import argparse
import csv
from pathlib import Path
import joblib
import numpy as np
from model_api import ROOT, CHANNELS, predict_eeg, predict_aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['PhysioNet', 'BNCI2014_001'], default='PhysioNet')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--neural', action='store_true', help='Explicitly use the backpropagation neural network')
    parser.add_argument('--aligned', action='store_true', help='Offline unlabeled-batch adaptation; requires participant IDs')
    parser.add_argument('--input', type=Path, help='Optional NPY data or NPZ with X and optional metadata')
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    if args.neural and args.aligned:
        parser.error('Choose --neural or --aligned, not both.')
    family = 'aligned' if args.aligned else ('neural' if args.neural else 'selected')
    path = args.model or ROOT / 'models' / f'{args.dataset}_{family}_fold{args.fold:02d}.joblib'
    model = joblib.load(path)
    if args.input:
        loaded = np.load(args.input, allow_pickle=False)
        data = dict(loaded) if isinstance(loaded, np.lib.npyio.NpzFile) else {'X': loaded}
        context = 'User-selected input file; any displayed label match is for this file only.'
    elif args.fold == 0:
        data = dict(np.load(ROOT / 'data' / args.dataset / 'first_fold/test.npz', allow_pickle=False))
        context = 'Public held-out test people from fixed fold 0; no retraining.'
    else:
        data = dict(np.load(ROOT / 'data' / f'{args.dataset}_22channels.npz', allow_pickle=False))
        mask = np.isin(data['subject'], model['split']['test_subjects'])
        data = {key: value[mask] if key in ['X', 'y', 'subject', 'session', 'trial_id', 'artifact'] else value for key, value in data.items()}
        context = f'Public held-out test people from fixed fold {args.fold}; no retraining.'
    X = np.asarray(data['X'])
    if X.ndim == 2:
        X = X[None]
    print(context, flush=True)
    print(f'Model: {path.name}; selected method: {model["selected_name"]}', flush=True)
    print(f'Input: {X.shape}, 250 Hz, microvolts. 0=LEFT, 1=RIGHT.', flush=True)
    options = dict(sampling_rate=int(data.get('fs', 250)), channels=list(data.get('channels', CHANNELS)), unit=str(data.get('unit', 'microvolt')))
    if model['kind'] == 'aligned_logcov':
        if 'subject' not in data:
            parser.error('Aligned models require an NPZ containing one subject ID per epoch.')
        print('OFFLINE BATCH ADAPTATION: using unlabeled EEG from each test person to estimate alignment.', flush=True)
        prediction = predict_aligned(X, data['subject'], path, **options)
    else:
        prediction = predict_eeg(X, path, **options)
    print(f'Predicted {len(prediction)} EEG trials.')
    if 'y' in data:
        print(f'Accuracy on THIS file/fold: {np.mean(prediction == data["y"]):.2%}')
        print('For the aggregate comparison across ALL held-out people, read reports/results.json.')
    names = {0: 'LEFT', 1: 'RIGHT'}
    rows = []
    ids = data.get('trial_id', np.array([f'epoch_{i+1}' for i in range(len(X))]))
    for i, label in enumerate(prediction):
        row = {'trial_id': str(ids[i]), 'predicted_label': int(label), 'command': 'TURN_LEFT' if label == 0 else 'TURN_RIGHT'}
        if 'y' in data:
            row['true_label'] = int(data['y'][i])
        rows.append(row)
        if i < 20:
            truth = f' (true {names[int(data["y"][i])]})' if 'y' in data else ''
            print(f'{ids[i]} -> {names[int(label)]} / {row["command"]}{truth}')
    destination = ROOT / 'reports' / f'latest_predictions_{args.dataset}_{family}.csv'
    with open(destination, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'All predictions saved to: {destination}')


if __name__ == '__main__':
    main()

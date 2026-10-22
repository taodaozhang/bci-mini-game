"""Load the saved model and replay independent test EEG; no retraining."""
from pathlib import Path
import numpy as np
from train_left_right_eeg import predict_eeg

root = Path(__file__).resolve().parent
if __name__ == '__main__':
    data = np.load(root / 'data' / 'splits' / 'test.npz', allow_pickle=False)
    predicted = predict_eeg(data['X'])
    names = {0: 'LEFT', 1: 'RIGHT'}
    print('Saved global model, independent session 5')
    print(f'Trials: {len(predicted)}; accuracy: {np.mean(predicted == data["y"]):.2%}')
    print('First 12 trials (true -> predicted):')
    for i in range(12):
        print(f'{data["trial_id"][i]}: {names[int(data["y"][i])]} -> {names[int(predicted[i])]}')

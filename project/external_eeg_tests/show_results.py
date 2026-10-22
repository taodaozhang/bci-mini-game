"""Display saved test results without downloading or training anything."""
import json
from pathlib import Path

if __name__ == '__main__':
    path = Path(__file__).resolve().parent / 'reports/results.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    print('Experiment | pooled accuracy | subject-macro accuracy | people | trials')
    for r in data['experiments']:
        print(f'{r["experiment"]} | {r["accuracy"]:.2%} | {r["subject_macro_accuracy"]:.2%} | {r["n_subjects"]} | {r["n"]}')
    print('\nFrozen-old tests include an electrode-reference mismatch.')
    print('Unseen-subject results use strictly separated train/validation/test people.')

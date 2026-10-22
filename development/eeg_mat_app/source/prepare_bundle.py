"""Assemble trusted existing models; does not train or modify models."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]


def main():
    assets=HERE/'assets'; models=assets/'models'; models.mkdir(parents=True,exist_ok=True)
    registry={'models':{},'heldout_folds':{},'legacy':None}
    folds=json.loads((PROJECT/'outputs/external_eeg_tests/data/prepared/BNCI2014_001/subject_folds.json').read_text(encoding='utf-8'))
    for fold in folds:
        number=f'{fold["fold"]:02d}'
        for person in fold['test_subjects']: registry['heldout_folds'][str(person)]=number
        for mode in ['selected','aligned','neural']:
            filename=f'BNCI2014_001_{mode}_fold{number}.joblib'
            source=PROJECT/'outputs/eeg_accuracy_upgrade/models'/filename
            shutil.copy2(source,models/filename)
            registry['models'][f'{mode}_{number}']={'filename':filename,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'test_subjects':fold['test_subjects']}
    source=PROJECT/'outputs/eeg_project/models/left_right_global.joblib'
    shutil.copy2(source,models/source.name)
    registry['legacy']={'filename':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (assets/'model_registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared',len(registry['models'])+1,'existing models;',sum(p.stat().st_size for p in models.glob('*.joblib')),'bytes')


if __name__=='__main__':main()

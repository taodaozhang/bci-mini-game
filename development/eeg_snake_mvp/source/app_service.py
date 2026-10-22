from pathlib import Path
from dataclasses import dataclass
import csv
import hashlib
import json
import sys
import time
from functools import lru_cache
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from eeg_inference import predict

VERSION = '1.0.0'


def resource_root():
    return Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent))


def program_root():
    return Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parents[1]


@lru_cache(maxsize=4)
def load_trusted_model(path_string, expected_hash):
    path = Path(path_string)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hash:
        raise ValueError('内置模型校验失败，请重新复制完整程序。')
    return joblib.load(path)


@dataclass
class Result:
    labels: np.ndarray
    model_id: str
    mode: str
    condition: str
    elapsed: float
    notes: list


def recognize(data, mode='aligned', progress=lambda text:None):
    start = time.perf_counter()
    registry = json.loads((resource_root()/'assets/model_registry.json').read_text(encoding='utf-8'))
    notes = list(data.notes)
    if len(data.channels)==3:
        entry = registry['legacy']
        condition = '原三通道全局模型；与原训练/验证人员有重合，不代表新受试者准确率。'
        mode = 'legacy'
    else:
        if mode not in ['selected','aligned','neural']:
            raise ValueError('未知模型模式。')
        if mode=='aligned' and not data.grouping_known:
            raise ValueError('此文件没有subject人员分组，不能做批量对齐。请选择普通模型，或在MAT中保存subject。')
        fold = registry['heldout_folds'].get(str(data.known_bnci_subject),'00')
        entry = registry['models'][f'{mode}_{fold}']
        condition = ('按文件名匹配留出受试者模型（该人未参加训练/选参）；已查看的公开数据，非全新盲测。' if data.known_bnci_subject else '人员来源未确认；使用固定第0折模型，不能将一致率称为独立泛化成绩。')
        if mode=='aligned':
            condition += '离线批量对齐使用本文件同一人的全部未标注左右手波形；批次改变可能改变判断。'
    progress('正在校验并加载内置模型…')
    model = load_trusted_model(str(resource_root()/'assets/models'/entry['filename']),entry['sha256'])
    if data.known_bnci_subject and len(data.channels)==22 and data.known_bnci_subject not in model['split']['test_subjects']:
        raise ValueError('受试者与内置模型登记不匹配。')
    progress('正在提取脑电特征并判断左右手…')
    with threadpool_limits(limits=2):
        labels = np.asarray(predict(model,data.X,data.subjects if mode=='aligned' else None),dtype=int)
    if labels.shape!=(len(data.X),) or not np.isin(labels,[0,1]).all():
        raise ValueError('模型输出格式异常。')
    notes.append(condition)
    return Result(labels,entry['filename'].removesuffix('.joblib'),mode,condition,time.perf_counter()-start,notes)


def summary(data,result):
    return {'version':VERSION,'source_file':data.source,'format':data.format,'model_id':result.model_id,'mode':result.mode,'condition':result.condition,'n_trials':len(result.labels),'left_count':int(sum(result.labels==0)),'right_count':int(sum(result.labels==1)),'file_agreement':None if data.y is None else float(np.mean(result.labels==data.y)),'inference_seconds':result.elapsed,'channels':data.channels,'fs':250,'unit':'microvolt','notes':result.notes}


def export_csv(path,data,result):
    path = Path(path)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['trial_id','subject','predicted_label','prediction','command','true_label','correct','artifact','model_id','mode','condition'])
        for i,p in enumerate(result.labels):
            w.writerow([data.trial_ids[i],data.subjects[i],int(p),'左手' if p==0 else '右手','TURN_LEFT' if p==0 else 'TURN_RIGHT','' if data.y is None else int(data.y[i]),'' if data.y is None else int(p==data.y[i]),int(data.artifact[i]),result.model_id,result.mode,result.condition])


def diagnose(path,output,mode='aligned'):
    from mat_reader import load_eeg
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        data = load_eeg(path)
        result = recognize(data,mode)
        report = dict(summary(data,result),ok=True,labels=result.labels.tolist(),trial_ids=data.trial_ids.tolist())
        export_csv(output.with_suffix('.csv'),data,result)
    except Exception as exc:
        import traceback
        report = {'ok':False,'error':str(exc),'traceback':traceback.format_exc()}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report['ok'] else 1

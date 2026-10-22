"""Explicit, non-overwriting exports with enough information for replay."""
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import uuid
from pathlib import Path
from app_service import resource_root,summary,export_csv


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def export_run(parent,data,result,session,speed,input_hash):
    folder=Path(parent)/('EEG_Snake_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')+'_'+session.run_id[:8]+'_'+uuid.uuid4().hex[:4])
    folder.mkdir(parents=False,exist_ok=False)
    info=summary(data,result)
    model_file=resource_root()/'assets/models'/(result.model_id+'.joblib')
    config={'version':'1.0.0','run_id':session.run_id,'source_file':data.source,'input_sha256':input_hash,
            'model_id':result.model_id,'model_sha256':file_hash(model_file),'mode':result.mode,
            'target_adaptation':result.mode=='aligned','condition':result.condition,
            'subject_batches':{str(s):int(sum(data.subjects==s)) for s in set(data.subjects)},
            'board_width':session.board.width,'board_height':session.board.height,'initial_length':1,'initial_heading':0,
            'seed':session.seed,'replay_speed':speed,'epoch_seconds':4,
            'prediction_labels':session.labels,'trial_ids':session.trial_ids,'subjects':session.subjects}
    info.update(run_id=session.run_id,state=session.state,end_reason=session.end_reason,score=session.board.score,
                snake_length=len(session.board.body),consumed=session.cursor,total=len(session.labels),seed=session.seed,
                note='游戏分数是苹果数量，不代表EEG分类准确率。')
    for name,obj in [('config.json',config),('run_summary.json',info)]:
        (folder/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    export_csv(folder/'predictions.csv',data,result)
    fields=['event_id','seq','trial_id','subject_id','predicted_label','command','execution','heading_before','heading_after','head_before','head_after','apple_before','apple_after','score','length','state']
    with (folder/'events.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(session.events)
    return folder

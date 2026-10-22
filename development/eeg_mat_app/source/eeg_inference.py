"""Inference-only implementation of the existing saved pipelines.

No model fitting, label access, or benchmark-dependent threshold selection.
Equivalence to the original APIs is checked by verify_app.py.
"""
import numpy as np
from scipy.signal import butter, sosfilt

CHANNELS22 = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
CHANNELS3 = ['C3','Cz','C4']
NINE = [CHANNELS22.index(c) for c in ['FC3','FCz','FC4','C3','Cz','C4','CP3','CPz','CP4']]
BANDS22 = [(8,12),(12,20),(20,30),(8,30)]
BANDS3 = [(4,8),(8,12),(12,16),(16,20),(20,24),(24,28),(28,32),(32,36),(36,40),(8,30)]


def covariances(X, bands, average_reference=False):
    X = np.asarray(X, dtype=np.float64)
    output = np.empty((len(X), len(bands), X.shape[1], X.shape[1]))
    for start in range(0, len(X), 128):
        batch = X[start:start+128].copy()
        if average_reference:
            batch -= batch.mean(axis=1, keepdims=True)
        for b, (lo, hi) in enumerate(bands):
            sos = butter(4, [lo,hi], btype='bandpass', fs=250, output='sos')
            z = sosfilt(sos,batch,axis=-1)[...,125:]
            z -= z.mean(axis=-1,keepdims=True)
            output[start:start+len(batch),b] = np.einsum('nct,ndt->ncd',z,z)/(z.shape[-1]-1)
    return output


def spatial_features(covs, filters):
    power = np.maximum(np.einsum('bkc,nbcd,bkd->nbk',filters,covs,filters),1e-15)
    return np.log(power/power.sum(axis=2,keepdims=True)).reshape(len(covs),-1)


def log_features(covs, count, subjects=None):
    if count == 9:
        covs = covs[:,:,NINE,:][:,:,:,NINE]
        p = np.eye(9)-np.ones((9,9))/9
        covs = np.einsum('ij,nbjk,kl->nbil',p,covs,p)
    trace = np.maximum(np.trace(covs,axis1=-2,axis2=-1),1e-12)
    covs = .95*covs/trace[...,None,None]+.05*np.eye(count)/count
    if subjects is not None:
        aligned = np.empty_like(covs)
        for person in np.unique(subjects):
            mask = subjects == person
            values,vectors = np.linalg.eigh(covs[mask].mean(axis=0))
            inverse = np.einsum('bik,bk,bjk->bij',vectors,1/np.sqrt(np.maximum(values,1e-12)),vectors)
            aligned[mask] = np.einsum('bij,nbjk,bkl->nbil',inverse,covs[mask],inverse)
        covs = aligned/np.maximum(np.trace(aligned,axis1=-2,axis2=-1)[...,None,None],1e-12)
    values,vectors = np.linalg.eigh(covs)
    logs = np.einsum('...ik,...k,...jk->...ij',vectors,np.log(np.maximum(values,1e-12)),vectors)
    row,col = np.triu_indices(count)
    features = logs[...,row,col]
    features[...,row!=col] *= np.sqrt(2)
    return features.reshape(len(covs),-1)


def predict(model, X, subjects=None):
    X = np.asarray(X,dtype=np.float64)
    if X.ndim != 3 or X.shape[1:] not in [(3,1000),(22,1000)] or not len(X) or not np.isfinite(X).all():
        raise ValueError('模型需要有限数值的 N×3×1000 或 N×22×1000 脑电数据。')
    if np.any(np.ptp(X,axis=-1)<1e-8):
        raise ValueError('检测到平直通道，无法识别。请检查波形和单位。')
    if X.shape[1] == 3:
        if model.get('channels') != CHANNELS3:
            raise ValueError('三通道数据与模型不匹配。')
        covs = covariances(X,model['bands'])
        return model['classifier'].predict(spatial_features(covs,model['spatial_filters']))
    kind = model['kind']
    if kind == 'vote':
        return (np.stack([predict(m,X,subjects) for m in model['members']]).sum(axis=0)>=2).astype(int)
    if kind == 'baseline':
        covs = covariances(X[:,NINE],BANDS3,True)
        legacy = model['legacy']
        return legacy['classifier'].predict(spatial_features(covs,legacy['spatial_filters']))
    if kind == 'aligned_logcov':
        if subjects is None or np.asarray(subjects).shape != (len(X),):
            raise ValueError('批量对齐需要每个试次的受试者分组 subject。')
        if min(np.unique(subjects,return_counts=True)[1]) < 5:
            raise ValueError('批量对齐每人至少需要5个试次，请改用普通模型。')
    features = log_features(covariances(X,BANDS22,True),model['feature_channels'],subjects if kind=='aligned_logcov' else None)
    return model['classifier'].predict(model['scaler'].transform(features))

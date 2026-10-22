"""Conservative MAT import: BNCI IV2a/IV2b and documented epoched arrays."""
from dataclasses import dataclass, field
from pathlib import Path
import itertools
import re
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from fractions import Fraction
from eeg_inference import CHANNELS22, CHANNELS3


class ImportProblem(ValueError):
    pass


class MetadataRequired(ImportProblem):
    def __init__(self, description):
        self.description = description
        super().__init__(description)


@dataclass
class EEGFile:
    X: np.ndarray
    channels: list
    trial_ids: np.ndarray
    subjects: np.ndarray
    y: np.ndarray | None
    artifact: np.ndarray
    format: str
    source: str
    notes: list = field(default_factory=list)
    known_bnci_subject: int | None = None
    grouping_known: bool = True
    original_fs: float = 250


def _read_hdf5(path):
    import h5py
    def read(obj, depth=0):
        if depth > 8:
            raise ImportProblem('MAT结构嵌套过深，请先导出为 X、channels、fs、unit 等普通变量。')
        if isinstance(obj,h5py.Group):
            return {k:read(v,depth+1) for k,v in obj.items() if not k.startswith('#')}
        value = obj[()]
        if obj.dtype.kind == 'O':
            flat = [read(obj.file[ref],depth+1) if ref else '' for ref in value.ravel()]
            return np.asarray(flat,dtype=object).reshape(value.shape).T
        matlab_class=obj.attrs.get('MATLAB_class','')
        if isinstance(matlab_class,bytes):matlab_class=matlab_class.decode('ascii',errors='replace')
        if matlab_class == 'char':
            value = np.atleast_2d(value).T
            rows = [''.join(chr(int(c)) for c in row if c) for row in value]
            return rows[0] if len(rows)==1 else np.array(rows)
        if np.iscomplexobj(value) or value.dtype.fields:
            raise ImportProblem('暂不支持复数或复合HDF5字段，请导出实数 X 数组。')
        return value.T if value.ndim > 1 else value
    with h5py.File(path,'r') as handle:
        return {k:read(v) for k,v in handle.items() if not k.startswith('#')}


def read_variables(path):
    path = Path(path)
    if path.suffix.lower() != '.mat':
        raise ImportProblem('请选择 .mat 文件。')
    if not path.is_file():
        raise ImportProblem('文件不存在，请重新选择。')
    if path.stat().st_size > 1024**3:
        raise ImportProblem('本版支持1GB以内的MAT文件，请先按会话或试次分文件。')
    try:
        import h5py
        if h5py.is_hdf5(path):
            return _read_hdf5(path)
        return {k:v for k,v in loadmat(path,struct_as_record=False,squeeze_me=False).items() if not k.startswith('__')}
    except ImportProblem:
        raise
    except Exception as exc:
        raise ImportProblem(f'无法读取MAT文件：{exc}\n请确认文件完整且没有加密。') from exc


def _field(obj,key,default=None):
    while isinstance(obj,np.ndarray) and obj.dtype.kind=='O' and obj.size==1:
        obj=obj.ravel()[0]
    return obj.get(key,default) if isinstance(obj,dict) else getattr(obj,key,default)


def _strings(value):
    if isinstance(value,str):
        return [value.strip()]
    a = np.asarray(value)
    if a.dtype.kind == 'O':
        return [s for item in a.ravel() for s in _strings(item)]
    if a.dtype.kind in 'US':
        if a.ndim==2 and all(len(str(c))==1 for c in a.ravel()):
            return [''.join(str(c) for c in row).strip() for row in a]
        return [str(v).strip() for v in a.ravel()]
    raise ImportProblem('通道名称或单位应当使用MATLAB字符串或cell字符串。')


def _scalar(value, name):
    a = np.asarray(value)
    if a.size != 1:
        raise ImportProblem(f'{name}应是单个数值。')
    result = float(a.ravel()[0])
    if not np.isfinite(result):
        raise ImportProblem(f'{name}不是有限数值。')
    return result


def _vector(value, count, name, broadcast=False):
    a = np.asarray(value).ravel()
    if broadcast and len(a)==1:
        a = np.repeat(a,count)
    if len(a)!=count:
        raise ImportProblem(f'{name}长度应为{count}，实际为{len(a)}。')
    return a


def _bnci(variables,path):
    if 'data' not in variables:
        return None
    runs = np.asarray(variables['data'],dtype=object).ravel()
    if not len(runs) or _field(runs[0],'trial') is None or _field(runs[0],'classes') is None:
        return None
    classes = _strings(_field(runs[0],'classes'))
    if classes[:2] != ['left hand','right hand']:
        raise ImportProblem('此 data 结构的前两类不是 left hand/right hand，不能自动套用BNCI左右手模型。')
    count = 22 if len(classes)==4 and classes[2:]==['feet','tongue'] else 3 if len(classes)==2 else 0
    if not count:
        raise ImportProblem('此MAT的类别结构不在本版BNCI格式范围内。')
    expected = 25 if count==22 else 6
    dataset = 'BNCI IV2a（22通道）' if count==22 else 'BNCI IV2b（3通道）'
    prefix = 'A' if count==22 else 'B'
    match = re.fullmatch(prefix+r'(0[1-9])([TE])',Path(path).stem,re.I)
    person = int(match.group(1)) if match else None
    Xs,ys,ids,flags,notes = [],[],[],[],[]
    skipped = 0
    offset = 500 if count==22 else 750
    for ri,run in enumerate(runs):
        if _scalar(_field(run,'fs'),'fs') != 250:
            raise ImportProblem('BNCI自动模式要求原始250Hz文件。')
        signal = np.asarray(_field(run,'X'))
        if signal.ndim!=2 or signal.shape[1]!=expected or not np.issubdtype(signal.dtype,np.number):
            raise ImportProblem(f'{dataset}应有{expected}列（末3列为EOG），实际{signal.shape}。')
        onsets = np.asarray(_field(run,'trial')).ravel()
        labels = np.asarray(_field(run,'y')).ravel()
        af = np.asarray(_field(run,'artifacts',np.full(len(onsets),-1))).ravel()
        if not (len(onsets)==len(labels)==len(af)):
            raise ImportProblem('trial、y、artifacts长度不一致，不能安全切分。')
        if not np.isin(labels,[1,2,3,4] if count==22 else [1,2]).all():
            raise ImportProblem('BNCI文件包含未知类别编码，不能套用固定映射。')
        for ti,(onset,label,flag) in enumerate(zip(onsets,labels,af)):
            if label not in [1,2]:
                skipped += 1
                continue
            if not np.isfinite(onset) or onset != int(onset) or onset<1:
                raise ImportProblem('trial必须是MATLAB从1开始的有效整数索引。')
            start = int(onset)-1+offset
            epoch = np.asarray(signal[start:start+1000,:count].T,dtype=np.float32)
            if epoch.shape!=(count,1000) or not np.isfinite(epoch).all():
                raise ImportProblem(f'第{ri+1}块第{ti+1}试次不足4秒或包含无效数值。')
            if np.any(np.ptp(epoch,axis=-1)<1e-8):
                raise ImportProblem(f'第{ri+1}块第{ti+1}试次有平直通道。')
            Xs.append(epoch); ys.append(int(label)-1); flags.append(int(flag))
            stem=f'{prefix}{person:02d}_{match.group(2).upper()}' if match else Path(path).stem
            ids.append(f'{stem}_R{ri:02d}_E{ti:03d}')
    if not Xs:
        raise ImportProblem('文件中没有可用的左右手试次。')
    notes.append(f'按原训练规则取试次起点后{offset/250:g}至{offset/250+4:g}秒；使用前{count}列EEG，排除末3列EOG。')
    if skipped:
        notes.append(f'按公开真值排除了{skipped}个脚/舌试次；本模型只做左右手二分类。')
    if count==3:
        notes.append('三通道使用原全局模型；T文件和E文件中的会话4与训练/验证重合，文件一致率不等于独立测试成绩。')
    if person is None:
        notes.append('文件名未包含已知受试者编号，按单人文件处理，无法确认与训练人员是否重合。')
    return EEGFile(np.stack(Xs),CHANNELS22 if count==22 else CHANNELS3,np.array(ids),np.repeat(person or 'unknown',len(Xs)),np.array(ys),np.array(flags),dataset,str(path),notes,person)


def load_eeg(path, options=None):
    variables = read_variables(path)
    known = _bnci(variables,path)
    if known is not None:
        return known
    options = options or {}
    key = next((k for k in ['X','eeg','EEG','data'] if k in variables and isinstance(variables[k],np.ndarray) and np.issubdtype(variables[k].dtype,np.number)),None)
    if key is None:
        raise ImportProblem('未找到数值变量X/eeg/EEG/data，也不是支持的BNCI文件。\n可用变量：'+', '.join(variables.keys())+'\n请参照使用说明中的通用MAT字段。')
    X = np.asarray(variables[key])
    if np.iscomplexobj(X) or X.ndim not in [2,3]:
        raise ImportProblem(f'{key}应为二维单试次或三维分段实数数组，实际{X.shape}。连续信号请先按4秒试次切分。')
    fs_value = next((variables[k] for k in ['fs','srate','sampling_rate'] if k in variables),options.get('fs'))
    unit_value = variables.get('unit',options.get('unit'))
    names_value = next((variables[k] for k in ['channels','chan_names','channel_names'] if k in variables),None)
    if fs_value is None or unit_value is None or (names_value is None and not options.get('confirm_channels')):
        raise MetadataRequired(f'通用MAT变量：{key}，形状：{X.shape}\n文件未完整记录采样率、单位或通道名。请根据文件实际情况确认，不能只为通过校验而修改数值。')
    fs = _scalar(fs_value,'采样率')
    if not 50<=fs<=4000 or abs(4*fs-round(4*fs))>1e-6:
        raise ImportProblem('采样率应为50至4000Hz且4秒对应整数采样点。')
    unit = _strings(unit_value)[0].replace('µ','u').replace('μ','u').lower()
    factor = {'uv':1,'microvolt':1,'microvolts':1,'mv':1000,'millivolt':1000,'v':1e6,'volt':1e6}.get(unit)
    if factor is None:
        raise ImportProblem('单位只支持microvolt/uV、mV或V。')
    samples = int(round(4*fs))
    names = _strings(names_value) if names_value is not None else None
    counts = [len(names)] if names is not None else [3,22]
    layout = options.get('layout','auto')
    if X.ndim==2:
        possibilities = [p for p in [(0,1),(1,0)] if X.shape[p[0]] in counts and X.shape[p[1]]==samples]
        if layout in ['CS','SC']:
            possibilities = [(layout.index('C'),layout.index('S'))]
        if len(possibilities)!=1:
            raise ImportProblem(f'无法唯一确定通道轴与4秒采样轴：{X.shape}。请选手动轴布局，单试次应为C×S或S×C。')
        X = X.transpose(possibilities[0])[None]
    else:
        possibilities = [p for p in itertools.permutations(range(3)) if X.shape[p[1]] in counts and X.shape[p[2]]==samples]
        if len(layout)==3 and set(layout)==set('NCS'):
            possibilities = [(layout.index('N'),layout.index('C'),layout.index('S'))]
        if len(possibilities)!=1:
            raise MetadataRequired(f'变量{key}形状{X.shape}有多个可能的试次/通道轴。请选择明确布局：N=试次数，C=通道，S=4秒采样点。')
        X = X.transpose(possibilities[0])
    if X.shape[1] not in [3,22] or X.shape[-1]!=samples or not len(X):
        raise ImportProblem(f'布局后应为N×3/22×{samples}，实际{X.shape}。')
    if X.nbytes > 768*1024**2:
        raise ImportProblem('分段数组过大，请先拆分文件。')
    expected_names = CHANNELS22 if X.shape[1]==22 else CHANNELS3
    if names is None:
        names = expected_names
    lookup = [c.strip().upper() for c in names]
    if len(lookup)!=len(set(lookup)) or set(lookup)!=set(c.upper() for c in expected_names):
        raise ImportProblem('通道名必须完整且不重复：'+', '.join(expected_names))
    X = X[:,[lookup.index(c.upper()) for c in expected_names]].astype(np.float64)*factor
    if not np.isfinite(X).all():
        raise ImportProblem('X含NaN或无穷大，请先处理无效值。')
    notes = ['通用MAT：已按元数据重排通道、转换为微伏；每段必须对应一个左右手想象试次。']
    if fs!=250:
        ratio = Fraction(250/fs).limit_denominator(10000)
        X = resample_poly(X,ratio.numerator,ratio.denominator,axis=-1)
        if X.shape[-1]!=1000:
            raise ImportProblem('重采样后不为1000点，请核对采样率。')
        notes.append(f'已将实际{fs:g}Hz的4秒波形重采样至250Hz。')
    if np.any(np.ptp(X,axis=-1)<1e-8):
        raise ImportProblem('存在平直通道，无法识别。')
    n = len(X)
    known_group = 'subject' in variables
    subject = _vector(variables.get('subject',np.array(['unknown'])),n,'subject',True)
    if not np.issubdtype(subject.dtype,np.number):
        subject = np.array(_strings(subject))
    if len(subject)!=n or any(str(s).strip() in ['', 'nan', 'None'] for s in subject):
        raise ImportProblem('subject需要有效的每试次人员编号。')
    if np.issubdtype(subject.dtype,np.number) and not np.isfinite(subject).all():
        raise ImportProblem('subject包含无效数值。')
    ids = np.array(_strings(variables['trial_id'])) if 'trial_id' in variables else np.array([f'trial_{i+1:04d}' for i in range(n)])
    if len(ids)!=n or len(set(ids))!=n:
        raise ImportProblem('trial_id必须与试次数一致且唯一。')
    y = None
    if 'y' in variables:
        candidate = _vector(variables['y'],n,'y')
        if np.isin(candidate,[0,1]).all():
            y = candidate.astype(int)
        else:
            notes.append('通用MAT的y不是约定的0=左/1=右，未计算一致率。波形仍可预测。')
    flags = _vector(variables.get('artifact',np.array([-1])),n,'artifact',True)
    if not known_group:
        notes.append('缺少subject，只能使用普通或神经网络模型；不能确认训练人员重合情况。')
    return EEGFile(X.astype(np.float32),expected_names,ids,subject,y,flags,'通用分段MAT',str(path),notes,None,known_group,fs)

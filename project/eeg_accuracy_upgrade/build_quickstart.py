"""Build an honest summary of ordinary prediction and separate batch adaptation."""
import json
from pathlib import Path
import numpy as np
from model_api import ROOT


def main():
    ordinary = json.loads((ROOT / 'reports/results.json').read_text(encoding='utf-8'))
    aligned = json.loads((ROOT / 'reports/alignment_results.json').read_text(encoding='utf-8'))
    extra = {r['dataset']: r['summary'] for r in aligned['datasets']}
    lines = ['# 快速开始：公开 EEG 离线预测', '', '不用连接脑电设备，全部使用已经下载的公开文件。模型已经训练完成，双击下面的入口只做预测。', '', '## 先运行哪一个', '', '- **run_aligned_demo.cmd**：适合这次完整公开文件的离线批量实验。按人使用一批无标签 EEG 做协方差对齐，再输出 LEFT/RIGHT。这是包含无标签目标数据自适应的条件。', '- **run_demo.cmd**：普通模型预测，不估计目标人员的批量对齐参数。', '- **run_neural_demo.cmd**：明确加载已经通过反向传播+Adam训练的神经网络，便于与传统方法比较。', '', '三个入口默认都使用事先固定的 PhysioNet 第 0 折测试人员。不是按测试分数挑最好的一折。窗口显示的单折成绩，可能与下表所有折的汇总成绩不同。终端输出 TURN_LEFT/RIGHT 只是离线指令文本，并未连接贪吃蛇。', '', '## 这次实际得到的准确率', '', '| 数据 | 原九通道跨人基线 | 新普通跨人方案 | 反向传播神经网络 | 离线无标签批量对齐 |', '|---|---:|---:|---:|---:|']
    for dataset in ordinary['datasets']:
        name, a = dataset['dataset'], dataset['algorithms']
        lines.append(f'| {name} | {a["baseline_9"]["accuracy"]:.2%} | {a["selected"]["accuracy"]:.2%} | {a["neural_selected"]["accuracy"]:.2%} | {extra[name]["accuracy"]:.2%} |')
    lines += ['', '**普通模式没有在两套数据上都提高。** 神经网络的训练拟合很高，但跨人泛化未超过原基线。小幅上升也不应表述为已经证明稳定提升。', '', '**最后一列使用了待测人员的无标签波形来估计个人参考，因此不等同于前面三列不拟合目标人员参数的评估条件。** 它没有使用目标标签训练或选参数；但需要一批数据，同一个人的输入批次改变时预测可能改变，不能把完整批次成绩称为单窗口或实时成绩。', '', '两种实验均沿用原来的试次和人员隔离。由于这些公开测试数据以前已经被查看过，本次属于重复基准比较，不是全新盲测。', '', '## 文件在哪里', '', '- `models`：保存的模型。名称中的 selected 是由验证人员选出的普通方案；neural 是神经网络；aligned 是离线批量对齐方案。fold00、fold01 等对应不同的人员隔离测试折。', '- `data/PhysioNet/first_fold/test.npz`：默认演示数据；同目录还有 train.npz 和 validation.npz。BCI IV 2a 目录结构相同。', '- `reports/latest_predictions_*.csv`：每次演示的全部左右手预测。', '- `reports/算法升级报告.md`：普通模式的算法比较、拟合差距和所有候选成绩。', '- `reports/离线批量对齐报告.md`：无标签批量对齐方法及限制。', '- `reports/verification.json`：保存模型、数据一致性及预测复核结果。', '', '## Python 中调用', '', '普通模式接口：`model_api.predict_eeg(X, model_path, channels=...)`。', '', '离线批量对齐接口：`model_api.predict_aligned(X, subject_ids, model_path, channels=...)`；subject_ids 是每段脑电所属人员的编号，不是左右手标签。当前工程保护要求每人至少五段，报告成绩使用完整人员批次；不保证五段即可达到报告准确率。', '', '两种模式都统一使用二十二通道、250 Hz、4 秒、微伏单位，X 形状为 `(试次数,22,1000)`。普通模式也支持单个 `(22,1000)` 窗口。顺序、完整调用示例与换电脑安装方法见 `使用说明.md`。预测演示只需本目录；从原始库重新准备数据或复现旧基线比较才需要同级 external_eeg_tests 项目。', '']
    (ROOT / '快速开始.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
    x = np.arange(len(ordinary['datasets']))
    series = [('baseline_9', 'Previous baseline', '#888e98'), ('selected', 'New ordinary method', '#3276ab'), ('neural_selected', 'Backprop MLP', '#36a08b'), ('aligned', 'Offline batch adaptation*', '#bc8745')]
    for j, (key, label, color) in enumerate(series):
        values = [(extra[r['dataset']]['accuracy'] if key == 'aligned' else r['algorithms'][key]['accuracy'])*100 for r in ordinary['datasets']]
        bars = ax.bar(x+(j-1.5)*.19, values, width=.19, label=label, color=color)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
    ax.set(xticks=x, xticklabels=[r['dataset'] for r in ordinary['datasets']], ylim=(0, 100), ylabel='Accuracy (%)', title='Public-data algorithm comparison')
    ax.axhline(50, color='gray', ls='--', lw=1)
    ax.legend(loc='lower right', fontsize=8)
    fig.supxlabel('*Adaptation uses unlabeled target EEG; a different evaluation condition.', fontsize=9)
    fig.savefig(ROOT / 'reports/overview.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

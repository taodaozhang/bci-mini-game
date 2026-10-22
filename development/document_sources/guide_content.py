from build_documents import P,H,B,C,T,I,BASE,ROOT,PROJECT

def guide():
    return [
    [H('1 先了解当前可以做什么'),
     P('本指南说明怎样加载已经训练好的左右手运动想象 EEG 模型，运行附带的公开测试文件，读取预测结果，并在 Python 程序中调用接口。整个过程不需要脑电设备。当前软件能够输出左右手标签及转向文本，尚未实现贪吃蛇窗口、自动回放控制器或实时采集。'),
     P('首次使用先看第3节；准备自己的离线数组看第4至6节；查看准确率与排错看第7和9节。第10节解释未来怎样接入游戏，其中的控制器属于待开发内容。'),
     H('当前实测结果'),
     T(['数据集','原九通道基线','新普通方案','反向传播 MLP','离线批量对齐'],[
       ['PhysioNet','61.83%','63.09%','61.28%','68.92%'],
       ['BCI IV 2a','66.98%','64.43%','63.81%','71.49%']], [3.2,3.3,3.3,3.6,3.6]),
     P('表中前三类不使用测试人员的整批波形估计个人对齐参数；最后一类使用测试人员的无标签波形进行批量对齐，不使用其左右手标签拟合或选参。因此四列不属于完全相同的评估条件。普通方案没有在两套数据上都提高，神经网络也没有超过原基线。'),
     P('这些公开测试数据此前已经查看过成绩，本次属于重复基准比较，不是全新盲测。表中百分比是所有外层测试折汇总后的试次准确率，不能视为未知用户、实时系统或游戏控制准确率。'),
     B('左手标签为0，输出 LEFT 与 TURN_LEFT。右手标签为1，输出 RIGHT 与 TURN_RIGHT。','selected 表示该折由验证人员成绩选定的方案，不代表事后挑测试成绩最高的模型。','neural 是真实通过反向传播和 Adam 训练的多层感知机；aligned 是离线无标签批量对齐方案。')],
    [H('2 找到数据和模型'),
     P('桌面总项目位置为 '+ROOT+'。本指南主要使用其中的新升级项目：'),
     C(PROJECT),
     P('下表的相对路径均以这个升级项目为起点。复制项目时应保留完整目录结构，不能只复制一个 joblib 文件就直接对原始波形调用分类器。'),
     T(['相对位置','保存内容'],[
       ['models','42个模型文件。PhysioNet有5折，IV 2a有9折，每折各有 selected、neural、aligned 三类。'],
       ['data/PhysioNet/first_fold','PhysioNet第0折的 train.npz、validation.npz、test.npz。'],
       ['data/BNCI2014_001/first_fold','IV 2a第0折的三组 NPZ。'],
       ['data/*_22channels.npz','该数据集全部合格试次的二十二通道数组和人员信息。'],
       ['model_api.py','普通预测和离线批量对齐的统一入口。'],
       ['legacy_helpers','旧算法兼容代码。预测时需要随项目保留。'],
       ['reports','基准成绩、逐试次预测、学习曲线、协议与核验结果。']], [6.4,10.6]),
     P('原始文件没有重复下载到升级目录。PhysioNet EDF 文件在总项目的 external_eeg_tests\\data\\raw\\physionet；IV 2a MAT 在 external_eeg_tests\\data\\raw\\bnci001。原始来源、许可、大小和 SHA-256 记录在总项目的 external_eeg_tests\\data\\sources.json。'),
     P('总项目根部的 train_left_right_eeg.py 与 models 文件夹属于最早的三通道 BNCI 2014-004 实验；external_eeg_tests 的模型属于九通道实验。它们和本指南的二十二通道接口不能直接混用。最早的76.25%属于已见受试者跨会话测试，与本指南的跨人测试条件不同。')],
    [H('3 不写代码运行一次演示'),
     B('打开升级项目文件夹，先双击 run_demo.cmd，确认普通预测可以加载。','需要完整文件的离线批量对齐时，双击 run_aligned_demo.cmd。该模式会读取同一人员的一批无标签波形。','需要明确试用神经网络时，双击 run_neural_demo.cmd。不要把神经网络这个名称等同于更高准确率。'),
     P('启动文件会优先使用项目自己的 .venv；若没有，则使用本机 Documents\\Codex 任务中的已安装环境。当前电脑无需再次安装依赖。删除或移动原任务目录后，应按第8节重建项目内环境。'),
     H('默认会看到什么'),
     C('''Model: PhysioNet_aligned_fold00.joblib
Input: (1005, 22, 1000), 250 Hz, microvolts.
OFFLINE BATCH ADAPTATION ...
Predicted 1005 EEG trials.
Accuracy on THIS file/fold: 66.17%
S001_R04_E001 -> RIGHT / TURN_RIGHT (true RIGHT)'''),
     P('以上是当前版本默认批量对齐演示的预期要点，文件路径显示可能因位置而不同。窗口保留至按键退出。它不重新训练，也不控制游戏。默认使用预先固定的 PhysioNet 第0折，不是最高分折。'),
     H('在 PowerShell 中切换数据'),
     C('''$project = 'C:\\Users\\Lenovo\\Desktop\\脑机小游戏\\eeg_accuracy_upgrade'
Set-Location -LiteralPath $project
.\\run_aligned_demo.cmd --dataset BNCI2014_001
.\\run_neural_demo.cmd --dataset BNCI2014_001
.\\run_demo.cmd --dataset PhysioNet --fold 1'''),
     P('PhysioNet 的折编号为0至4，IV 2a 为0至8。普通、神经网络、对齐模式均可通过相应启动文件切换数据和折号。每次运行都会更新相应的 latest_predictions CSV；如需保留某次结果，先另存副本。')],
    [H('4 准备合格的离线输入'),
     P('新接口接收已经分好试次的波形，不直接把任意 MAT、EDF 或普通表格当作输入。NPY 只保存 X；NPZ 可以同时保存 X、通道、采样率、人员和标签。推荐使用 NPZ，避免依赖默认假设。'),
     T(['字段','要求','用途'],[
       ['X','浮点数组 N×22×1000','N为试次数。普通API也接受22×1000；批量对齐应提供三维批次。'],
       ['channels','22个不重复通道名称','推荐始终传入，接口按名称重排。名称大小写应与规定一致。'],
       ['fs','标量250','采样率，单位Hz。1000点对应4秒。'],
       ['unit','字符串 microvolt','数值单位为微伏，不能把伏单位数组仅改标签后使用。'],
       ['subject','长度N的人员编号','批量对齐必需；同一个人用同一个编号。不能传左右手标签y。'],
       ['y','可选，长度N，0或1','仅用于预测完成后的结果核对，不传入推理函数。'],
       ['trial_id','可选，长度N','每次试验的唯一标识，便于追踪与命令去重。'],
       ['artifact','可选，长度N','1为原标记有伪迹，0为无伪迹，-1为无这类标记。']], [2.5,5.2,9.3]),
     H('二十二通道顺序'),
     C('''Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2,
C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz'''),
     P('即使某一折内部选用九个特征通道，新接口也统一接收以上二十二通道。channels未传入时程序只能假定顺序正确，不能从数值自动猜出电极名称。X中不能有 NaN 或 Inf，N应大于0，y、subject和trial_id必须与X逐行对应。'),
     P('当前接口会按模型类型完成相应的参考处理、滤波、特征提取和标准化。不要在输入前再次执行同样的数字滤波或特征标准化。也不要把1000个采样点改成其他长度来加速回放；回放速度与输入窗口长度是两件事。')],
    [H('5 在 Python 中使用普通模型'),
     P('将以下代码保存为升级项目目录中的 example_predict.py，再用该项目环境运行。示例只取公开测试文件的前10个试次；普通模型不需要用整批目标数据估计个人参考。'),
     C('''from pathlib import Path
import numpy as np
from model_api import predict_eeg

root = Path(__file__).resolve().parent
file = root / 'data/PhysioNet/first_fold/test.npz'
model = root / 'models/PhysioNet_selected_fold00.joblib'
data = np.load(file, allow_pickle=False)

labels = predict_eeg(
    data['X'][:10],
    model,
    sampling_rate=int(data['fs']),
    channels=data['channels'],
    unit=str(data['unit']),
)
for trial_id, label in zip(data['trial_id'][:10], labels):
    command = 'TURN_LEFT' if int(label) == 0 else 'TURN_RIGHT'
    print(trial_id, int(label), command)'''),
     P('返回值是长度N的标签数组。接口没有返回已校准的概率或置信度；不能把SVM分数或神经网络内部数值自行命名为准确概率。对单次窗口可传 data["X"][0]，返回值仍是长度1的数组。'),
     H('使用已有启动器预测指定文件'),
     C('''.\\run_demo.cmd --input ".\\data\\PhysioNet\\first_fold\\test.npz" `
  --model ".\\models\\PhysioNet_selected_fold00.joblib"'''),
     P('上例需在升级项目目录的 PowerShell 中执行，反引号表示续行。自定义模型和数据应配套。只提供 --model 而不提供 --input 时，默认输入仍由 --dataset 决定；不要误用 PhysioNet 默认文件来声称得到 IV 2a 的测试成绩。'),
     P('joblib 文件包含类型不同的模型字典，有些是投票组合，有些是兼容旧基线。应调用 model_api，而不是直接把原始X送入字典内的 classifier.predict。仅加载自己生成或可信来源的模型文件。')],
    [H('6 使用离线无标签批量对齐'),
     P('对齐模式先按人员、按频带计算这一批波形的平均协方差，再做空间白化和分类。它只要求人员分组，不需要该人员的左右手标签；但确实使用了待测波形。因此它适用于当前完整公开文件的离线实验。'),
     C('''from pathlib import Path
import numpy as np
from model_api import predict_aligned

root = Path(__file__).resolve().parent
data = np.load(
    root / 'data/PhysioNet/first_fold/test.npz',
    allow_pickle=False,
)
labels = predict_aligned(
    data['X'],
    data['subject'],
    root / 'models/PhysioNet_aligned_fold00.joblib',
    channels=data['channels'],
    sampling_rate=int(data['fs']),
    unit=str(data['unit']),
)
# 标签只在预测完成后核对，不传入上面的函数。
accuracy = np.mean(labels == data['y'])
print('当前文件一致率', f'{accuracy:.2%}')'''),
     B('subject_ids必须是真实的受试者分组，绝不能用y代替。把左手和右手分别当成两个人会改变处理条件，破坏结果解释。','当前代码要求每个人至少有5个试次，这是工程保护条件，不代表5个试次就能达到报告准确率。','删除、增加或重新分配同一个人的试次，会改变平均协方差，进而可能改变每条预测。复现实验应使用原来完整的人员批次。','不要把aligned模型交给predict_eeg。也不要把普通模型交给predict_aligned。两者的预处理条件不同。'),
     P('如果仅有一个窗口，使用普通模型。若把不同人员混成一个分组，即使程序可以运行，仍不能解释为正确的个人对齐。当前模型没有静息类别，也没有证明为实时或免校准控制器。')],
    [H('7 读取结果和理解准确率'),
     T(['默认 PhysioNet 第0折','当前单折一致率','全部5折汇总'],[
       ['普通验证选择方案','58.51%','63.09%'],
       ['反向传播神经网络','55.62%','61.28%'],
       ['离线无标签批量对齐','66.17%','68.92%']], [7,5,5]),
     P('默认文件有1005个试次，汇总评估共有4917个试次，两个分母不同。IV 2a 汇总有2592个试次。不能把同一模型在它的训练人员上的分数称为独立测试，也不能把最初三通道模型76.25%的跨会话结果与这里的跨人分数直接排名。'),
     H('预测 CSV 的字段'),
     T(['列名','含义'],[
       ['trial_id','保持输入顺序的试次标识；输入缺失时由程序生成编号。'],
       ['predicted_label','0为左手，1为右手。'],
       ['command','TURN_LEFT或TURN_RIGHT。这里只是文本，没有模拟键盘。'],
       ['true_label','仅当输入文件含y时出现，用于预测后核对。']], [5,12]),
     P('文件位于reports，命名为 latest_predictions_数据集_selected.csv、latest_predictions_数据集_neural.csv 或 latest_predictions_数据集_aligned.csv。每种组合再次运行会覆盖对应latest文件。建议将需要留存的一次运行复制为带日期或运行编号的新文件。'),
     P('完整比较见 reports/results.json；批量对齐见 reports/alignment_results.json；各折人员与选择规则见 *_fold_results.json；保存和重载核验见 verification.json。训练准确率与学习曲线在报告中单独记录，不作为泛化成绩。'),
     P('命令行对自选文件只计算其标签一致率，不会完整判断该文件是否构成新盲测。报告时至少同时写清数据集、人员范围、折号、样本数、输入通道、是否批量对齐、模型文件和代码版本。')],
    [H('8 环境迁移和重新训练'),
     P('当前启动器中的后备解释器路径指向原Codex任务环境，所以直接复制cmd到另一台电脑可能失败。推荐把升级项目完整复制后，在项目内建立自己的.venv。普通预测和附带文件演示不需要读取同级原始库；重建二十二通道数据和复现原九通道基线比较才需要同级external_eeg_tests。'),
     H('另一台电脑的环境准备'),
     C('''# 先安装可用的 Python 3.12，再进入升级项目目录
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe demo_predict.py --aligned'''),
     P('若没有py命令，应使用所安装Python的完整python.exe路径创建环境。本次保存模型使用Python 3.12、NumPy 2.5.4、SciPy 1.18.1、scikit-learn 1.9.1、joblib 1.6.0。优先按requirements.txt恢复版本；不承诺所有跨版本模型都兼容。CPU即可运行当前MLP，不需要额外安装PyTorch。'),
     H('重新比较算法'),
     C('''.\\.venv\\Scripts\\python.exe train_compare.py
.\\.venv\\Scripts\\python.exe train_alignment.py
.\\.venv\\Scripts\\python.exe verify_models.py
.\\.venv\\Scripts\\python.exe build_quickstart.py'''),
     P('以上操作会更新模型、学习曲线和报告，耗时明显长于演示。重跑前应复制要保留的models和reports；不应为了提高训练分数而把测试人员并回训练或用测试标签选择停止轮次。准备数据代码会优先复用已有二十二通道NPZ。'),
     P('建议使用64位Python和足够的内存，迁移前先查看整个项目的实际大小，并预留复制和临时文件空间。CPU、内存和耗时建议是运行规划，不是本项目已经完成的跨电脑性能测试。只运行demo不会重训练，但会更新latest预测CSV。')],
    [H('9 常见问题和检查方法'),
     T(['现象','应检查和处理'],[
       ['窗口提示找不到Python','原任务环境可能被移动。按第8节在项目内建立.venv，启动器会优先使用它。'],
       ['找不到模型或NPZ','确认进入升级目录；核对dataset和fold；PhysioNet只支持0至4，IV 2a只支持0至8。'],
       ['提示输入形状错误','新接口要求N×22×1000。旧三路或九路数据不能直接使用。'],
       ['提示单位或采样率错误','核对fs=250及unit=microvolt；应真正转换波形单位和采样率，不能仅修改文字。'],
       ['提示通道名称错误','检查22个名称是否完整、不重复、大小写一致；不要传EOG代替EEG。'],
       ['提示人员不足五个试次','对齐模式按subject计数。使用完整人员批次，或者改用普通模型。'],
       ['aligned缺少subject','NPZ中应提供与每条X对应的人员编号；NPY通常不包含这些信息。'],
       ['看见版本兼容警告','恢复requirements中的环境，不要在不兼容环境中解释性能变化。'],
       ['分数与汇总不一致','核对是第0折还是所有折；是否换了批次；是否使用了相同模型、人员和预处理。'],
       ['输出了方向但蛇不动','目前没有游戏程序接收文本；这属于待开发的MVP功能。']], [5.1,11.9]),
     H('最小排错记录'),
     P('记录完整报错、运行命令、数据集、折号、模型文件名、X.shape、channels、fs、unit，以及对齐模式每个人的试次数。不要只记录最后一行准确率。'),
     C('''# 在同一环境中查看版本；此命令不重新训练
.\\.venv\\Scripts\\python.exe -m pip show numpy scipy scikit-learn joblib
# 查看命令行可选参数
.\\.venv\\Scripts\\python.exe demo_predict.py --help''')],
    [H('10 怎样交给贪吃蛇'),
     P('这一节是接入设计说明，当前项目尚未实现。终端print只显示文字，游戏必须主动读取标签或事件后更新自己的方向变量。对于公开文件回放，推荐同一个桌面程序先完成预测，再按试次顺序消费指令；不需要模拟操作系统键盘或搭建服务器。'),
     P('拟议数据流为 文件校验 → 模型预测 → 保存带试次编号的指令 → 回放调度 → 贪吃蛇状态更新 → 导出运行记录。预测和游戏使用的标签不得由true_label替代。'),
     T(['当前朝向','左手 0 左转后','右手 1 右转后'],[
       ['上','左','右'],['右','上','下'],['下','右','左'],['左','下','上']], [5.4,5.8,5.8]),
     C('''# 游戏控制核心示意，不是当前已实现的游戏
DIRECTIONS = [(0,-1), (1,0), (0,1), (-1,0)]
heading = 1  # 0上 1右 2下 3左

def consume_new_prediction(label, heading):
    if label not in (0, 1):
        raise ValueError('Only 0 or 1 is allowed')
    heading = (heading + (-1 if label == 0 else 1)) % 4
    dx, dy = DIRECTIONS[heading]
    return heading, dx, dy'''),
     P('拟议MVP采用逐试次步进：一条新的试次只转向一次，再前进一格；没有新试次就不推进。应按event_id去重，不能每帧重复执行同一条LEFT。暂停不消费新试次；重启应恢复固定种子和回放索引，才能复现。'),
     P('公开左右手想象任务没有要求参与者把蛇带向食物，因此游戏分数不能评价模型分类准确率，也不能说明当前操作者正在自主脑控。不要用自动导航把错误预测替换后仍宣称是模型输出。Project Proposal与MVP PRD另行规定待开发范围、事件协议和验收标准。')],
    [H('11 运行前后核对和证据来源'),
     B('运行前确认当前使用的是eeg_accuracy_upgrade，文件和模型均来自可信项目。','确认数据集、模型类型和折号一致；自定义数据采用明确的通道、单位和人员信息。','普通模式可以逐窗口预测；对齐模式使用完整人员批次且不传入y作为分组。','运行后核对预测条数与输入条数一致，并将需要保留的latest CSV另存。','写报告时将“单折”“全折汇总”“普通”“批量对齐”分开标注。','需要开发游戏时，以PRD中的待开发要求为准，不能把本指南的控制示意当成现成功能。'),
     H('项目内部证据'),
     P('实测数值来自升级项目reports/results.json和alignment_results.json；默认第0折数值来自相应预测CSV与模型重载演示。verification.json记录了原试次、标签、九通道波形、人员隔离、旧基线复现、验证选择规则和42个保存模型的预测一致性核查。'),
     P('模型文件名里的fold00是代码编号0，与自然语言“第1轮”同义。PhysioNet采用5轮，IV 2a采用9轮。每个合格人员在外层测试中只贡献一次预测；训练和选参只使用其他人员。完整训练细节、候选算法和局限可从同目录的训练协议与学习曲线追溯。'),
     H('公开来源'),
     P('以下链接用于追溯数据与方法，检索日期为2026年10月20日。数据数量和本项目成绩以本地审计文件为准，论文成绩不能替代本项目实测。'),
     B('PhysioNet EEG Motor Movement Imagery Dataset  https://physionet.org/content/eegmmidb/1.0.0/','BNCI数据目录及BCI IV 2a说明  https://bnci-horizon-2020.eu/database/data-sets','无标签对齐方法背景  https://arxiv.org/abs/1808.05464','MLPClassifier与Adam实现  https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html'),
     P('本项目的对齐实现是逐频带正则化协方差白化变体，并未声称完整复现论文。原始数据许可与引用要求保存在external_eeg_tests/data/sources.json。原始文件保持不变，处理后的划分用于本地实验。若对外分享数据，应遵守对应数据源的许可和署名要求。')]
    ]

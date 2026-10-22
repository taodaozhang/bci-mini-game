# 脑机小游戏 / BCI Mini Game

本仓库保存脑电左右手运动想象项目的训练代码、模型使用程序、MAT 数据查看工具、项目文档，以及脑电贪吃蛇 MVP 1.0。

账号：`taodaozhang`。默认作为私有项目归档。快照日期：2026-10-22。

## 下载并运行

在本仓库 **Releases** 中下载完整项目快照的全部分卷及 `parts.json`，再下载仓库里的 `restore_project.ps1`。

在这些文件所在目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\restore_project.ps1
```

脚本先校验每个分卷的 SHA-256，拼成 ZIP 并校验，再解压到一个新目录，不覆盖已有项目。程序入口为：

```text
脑机小游戏/
  脑电贪吃蛇_MVP_1.0/脑电贪吃蛇_MVP_1.0.exe
  脑电左右手识别器.exe
```

双击 EXE 即可使用，不需要先安装 Python。贪吃蛇中拖入 MAT 或点击“加载示例”，识别完成后点击“开始游戏”。

## 仓库内容

| 位置 | 内容 |
| --- | --- |
| `project/` | 桌面“脑机小游戏”中的源码、说明、报告和项目文档，保留原相对路径 |
| `development/eeg_snake_mvp/` | 贪吃蛇窗口、游戏引擎、MAT 导入、模型推理、打包和测试源码 |
| `development/eeg_mat_app/` | 独立 MAT 左右手识别器源码和测试 |
| `development/document_sources/` | 模型使用指南、Project Proposal 和 MVP PRD 的生成源码 |
| `backup/file_manifest.json` | 桌面项目每个文件的路径、大小和 SHA-256，含备份版本及缓存 |
| `backup/file_manifest.csv` | 可在 Excel 中查看的同一份文件清单 |
| `backup/parts.json` | 完整快照的分卷顺序、大小和 SHA-256 |
| Releases | 完整目录快照，包括原始 MAT/EDF、预处理数据、模型、EXE、示例和已有备份 |

GitHub 普通 Git 文件上限为 100 MiB，因此数据、模型及 EXE 放在同一仓库的 Releases。每个分卷小于 2 GiB。完整性以清单及校验为准，不把只克隆源码称为已下载全部数据。

## 已完成的功能

- 左右手想象 EEG 数据准备、训练/验证/测试划分和多算法比较。
- 普通模型、反向传播 MLP、离线无标签批量对齐的保存和调用。
- 支持项目中的 BNCI 3 通道和 22 通道原始 MAT，以及符合字段约定的分段 MAT。
- MAT 文件拖放、按受试者选择、模型预测与公开真值对照。
- 贪吃蛇从一格开始，随机苹果，吃到增长；按模型输出左右转向，支持暂停、单步、重放、变速和运行导出。
- 模型使用指南、Project Proposal、MVP PRD。

## 结果解释

这是公开 EEG 文件的离线预测与游戏回放项目。每条预测转向一次并前进一格。公开任务不是按苹果位置采集，游戏分数不等于分类准确率，也不证明当前操作者在实时脑控。

离线批量对齐利用目标人员的未标注波形估计参考；不能将它称为单试次、免校准、完全不接触目标数据的泛化成绩。相同模型和输入通常产生相同预测，重新回放复用已计算的标签。

本次快照保留桌面目录中已有的更新版本和旧 EXE 备份。开发源码以归档时本地工作区文件为准；没有重新训练模型或替换现有程序。

## 数据和许可

数据来自 BNCI / BCI Competition IV 以及 PhysioNet EEG Motor Movement/Imagery：

- https://bnci-horizon-2020.eu/database/data-sets
- https://physionet.org/content/eegmmidb/1.0.0/

各数据集继续适用原始许可、署名和引用要求，详情见项目中的 `sources.json` 与说明文件。本仓库未为第三方数据或项目源码另行指定新的开源许可。

GitHub 存储方式依据：[大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[Release 文件限制](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。

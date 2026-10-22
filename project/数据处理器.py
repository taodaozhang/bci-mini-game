from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
import matplotlib.pyplot as plt

# 1. 选择数据集，只提取左右手运动想象
dataset = BNCI2014_001()
paradigm = LeftRightImagery()

# 2. 获取第一名受试者的数据
# 第一次运行需要联网下载
X, y, metadata = paradigm.get_data(
    dataset=dataset,
    subjects=[1]
)

# 3. 查看数据结构
print("数据形状：", X.shape)
print("前十次试验的标签：", y[:10])
print("试验信息：")
print(metadata.head())

# 4. 画出第一次试验、第一个通道的脑电
# 自动找到一次左手试验和一次右手试验
left_index = next(i for i, label in enumerate(y)
                  if label == "left_hand")
right_index = next(i for i, label in enumerate(y)
                   if label == "right_hand")

# 两张子图，共用坐标范围，方便比较
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

axes[0].plot(X[left_index, 0, :], color="blue")
axes[0].set_title("Left hand — channel 1")
axes[0].set_ylabel("EEG (microvolts)")

axes[1].plot(X[right_index, 0, :], color="orange")
axes[1].set_title("Right hand — channel 1")
axes[1].set_ylabel("EEG (microvolts)")
axes[1].set_xlabel("Sample index")

plt.tight_layout()
plt.show()
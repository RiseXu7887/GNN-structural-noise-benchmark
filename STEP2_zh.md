# 第二步：完整对照、门控消融与 GPU

将本包中的 run_suite.py、check_gpu.py、STEP2_zh.md 放进原来的 cora_step1 文件夹，与 train.py 同级。
不需要替换原来的 train.py 或重装 CPU 环境。

## 先得到完整实验结果

```powershell
.\.venv\Scripts\python.exe run_suite.py --device cpu
```

总共 15 次训练，每次 200 轮：MLP、GCN、GAT、Gated、Fixed × 种子 42/43/44。
测试时删边比例 0/10/30/50%，每个非零比例重复 3 个扰动图。
训练与验证始终使用原始图；不在每个删边图上重新训练。
每次完整执行建立独立 experiment_cpu_时间 文件夹，不覆盖旧结果。

模型：
- MLP：只看节点特征，不用图结构。
- GCN：图卷积基线。
- GAT：图注意力基线（注意力并不保证抵抗随机删边）。
- gated：学习第一层自身分支与图卷积分支的混合比例。
- fixed：相同两个分支，混合比例固定 0.5，作为门控消融。

输出：aggregate.csv 总表；seed_means.csv 每个训练种子的均值；all_results.csv 所有评估；RESULTS.md 可读结果表。
每次训练的日志、权重、学习曲线数据和配置保存在同一实验目录。
每个种子内先平均 3 个删边样本，再对 3 个训练种子求平均与样本标准差。
不要把 9 个扰动评估称为 9 次独立训练。标准差不是置信区间。

## Windows RTX 5090：独立 GPU 环境

在原来的 cora_step1 目录逐条执行，任何一条报错就暂停并提供输出：

```powershell
nvidia-smi
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-gpu\Scripts\python.exe check_gpu.py
```

PyTorch 2.7 开始支持 Blackwell；官方为 Linux/Windows 提供 2.7.1 CUDA 12.8 安装指令。
这里选固定版本便于复现，并非宣称这是最新版本。
check_gpu.py 不只检查 is_available，还会对全部 5 个模型运行 CUDA 前向、反向和参数更新。
这个小型随机图只用来验证 GPU 功能，绝不计入研究结果。
本包 GPU 路径未在助手环境实测（没有 GPU），需要你本机验证。

出现 GPU READY 后：

```powershell
.\.venv-gpu\Scripts\python.exe train.py --model gcn --epochs 200 --seed 42 --device cuda --out runs_gpu
```

再运行整套 GPU 实验：

```powershell
.\.venv-gpu\Scripts\python.exe run_suite.py --device cuda
```

GPU 结果会单独放到 experiment_cuda_时间 目录。
CPU 与 GPU 版本不同、数值计算也可能不同，研究表格使用同一设备/环境下的整套结果，不拼接各自最好成绩。
Cora 很小，GPU 不一定更快。此阶段目标是验证 GPU 可用，不是测试显卡极限。
日志里的秒数不适合用于严谨的 CPU/GPU 性能比较（预热、同步和版本未受控）。

## 解读限制

固定混合与可学习门控都新增了自身分支；门控版本又多了门控参数。
门控与普通 GCN 的全部差异不能归因于门控。
GAT 隐藏层是 8 头×8 维，GCN 是 16 维，参数量未对齐；两者还使用不同默认学习率/dropout。
因此结果是给定配置下的探索性比较，不是架构本身的严格排名。
仅有一个数据集/公开划分、3 个种子；不能据此宣称普遍优越或统计显著。
如果要从结果再改超参数，必须用验证集决定，不依据测试表选配置。

来源：
https://pytorch.org/blog/pytorch-2-7/
https://pytorch.org/get-started/previous-versions/

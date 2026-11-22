# CoraFull：扩大真实数据规模 + 可选训练标签数量实验

本包不含训练结果。助手只检查了真实数据加载、划分、统计汇总和编码修复；所有正式训练由你本机执行。

## 解压与兼容

将 ZIP 内所有文件解压到原 cora_step1 目录，让 train_large.py、run_large.py 与原来的 .venv-gpu 文件夹同级。
允许覆盖 train.py、run_suite.py：它们修复中文路径导致的 UnicodeEncodeError；不改变旧模型公式或旧训练参数。
不删除原来的 runs 或 experiment_cuda_* 结果。
新脚本使用 Python 3.12 / torch 2.7.1+cu128 / PyG 2.6.1 的现有 GPU 环境，无需重新安装依赖。
原始 CoraFull 数据随包附带；第一次加载会生成 processed 缓存，不需要再次联网下载。

截图中的旧批量实验并未全部完成：单独 GCN 成功，批量 MLP 已保存结果后在打印中文输出路径时崩溃。
修复后若还要完成原始 Cora 实验：

```powershell
.\.venv-gpu\Scripts\python.exe run_suite.py --device cuda
```

这会按原规则建立一次新实验，不自动接续旧目录。

## 方案 A：先跑更大的真实数据（建议先执行这一条）

```powershell
.\.venv-gpu\Scripts\python.exe run_large.py --dataset CoraFull --device cuda --out .\large_results
```

默认 5 模型 × 3 个训练种子 = 15 次训练，每次 200 轮。
5 模型是 MLP、GCN、GAT、Gated 和 Fixed。
训练种子为 42/43/44。训练与验证用原始图；测试删边比例为 0/10/30/50%。
每个非零删边比例使用 5 个扰动图；同一扰动图在所有模型、训练种子与标签预算之间共享。
共有 240 条评估记录，但只有 15 次独立初始化训练，不能称作 240 次独立实验。

## 方案 B：检验“更多带标签训练节点”是否改变结果（可选）

在你愿意跑更久时执行此命令，单独输出，避免与 A 的配置混合：

```powershell
.\.venv-gpu\Scripts\python.exe run_large.py --dataset CoraFull --device cuda --train-fractions 0.25 0.5 1.0 --out .\label_budget_results
```

这会运行 45 次训练，其中包括重新运行 100% 训练池条件。
不必同时运行 A/B，两者选择一个即可；先做 A、以后再做 B 也可以，但会有重复计算。
训练样本数依次为 2938、5907、11849。由于按类别向下取整，前两个值不恰好等于总数的 25%/50%。
训练集合逐级包含；验证集固定 3932 个节点，测试集固定 4012 个节点。
图始终是同一张完整 CoraFull 图：19,793 节点，126,842 条有向存储边，8,710 维特征，70 类。
完整训练池/验证/测试使用一次按类别分层约 60%/20%/20% 划分，split-seed=2026。
某一预算未使用的训练池节点不参与损失，也不会转入验证/测试；但其特征和边仍参与消息传递，属于传导式学习。
每个类别在三个集合中均有节点。CoraFull 最小类别有 15 个节点。

B 研究的是监督标签数量，不是图节点数量。如果要控制“图本身变大”的效果，还需要另行设计子图实验，不能把 B 描述成图规模实验。

## 持续运行与中断续跑

每 20 轮显示一次进度，并将 UTF-8 日志写入每次训练的 console.log。
中断后重新执行完全相同的命令即可：已完整保存的训练显示 SKIP，未完成的那一次从第 1 轮重跑。
不是从最后一个 epoch 恢复；只在整次训练边界恢复。
参数、设备、版本或训练代码发生改变时，要求使用新的 --out 目录，防止混入不同实验。
如遇 CUDA out of memory、驱动报错或文件权限错误，停止并提供日志；脚本不会默默退回 CPU。
不存在预设完成时间或准确率保证。可以用前几个任务的实际 seconds 判断剩余耗时。
Ctrl+C 会终止当前子进程；不要同时让多个大型套件争抢同一 GPU。

## 输出

完成后出现 ALL FINISHED。
- RESULTS.md：每个训练标签预算下的准确率表。
- aggregate.csv：准确率、Macro-F1、百分点下降、相对下降的均值和标准差。
- seed_means.csv：先在单个训练种子内平均扰动后的结果。
- all_results.csv：所有测试评估记录。
- gating_paired_differences.csv：按相同训练种子配对的 Gated - Fixed 准确率差，单位百分点。
- suite_spec.json：整套实验设置、版本与训练代码哈希。
- 每次训练子目录：模型权重、学习曲线数据、具体划分索引、类别计数、版本和显存信息。

先在每个训练种子内平均扰动，再对训练种子求均值与样本标准差。
这里的标准差仅描述固定划分、固定扰动集合下的初始化变化；不是置信区间，也不代表跨数据划分的波动。
默认仅 3 个初始化种子，不声称统计显著。不能根据测试集得分挑选某个种子或重新调参。

## 与原 Cora 实验的区别

CoraFull 不是将 Cora 简单复制几遍：节点更多，但类别由 7 变成 70，特征由 1433 变成 8710，图内容也不同。
因此不能直接比较两张图的准确率并归因于样本量。
新实验全部采用 64 维隐藏输出；GAT 为 8 头 × 8 维。原 GCN/MLP 是 16 维。
新实验采用按类别分层 60/20/20；原实验采用 Cora public split。
这些是两套不同协议，旧的 82% 不是新实验的成绩目标或直接基准。

若想在相同新协议下也跑小 Cora，可执行：

```powershell
.\.venv-gpu\Scripts\python.exe run_large.py --dataset Cora --device cuda --out .\cora_same_protocol
```

即使协议统一，跨 Cora/CoraFull 的差异仍混合了类别、特征与图结构变化，不是纯粹的样本量因果实验。

学习率/dropout 沿用原模型默认值：GAT lr=.005/dropout=.6，其余 lr=.01/dropout=.5；Adam weight_decay=5e-4。
Gated/Fixed 的自身分支与门控公式保持不变；Gated 比 Fixed 多门控参数，不能声称严格等参数。
检查点仅按干净验证图上的 accuracy 选择（同分取最早），训练 200 轮后再测测试集。
Macro-F1 对每个类别等权，用来补充 70 类可能不均衡时的整体准确率；它不参与选模。
GPU 采用 FP32，关闭 TF32；未强制所有 CUDA 算子严格确定性，重复运行可能有小幅差异。

## 数据来源与验证状态

官方文档：https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.CoraFull.html
原始数据：https://github.com/abojchevski/graph2gauss/blob/master/data/cora.npz
随包 data/CoraFull/cora/raw/cora.npz 是 PyG CoraFull 数据类使用的同源文件，SHA256 见 DATA_PROVENANCE.json。
原始数据约 12.9 MB，解压处理后的稠密特征约 690 MB（十进制）；训练还需要额外内存与显存。
不能用压缩文件大小估计训练峰值显存。实际峰值显存记录在每次训练 summary.json。

已验证：语法、真实数据加载、掩码互斥/覆盖、每类划分、训练子集嵌套、验证/测试固定、cp1252 中文输出回归、汇总的合成数值测试。
未执行：任何新版正式训练、Windows 实机回归、GPU 显存/耗时实测。合成测试数据不包含在包内，不冒充研究结果。

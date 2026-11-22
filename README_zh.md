# 第一步：Cora 环境、数据与第一次训练

项目：图结构噪声下的 GNN 鲁棒性：GCN 与 GAT 的对比研究。
此包先完成真实 Cora 上的 GCN 训练，附下一步需要的 MLP/GAT、门控与固定混合消融入口。

## Windows 操作

1. 完整解压 ZIP。进入包含 train.py、setup_windows.cmd 的文件夹。
2. 在资源管理器地址栏输入 powershell 并回车。
3. 运行 `python --version`；也可以运行 `py -0p` 查看已安装版本。
4. 输入：

```powershell
.\setup_windows.cmd
.\.venv\Scripts\python.exe train.py --model gcn --epochs 200 --seed 42 --device cpu
```

安装脚本优先寻找 Python 3.12，其次 3.11/3.10，并创建当前目录的 .venv。
无需 conda，无需激活脚本，也无需改变 PowerShell 执行策略。已验证的环境是 Linux/Python 3.12；Windows 安装脚本未在 Windows 实机执行。
如果只有 Python 3.13 或更新版本，请先停止并提供版本输出，我们选择兼容环境；不要覆盖已有 Python。
此安装使用 PyTorch 2.5.1 CPU 版本，不会使用 RTX 5090。GPU 配置是后续独立步骤，不能直接拿此 CPU 环境宣称验证了 5090。
依赖安装需要联网。数据原始文件在包内时无需重新下载 Cora；首次会由 PyG 处理。

## 运行时看到什么

- Nodes=2708，Directed-edge entries=10556，Features=1433，Classes=7。
- Train/Val/Test=140/500/1000。
- 每 20 轮输出一次训练损失、验证准确率；第 1 轮和最后一轮也会输出。
- FINISHED 表示训练、验证选模、测试评估、结果保存均已完成。
- loss 是训练误差；val_acc 是验证节点的分类准确率；clean_test_acc 是原始图上的测试准确率。
- 2708 个节点里，只有 140 个标签参与训练损失；其余节点的特征和图结构参与消息传递。这是传导式节点分类设定，不是全体标签训练。

输出到 runs/模型_种子_时间/：
- history.csv：每轮训练损失和验证准确率。
- summary.json：参数、版本、运行时间、最佳验证轮次、最终测试准确率。
- best_model.pt：验证准确率最高的模型权重（同分取首次）。
- results.csv：测试结果；默认只测无删边情况。

verified_runs/ 是助手在 Linux CPU 上的真实验证运行，与你本机产生的 runs/ 分开。
smoke_runs/ 若存在，只用于检查其他模型路径能训练，短训练分数不用于论文比较。

## 研究约定

Cora 使用 PyG Planetoid 的 public 划分；NormalizeFeatures 对每个节点特征归一化。
只在原始图训练与验证；不使用测试分数选择轮次。训练 200 轮后恢复最佳验证模型，再测试。
GCN 的 cached=False，删边后会重新计算邻接归一化。
数据中的 10556 是有向存储条目数，对应 5278 对无向边。
删边按无向边对采样，两个方向一起删，数量四舍五入；卷积层添加的自环保留。
相同训练种子/重复编号/删边率在不同模型间使用相同扰动图；不同删边率的删除集合嵌套。

## 下一步：完整对照（第一步成功后再运行）

```powershell
foreach ($seed in 42,43,44) {
  foreach ($model in 'mlp','gcn','gat') {
    .\.venv\Scripts\python.exe train.py --model $model --seed $seed --epochs 200 --drop-rates 0 0.1 0.3 0.5 --repeats 3
    if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
  }
}
```

三种初始化种子，固定同一个公开数据划分。每个非零删边率测三个扰动样本。
同一训练模型的扰动重复不是独立训练重复；汇总时先在训练种子内平均，再跨训练种子统计。
MLP 不读取边，固定权重下各删边率输出应完全相同，是实现正确性的检查。
相对下降=(原始图准确率-删边图准确率)/原始图准确率；百分点下降=(原始-删边)*100。
下降可以为负，不应把偶然提升裁剪掉。

## 可选门控与消融

`--model gated`：第一层生成 h_graph=ReLU(GCN(x,E)) 与 h_self=ReLU(Linear(x))；
每个节点 g=sigmoid(Linear(concat(h_self,h_graph)))，得到 h=g*h_graph+(1-g)*h_self，再进入第二层 GCN。
GCN 分支本身也含自环，因此这里不是严格分离的“纯邻居 vs 纯自身”。
`--model fixed`：完全相同的两个分支和第二层，将 g 固定为 0.5，不学习门控。

```powershell
.\.venv\Scripts\python.exe train.py --model gated --epochs 200 --drop-rates 0 0.1 0.3 0.5
.\.venv\Scripts\python.exe train.py --model fixed --epochs 200 --drop-rates 0 0.1 0.3 0.5
```

门控是一个待检验的小扩展，不宣称新方法或保证提升。
gated vs fixed 用来检验自适应混合的作用；gated 比 fixed 多门控参数，并非严格等参数量对照。
与普通 GCN 比较还混入新增自身分支的影响，不能把全部差异归因于门控。
GAT 使用常见的 8 头×8 隐藏维配置；GCN/MLP 隐藏维为 16。不同参数量会记录，不能称为等容量比较。
这套默认超参数没有系统调优，单次运行不能证明谁更鲁棒。

## 数据与文档来源

- https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html
- https://github.com/kimiyoung/planetoid/tree/master/data
- https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html
- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
- https://pytorch.org/get-started/locally/

打包的 data/Cora/raw 为上述 Planetoid 原始文件；不包含 Python 虚拟环境。不同系统/依赖/设备下分数可能略有不同。

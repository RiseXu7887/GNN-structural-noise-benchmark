# CoraFull 修订版：冻结训练设置后重跑主对照

## 为什么修订

用户上传的 diagnostic_summary.json 包含真实 RTX 5090 验证诊断，4 次训练均为 seed=42、同一划分、2000 轮；未评估测试集。

| 模型 | weight_decay=0.0005 最佳验证准确率 | weight_decay=0.00005 最佳验证准确率 |
|---|---:|---:|
| GCN | 24.64% | 67.65% |
| Fixed | 11.93% | 63.76% |

在同一轮次预算下，降低正则化显著改善了这两个具体运行的验证表现。这里的“改善”是数值描述，不是统计显著性声明。
原先的 200 轮也不足：GCN 在原正则化下第 200 轮验证准确率约 16.12%，延长后的最佳值为 24.64%。
因此旧配置同时存在训练预算不足和不合适的正则化强度问题；不是 GPU 未工作。
不能把这 4 次单种子验证成绩作为最终测试成绩，也不能从中推断所有模型的最优超参数。

## 使用方法

把 run_large_v2.py 和 train_large_v2.py 放在原 cora_step1 文件夹里，与 train.py、large_data.py、data/ 和 .venv-gpu 同级。
不会覆盖旧脚本或实验结果，也不需要重新安装依赖。

```powershell
.\.venv-gpu\Scripts\python.exe run_large_v2.py --dataset CoraFull --device cuda --epochs 2000 --weight-decay 0.00005 --out .\large_results_v2
```

完成 5 模型 × 3 种子 = 15 次主对照，不先重跑 45 次标签预算实验。
各模型都使用相同 max_epochs=2000、weight_decay=0.00005；验证准确率选择最佳检查点，同分取首次。
保留原始模型结构、同一数据、split-seed=2026、64 维隐藏输出及原学习率/dropout。
测试删边率为 0/10/30/50%，每个非零率使用 5 个扰动样本。保持跨模型/种子相同扰动图。
训练与验证始终用干净图。测试评估在训练结束、恢复最佳验证权重之后才执行。
中断后执行同一命令，可跳过完整保存的训练；未完成训练从第一轮重跑。
版本、配置或训练源码改变后应指定新的 --out 目录，不能混合结果。

运行结束后，请提供 large_results_v2/RESULTS.md 与 large_results_v2/aggregate.csv。
如果有异常，附相应模型子目录的 console.log、summary.json。

## 如何使用已经完成的 45 次实验

保留旧目录 label_budget_results，标注为“初始配置：200 轮、weight_decay=0.0005”。
这批结果反映的是受到训练配置限制时的标签预算对照，不能用来证明“更多标签没有价值”。
也不能将旧配置下的 25%/50% 分数与修订配置下的 100% 分数放在同一学习曲线中。
若以后确需修订版标签预算实验，可在主对照检查完成后运行：

```powershell
.\.venv-gpu\Scripts\python.exe run_large_v2.py --dataset CoraFull --device cuda --epochs 2000 --weight-decay 0.00005 --train-fractions 0.25 0.5 1.0 --out .\label_budget_results_v2
```

该命令是可选项，当前先运行上面的 15 次主对照。

## 解读边界

这次使用了 GCN/Fixed 的一个初始化种子和固定验证集选择统一设置，没有对每种架构做同等预算的超参数搜索。
旧测试成绩此前已被查看；虽然本次配置选择直接依据验证诊断，整项研究仍应定位为探索性课程实验，不能声称严格盲测或普遍最优。
验证集上两种模型的提升不保证门控会胜出，也不保证 GAT 在此配置下充分收敛。新运行保留每轮验证曲线以检查这一点。
不要为得到预设排名继续根据测试成绩调参；最终如实报告新结果和局限。

## 包内文件

- run_large_v2.py / train_large_v2.py：新的本地训练入口。
- convergence_diagnostic.png / convergence_diagnostic.pdf：根据用户上传的验证诊断生成的曲线，只有验证成绩。
- diagnostic_comparison.csv：上述诊断的精确汇总。
- README_V2_zh.md：本说明。

助手仅执行语法/导入/命令行检查，并根据已有 JSON 生成图表；未执行修订版训练，没有提供虚构测试成绩。

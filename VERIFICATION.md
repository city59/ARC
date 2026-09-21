# ARC 独立版本验证

日期：2026-09-19。

模型类为 `modules.arc.ARC`。训练入口、测试和运行脚本仅导入本目录的模块，数据在本目录 `data/movie`、`data/music`、`data/book`，运行不需要相邻项目目录。

## 已执行检查

- Python 3.9 / PyTorch 2.3.0 CPU：19 项单元与集成测试通过。
- Python 3.9 / PyTorch 1.12.1：相同 19 项测试通过。
- 三套真实数据的独立 CPU 冒烟流程通过：上下文预训练、冻结上下文、推荐训练、全候选物品评估、保存 checkpoint。
- 核对三套数据生成的配置、训练日志、结果和 checkpoint，`model_name` 均为 `ARC`。

```powershell
Set-Location 'F:\1-submitted\TMKGRec (ICASSP)\arc'
& 'D:\python3.9\python.exe' -m unittest discover -s tests -v
& 'D:\Anaconda3\python.exe' -m unittest discover -s tests -v
& 'D:\python3.9\python.exe' run_all.py --smoke-test --device cpu --output-dir runs/arc_verification
```

结果位于 `runs/arc_verification/<dataset>/smoke/`。冒烟配置为 16 维、每阶段 3 次更新、评估每套数据 2 名用户；这用于验证重命名后的独立项目可以运行，不代表收敛性能。

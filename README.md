# TFM-ICGN：四种数字图像相关方法参考实现

本项目提供 TFM-ICGN、FFT-ICGN、SIFT-ICGN 和 RG-ICGN 四种二维数字图像相关（Digital Image Correlation，DIC）方法的参考代码，用于估计参考图像与变形图像之间的面内位移。

TFM-ICGN 使用基于 Transformer 的 TFM-DIC 网络生成位移初值，再通过二阶逆组合 Gauss–Newton（IC-GN）进行亚像素精化。其他三种方法分别采用 FFT 互相关、SIFT 特征匹配和可靠性引导传播提供初始化。

当前版本包含方法代码、最终模型权重和 tension009 实验的 REF/TAR 示例图，不包含训练脚本、训练数据集及论文原始实验结果文件。

## 1. 方法与入口

| 方法 | 初始化方式 | 入口文件 |
| --- | --- | --- |
| TFM-ICGN | TFM-DIC 网络预测 | `source/run_gmdic_icgn.py` |
| FFT-ICGN | FFT 互相关 | `source/run_fftcc_icgn.py` |
| SIFT-ICGN | SIFT 匹配与局部仿射估计 | `source/run_sift_icgn.py` |
| RG-ICGN | 中心种子 FFT 初始化，按 ZNCC 优先级向四邻域传播 | `source/run_RG_icgn.py` |

四种入口均使用项目的二阶、12 参数 IC-GN 求解器。RG-ICGN 是本项目的可靠性引导实现，不等同于完整的官方 Ncorr 实现；每点最多尝试一次，失败点不传播，也不自动补种。

当前代码的网络模块和类名均为 `tfmdic`；入口及输出文件中保留的 `gmdic`、`GM`、`GMGN` 是历史命名。

## 2. 目录结构

```text
.
├── .gitignore
├── README.md
├── LICENSE
├── NOTICE
├── LICENSES/
│   ├── MPL-2.0.txt
│   └── BSD-3-Clause.txt
├── requirements.txt
├── model_config.json
├── data/
│   └── tension009/
│       ├── tensionREF.bmp
│       └── tensionTAR.bmp
├── checkpoints/
│   └── step_250000.pth
└── source/
│   ├── run_gmdic_icgn.py
│   ├── run_fftcc_icgn.py
│   ├── run_sift_icgn.py
│   ├── run_RG_icgn.py
│   ├── networks/
│   │   ├── tfmdic.py
│   │   └── ...
│   ├── icgn/
│   │   └── ...
│   └── utils/
│       └── utils.py
```

`networks/` 提供特征提取、Transformer 和相关匹配模块；`icgn/` 提供初始化、梯度计算、插值、IC-GN 及指标统计。

## 3. 环境依赖

本次语法、权重加载与局部求解检查使用以下本地环境；这不是所有平台的完整兼容性测试。

| 软件 | 版本 |
| --- | --- |
| Python | 3.10.16 |
| PyTorch | 2.5.1 |
| NumPy | 2.0.1 |
| SciPy | 1.15.2 |
| OpenCV | 4.10.0 |
| Matplotlib | 3.10.0 |

使用 Python 3.10，在项目根目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

requirements.txt 已包含 PyTorch 2.5.1。如需指定 CPU 或 CUDA 构建，请先安装适合运行平台的 PyTorch 2.5.1，再安装本文件中的其余依赖。入口会根据 CUDA 是否可用选择设备，但本次尚未验证完整的 CPU/GPU 图像推理流程。其余三种入口不依赖网络权重。

## 4. 准备输入

已附带 tension009 实验图像：`data/tension009/tensionREF.bmp` 和 `data/tension009/tensionTAR.bmp`，为原实验图像的原样副本。在项目根目录运行时，将所用入口的 `DATA_DIR` 设置为 `'data/tension009'` 即可使用。当前入口仍保留原路径，需手动修改。本示例不包含真值 CSV，因此指标模块将报告 IC-GN 修正量而非真值误差。

为每组图像创建独立目录，放入一幅参考图像和一幅变形图像，例如：

```text
data/example/
├── sample_REF.png
└── sample_TAR.png
```

文件名分别包含大写 `REF` 和 `TAR`。脚本按 `*REF*.*`、`*TAR*.*` 查找文件，并取首个匹配项，因此每个目录应仅放一组匹配图像。两幅图像应具有相同尺寸，脚本使用 OpenCV 以灰度方式读取。

编辑准备运行的入口文件，将其中的 `DATA_DIR` 设为图像目录，例如：

```python
DATA_DIR = r"D:\DIC_data\example"
```

四种入口的 `DATA_DIR` 分别设置，当前代码没有统一命令行参数。比较同一图像对时，应将四处路径指向同一数据目录。默认 POI 边界为 12 像素，测试图像需足以容纳子区和边界。

## 5. 运行方法

在项目根目录打开终端，按需运行：

```bash
python source/run_gmdic_icgn.py
python source/run_fftcc_icgn.py
python source/run_sift_icgn.py
python source/run_RG_icgn.py
```

TFM-ICGN 已集成网络推理和 IC-GN 精化，无需提前生成初值 CSV。入口从项目内的 `checkpoints/step_250000.pth` 加载权重，路径相对于脚本位置解析。

输出写入 `DATA_DIR`，重复运行会更新同名结果。FFT、SIFT 和 RG 入口包含顶层执行代码，应作为脚本运行，避免将它们作为普通库模块导入。

## 6. 模型与求解参数

### TFM-DIC

| 参数 | 当前入口使用值 |
| --- | --- |
| `num_scales` | 1 |
| `feature_channels` | 128 |
| `num_transformer_blocks` | 16 |
| `num_head` | 1 |
| `attention_type` | `swin` |
| `ffn_dim_expansion` | 4 |
| `upsample_factor` | 2 |
| `attn_splits_list` | `[8]` |
| `corr_radius_list` | `[5]` |
| 特征流传播 | 关闭 |

特征分辨率为输入图像的 1/2，采用特征网格上半径为 5 的局部相关匹配和学习式 2 倍上采样。`attn_splits_list=[8]` 表示窗口划分数量，不表示所有输入尺寸下均采用固定 8×8 窗口。当前推理并非无限搜索范围的全图两两相关。

附带权重对应 250,000 步训练；训练批量为 2、随机种子为 326、学习率为 0.0002、权重衰减为 0.0001、损失加权系数 gamma 为 0.9。训练记录中的配置已整理至 `model_config.json`。

`model_config.json` 是参数记录，当前入口不会自动读取该文件。自行实例化 `tfmdic` 时须显式传入上述模型参数，不能直接使用类的全部默认参数加载本权重。

### IC-GN

| 参数 | 当前入口使用值 |
| --- | --- |
| 形函数 | 二阶、12 参数 |
| 子区半径 | 10，即 21×21 像素 |
| 收敛阈值 | 0.0001 |
| 最大迭代次数 | 50 |
| 默认 POI 步长 | 1 像素 |
| 优化准则 | ZNSSD |
| 参考图像梯度 | 四阶差分 |
| 目标图像插值 | B 样条插值 |

## 7. 输出与指标

位移单位为像素，`U` 对应图像列方向，`V` 对应图像行方向。

| 方法 | 主要输出 |
| --- | --- |
| TFM-ICGN | 初值 `GM_U.csv`、`GM_V.csv`；精化结果 `GMGN_U.csv`、`GMGN_V.csv`；对应位移图 |
| FFT-ICGN | `FFT_ICGN2_U.csv`、`FFT_ICGN2_V.csv`、`FFT_ICGN2_displacement.png` |
| SIFT-ICGN | `SIFT_ICGN2_U.csv`、`SIFT_ICGN2_V.csv`、`SIFT_ICGN2_displacement.png` |
| RG-ICGN | `RG_ICGN2_U.csv`、`RG_ICGN2_V.csv`、`RG_ICGN2_displacement.png` |

默认精化结果裁去四周 12 像素边界，数组位置需结合裁剪偏移映射回原图。TFM 网络初值文件保留完整图像尺寸，CSV 每行末尾带有逗号。

不同入口的无效点输出约定尚未统一。尤其 RG 的未访问点会在 CSV 中填 0，部分失败点保留初值；不能仅依据 CSV 中的零值判断真实零位移或求解成功。

TFM、FFT、SIFT 入口会更新数据目录中的 `实验指标汇总.csv`。当前 RG 入口仅使用其独立的控制台统计，没有接入该汇总逻辑，运行四个入口不会自动得到完整、统一的四方法结果表。

如需计算初值偏差 IDD，可在图像目录提供同前缀的 `*_GT_U.csv` 和 `*_GT_V.csv`。支持完整尺寸或可通过对称边界裁剪对齐的真值矩阵，并要求覆盖选中点。没有真值时，相关入口报告 IC-GN 精化前后的位移修正量；修正量不等于相对真值误差。

## 8. 开源整理版与实验版本的差异

本版已移除 RG 的初值分量 20 px 限制，以及一阶、二阶 IC-GN 和 SIFT 入口中的最终位移分量 12 px 限制。有限值、边界、ZNCC 和收敛检查仍保留。删除分量阈值不意味着方法能对任意大位移保证收敛。

FFT 仍使用邻域中值偏差过滤：以 5×5 邻域计算中值，将位移向量相对中值的偏差大于 12 px 的点标为无效。这与最终位移分量的绝对上限不同。

代码中的算法注释与独立说明字符串已移除，许可证要求的法律声明予以保留；网络模块和类名统一为 `tfmdic`。本版完成了语法检查、模型权重严格加载，以及一阶/二阶 IC-GN 的 24 px 合成平移检查。尚未使用本版重新执行论文的完整实验；原论文数值不能直接视为本版重新测得的结果。

## 9. 许可状态

本项目采用分文件许可：默认使用 [Apache-2.0](LICENSE)；七个 IC-GN 核心文件使用 [MPL-2.0](LICENSES/MPL-2.0.txt)，RG 入口使用 [BSD-3-Clause](LICENSES/BSD-3-Clause.txt)。具体文件范围和必须保留的法律声明见 [NOTICE](NOTICE)。随附模型权重、示例图像和项目文档采用 Apache-2.0，既有第三方权利不受影响。

本包包含相应源码。再分发时须保留适用的许可、版权及修改声明；代码头部的法律声明不属于已清除的算法注释。

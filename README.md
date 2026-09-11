# TFM-ICGN | 数字图像相关参考实现 · Digital Image Correlation

[中文](#chinese) | [English](#english)

<a id="chinese"></a>

## 中文

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

## 8. 许可状态

本项目采用分文件许可：默认使用 [Apache-2.0](LICENSE)；七个 IC-GN 核心文件使用 [MPL-2.0](LICENSES/MPL-2.0.txt)，RG 入口使用 [BSD-3-Clause](LICENSES/BSD-3-Clause.txt)。具体文件范围和必须保留的法律声明见 [NOTICE](NOTICE)。随附模型权重、示例图像和项目文档采用 Apache-2.0，既有第三方权利不受影响。

本包包含相应源码。再分发时须保留适用的许可、版权及修改声明；代码头部的法律声明不属于已清除的算法注释。

---

<a id="english"></a>

# English

[中文](#chinese) | **English**

This repository provides reference implementations of four two-dimensional digital image correlation (DIC) methods: **TFM-ICGN, FFT-ICGN, SIFT-ICGN, and RG-ICGN**. They estimate in-plane displacement between a reference image and a deformed image.

TFM-ICGN uses the Transformer-based TFM-DIC network to predict an initial displacement field, followed by second-order inverse-compositional Gauss–Newton (IC-GN) refinement for subpixel estimation. The other three methods use FFT cross-correlation, SIFT feature matching, and reliability-guided propagation for initialization, respectively.

The repository includes method implementations, the final model checkpoint, and the REF/TAR image pair from the `tension009` experiment. Training scripts, the training dataset, and the original experimental result files are not included.

## 1. Methods and entry points

| Method | Initialization | Entry point |
| --- | --- | --- |
| TFM-ICGN | TFM-DIC network prediction | `source/run_gmdic_icgn.py` |
| FFT-ICGN | FFT cross-correlation | `source/run_fftcc_icgn.py` |
| SIFT-ICGN | SIFT matching and local affine estimation | `source/run_sift_icgn.py` |
| RG-ICGN | FFT initialization of a central seed, followed by ZNCC-prioritized propagation to four-connected neighbors | `source/run_RG_icgn.py` |

All four entry points use the project's second-order, 12-parameter IC-GN solver. RG-ICGN is this project's reliability-guided implementation and is not equivalent to the complete official Ncorr implementation. Each point is attempted at most once; failed points do not propagate, and additional seeds are not introduced automatically.

The network module and class are both named `tfmdic`. The identifiers `gmdic`, `GM`, and `GMGN` retained in entry points and output filenames are historical names.

## 2. Repository structure

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
    ├── run_gmdic_icgn.py
    ├── run_fftcc_icgn.py
    ├── run_sift_icgn.py
    ├── run_RG_icgn.py
    ├── networks/
    │   ├── tfmdic.py
    │   └── ...
    ├── icgn/
    │   └── ...
    └── utils/
        └── utils.py
```

`networks/` contains feature extraction, Transformer, and correlation matching modules. `icgn/` contains initialization, gradient computation, interpolation, IC-GN refinement, and metric reporting.

## 3. Dependencies

The following local environment was used for syntax checks, checkpoint loading, and targeted solver checks. This does not constitute comprehensive compatibility testing across platforms.

| Software | Version |
| --- | --- |
| Python | 3.10.16 |
| PyTorch | 2.5.1 |
| NumPy | 2.0.1 |
| SciPy | 1.15.2 |
| OpenCV | 4.10.0 |
| Matplotlib | 3.10.0 |

With Python 3.10, install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` includes PyTorch 2.5.1. To select a specific CPU or CUDA build, install the appropriate PyTorch 2.5.1 build first, then install the remaining dependencies. The TFM-ICGN entry point selects its device according to CUDA availability; the complete image-processing pipeline has not yet been validated on both CPU and GPU. The other three methods do not require network weights.

## 4. Prepare input images

The original `tension009` image pair is included as `data/tension009/tensionREF.bmp` and `data/tension009/tensionTAR.bmp`. To use it, run commands from the repository root and set `DATA_DIR` in each entry point you intend to run to `'data/tension009'`. The scripts currently retain their original paths, so this setting must be changed manually. Ground-truth CSV files are not included in this example; the metric module therefore reports the IC-GN correction magnitude rather than ground-truth error.

For another image pair, create a separate directory containing one reference image and one deformed image:

```text
data/example/
├── sample_REF.png
└── sample_TAR.png
```

Filenames must contain uppercase `REF` and `TAR`, respectively. The scripts search for `*REF*.*` and `*TAR*.*` and use the first match, so each directory should contain only one matching image pair. Both images should have identical dimensions. OpenCV reads them as grayscale images.

Set `DATA_DIR` in the entry point to your image directory, for example:

```python
DATA_DIR = r"D:\DIC_data\example"
```

Each entry point has its own `DATA_DIR`; there is currently no shared command-line interface. To compare the same image pair, point all four scripts to the same directory. The default points-of-interest (POI) domain excludes a 12-pixel border, so input images must be large enough to accommodate the subsets and border.

## 5. Run the methods

Open a terminal in the repository root and run the methods you need:

```bash
python source/run_gmdic_icgn.py
python source/run_fftcc_icgn.py
python source/run_sift_icgn.py
python source/run_RG_icgn.py
```

TFM-ICGN integrates network inference and IC-GN refinement; no initial-displacement CSV needs to be generated beforehand. The checkpoint is loaded from `checkpoints/step_250000.pth`, with its path resolved relative to the script location.

Outputs are written to `DATA_DIR`. Repeated runs update files with the same names. The FFT, SIFT, and RG entry points execute code at module level and should be run as scripts rather than imported as ordinary library modules.

## 6. Model and solver parameters

### TFM-DIC model

| Parameter | Value used by the entry point |
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
| Feature-flow propagation | Disabled |

Features are extracted at half the input resolution. Matching uses local correlation with a radius of 5 on the feature grid, followed by learned 2× upsampling. `attn_splits_list=[8]` specifies the number of window splits, not a fixed 8×8 window for every input size. The current inference configuration does not perform unrestricted all-pairs image matching.

The checkpoint corresponds to 250,000 training steps, a batch size of 2, random seed 326, learning rate 0.0002, weight decay 0.0001, and loss-weighting factor gamma 0.9. These settings are recorded in `model_config.json`.

`model_config.json` documents the configuration; the entry point does not read it automatically. When instantiating `tfmdic` yourself, explicitly provide the model parameters above rather than relying on all of the class defaults to load this checkpoint.

### IC-GN solver

| Parameter | Value used by the entry points |
| --- | --- |
| Shape function | Second order, 12 parameters |
| Subset radius | 10, giving a 21×21-pixel subset |
| Convergence threshold | 0.0001 |
| Maximum iterations | 50 |
| Default POI spacing | 1 pixel |
| Optimization criterion | ZNSSD |
| Reference-image gradients | Fourth-order finite differences |
| Target-image interpolation | B-spline interpolation |

## 7. Outputs and metrics

Displacement is measured in pixels. `U` is displacement along image columns, and `V` is displacement along image rows.

| Method | Main outputs |
| --- | --- |
| TFM-ICGN | Initial fields: `GM_U.csv`, `GM_V.csv`; refined fields: `GMGN_U.csv`, `GMGN_V.csv`; corresponding displacement plots |
| FFT-ICGN | `FFT_ICGN2_U.csv`, `FFT_ICGN2_V.csv`, `FFT_ICGN2_displacement.png` |
| SIFT-ICGN | `SIFT_ICGN2_U.csv`, `SIFT_ICGN2_V.csv`, `SIFT_ICGN2_displacement.png` |
| RG-ICGN | `RG_ICGN2_U.csv`, `RG_ICGN2_V.csv`, `RG_ICGN2_displacement.png` |

By default, refined outputs exclude a 12-pixel border on all sides. Account for this offset when mapping output coordinates back to the original image. The TFM initial-displacement files retain the full image dimensions and contain a trailing comma on each CSV row.

Invalid-point conventions differ between entry points. In particular, RG writes unvisited points as zeros in its CSV outputs, and some failed points retain their initial estimates. A zero in the CSV is therefore not sufficient to identify either a true zero displacement or a successful solution.

The TFM, FFT, and SIFT entry points update `实验指标汇总.csv` in the data directory. The current RG entry point uses separate console reporting and does not update this summary. Running all four entry points does not automatically produce a complete, consistently defined four-method results table.

To calculate initial displacement deviation (IDD), provide matching `*_GT_U.csv` and `*_GT_V.csv` ground-truth files with the same prefix in the image directory. Full-size matrices and matrices aligned by symmetric border cropping are supported, provided they cover all selected points. Without ground truth, the relevant entry points report the displacement correction between initialization and IC-GN refinement. This correction is not an error measured against ground truth.

## 8. Licensing

Licensing is specified per file. [Apache-2.0](LICENSE) is the default license; seven IC-GN core files use [MPL-2.0](LICENSES/MPL-2.0.txt), and the RG entry point uses [BSD-3-Clause](LICENSES/BSD-3-Clause.txt). See [NOTICE](NOTICE) for the exact file scope and required legal notices. The included model checkpoint, example images, and project documentation use Apache-2.0; existing third-party rights remain unaffected.

The corresponding source code is included. Redistributions must retain applicable license, copyright, and modification notices. Legal notices in source-file headers are separate from the explanatory algorithm comments that were removed.

[Back to 中文](#chinese)

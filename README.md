<div align="center">

# Robotic Arm Grasping: 视觉抓取预测与六自由度策略学习

**基于 PyBullet Franka Panda 仿真的监督视觉抓取（GR-ConvNet3 闭环伺服）与六自由度强化学习策略（GA-DDPG + OMG + DAgger）代码实现。**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](#环境与前置依赖)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B%20%7C%20CUDA-ee4c2c?style=flat-square)](#环境与前置依赖)
[![PyBullet](https://img.shields.io/badge/Simulator-PyBullet%203.25-4c8bf5?style=flat-square)](#核心模块)

</div>

[核心模块](#核心模块) · [代码继承与致谢](#代码继承与致谢) · [环境与前置依赖](#环境与前置依赖) · [快速开始](#快速开始)

> [!NOTE]
> 本仓库仅包含核心训练、推理、仿真控制与测试代码；数据集（Jacquard / QDG500 / YCB）、训练权重（`.pt`）、3D 网格与贴图资产、实验日志及课程汇报文稿均未纳入版本控制。

---

## 核心模块

项目包含两条机械臂抓取技术路线的完整代码实现：

| 路径 | 范式 | 说明 |
| --- | --- | --- |
| [`train_jacquard_v2.py`](train_jacquard_v2.py) | 监督视觉抓取 | 基于 Jacquard v2 RGB-D 数据集训练 `GR-ConvNet3`，支持 GPU 融合图像增强与按物体 ID 划分 |
| [`pybullet_grasp.py`](pybullet_grasp.py) | 监督视觉闭环控制 | Franka Panda 腕部 RGB-D 相机 20 Hz 闭环视觉伺服、π 对称抓取轴平滑、斜向接近规划与本体感知重试（支持 `--control-mode closed-loop / open-loop`、途中扰动 `--disturb-cm` 与 GUI 交互） |
| [`test_pybullet_grasp.py`](test_pybullet_grasp.py) | 单元测试 | 覆盖坐标反投影、抓取轴几何、对称轴对齐与闭环步进边界检查 |
| [`rl_grasp/ga_env.py`](rl_grasp/ga_env.py) | 6-DoF 策略环境 | 96×96 腕部深度分割、基座系历史点云聚合（3,074 维观测）、7 维连续动作与 `retract()` 接近—合夹解耦 |
| [`rl_grasp/pointnet2.py`](rl_grasp/pointnet2.py) | 点云编码器 | 半宽三级 `PointNet++`（`SA1 / SA2 / SA3`），支持 Linux 原生 `pointnet2_ops` CUDA 扩展与权重完全等价的纯 PyTorch 回退实现 |
| [`rl_grasp/ga_model.py`](rl_grasp/ga_model.py) | GA-DDPG 网络与更新 | `Actor`、`Critic`（TD3 双 Q + Goal-Auxiliary 辅助头）与 5 关键点位姿误差 `pose_error` |
| [`rl_grasp/ga_planner.py`](rl_grasp/ga_planner.py) | 在线规划与纠偏 | 实时 OMG 前向轨迹优化、最近抓取目标监督与在线 DAgger 专家标签生成 |
| [`rl_grasp/qdg_env.py`](rl_grasp/qdg_env.py) / [`rl_grasp/qdg_demo.py`](rl_grasp/qdg_demo.py) | QDG500 多物体场景 | QDGset 网格加载、凸包碰撞与示教轨迹采集 |
| [`rl_grasp/train_ga.py`](rl_grasp/train_ga.py) / [`rl_grasp/ga_async.py`](rl_grasp/ga_async.py) | 分布式训练 | 多进程异步采样、BC 预热、TD3 + GA + DAgger 联合训练与 CUDA Graph 双路径捕获 |
| [`rl_grasp/demo_ga.py`](rl_grasp/demo_ga.py) | 6-DoF GUI 演示 | 24 Hz 实时节流回放、腕部相机分割监控面板、鼠标中键旋转视角与多模式切换 |
| [`rl_grasp/train_qt.py`](rl_grasp/train_qt.py) / [`rl_grasp/train_qt_rgb.py`](rl_grasp/train_qt_rgb.py) / [`rl_grasp/train.py`](rl_grasp/train.py) | 基线方法 | 早期 QT-Opt（状态/RGB）与 SAC 抓取基线实现 |

---

## 代码继承与致谢

本项目在以下开源仓库与研究工作基础上进行二次开发与扩展：

1. **监督视觉抓取上游仓库：[`skumra/robotic-grasping`](https://github.com/skumra/robotic-grasping)**（基于提交 [`183c6f6`](https://github.com/skumra/robotic-grasping/tree/183c6f68c44c1c7ff0f07707e2db6fcfd6840d2d)）
   - [`train_jacquard_v2.py`](train_jacquard_v2.py) 与 [`pybullet_grasp.py`](pybullet_grasp.py) 直接复用上游 `GR-ConvNet3`（`GenerativeResnet`）网络结构、`post_process_output` 抓取解码以及 Jacquard 数据读取与 IoU 匹配工具；
   - 在上游静态单帧预测的基础上，本项目新增了 PyBullet Panda 腕部相机仿真、20 Hz 闭环视觉伺服、本体感知失败重试以及 GPU 融合批增强训练入口。

2. **六自由度策略学习框架参考**
   - **GA-DDPG**：参考 Wang 等人 *Goal-Auxiliary Actor-Critic for 6D Robotic Grasping with Point Clouds*（CoRL 2021）的 Goal-Auxiliary 监督与 `retract()` 解耦思路；
   - **点云编码器**：[`rl_grasp/pointnet2.py`](rl_grasp/pointnet2.py) 原生路径调用 [`pointnet2_ops`](https://github.com/erikwijmans/Pointnet2_PyTorch) CUDA 扩展（可通过 [`rl_grasp/install_pointnet2.sh`](rl_grasp/install_pointnet2.sh) 在 Linux CUDA 环境编译），并提供用于本地 Windows 推理与演示的纯 PyTorch 等价回退模块；
   - **QDGset 物体集**：[`rl_grasp/qdg_env.py`](rl_grasp/qdg_env.py) 与 [`rl_grasp/qdg_demo.py`](rl_grasp/qdg_demo.py) 适配 [QDGset](https://huggingface.co/datasets/QDGrasp/QDGset_no_shapenet) 物体网格与 6D 抓取标注格式。

---

## 环境与前置依赖

- **Python**：3.11
- **核心依赖**：`torch`（CUDA）、`torchvision`、`pybullet==3.25`、`numpy==1.23.5`、`scipy`、`pillow`、`scikit-image`、`opencv-python`、`tifffile`、`imagecodecs`、`trackio`
- **上游仓库准备**：将 `skumra/robotic-grasping` 克隆到仓库根目录下的 `robotic-grasping/` 并检出 `183c6f68c44c1c7ff0f07707e2db6fcfd6840d2d`：

```powershell
git clone https://github.com/skumra/robotic-grasping.git
git -C robotic-grasping checkout 183c6f68c44c1c7ff0f07707e2db6fcfd6840d2d
```

---

## 快速开始

### 1. 运行单元测试

```powershell
python -m unittest test_pybullet_grasp
python -m unittest discover -s rl_grasp -p "test_*.py"
```

### 2. 监督视觉闭环抓取仿真（GR-ConvNet3）

需先准备本地训练权重（或上游 `robotic-grasping/trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93`）：

```powershell
# 交互式 GUI 闭环抓取演示（Space 抓取 / R 重置 / N 换物体 / T 换倾角 / C 切换闭环与开环）
python pybullet_grasp.py --checkpoint path/to/best.pt --gui --interactive --object-set ycb --random-seed

# 途中 6 cm 物体扰动闭环测试
python pybullet_grasp.py --checkpoint path/to/best.pt --object long_box --trials-per-object 5 --disturb-cm 6.0
```

### 3. 六自由度 GA-DDPG 策略训练与演示

```powershell
# 启动 QDG500 + OMG + GA-DDPG 训练（Linux CUDA，详见 rl_grasp/run_autodl_qdg500_omg_v5.sh）
bash rl_grasp/run_autodl_qdg500_omg_v5.sh

# 本地 GUI 策略演示（需指定训练权重与 QDG500 数据目录）
python rl_grasp/demo_ga.py --checkpoint path/to/best.pt --dataset path/to/qdg500 --speed 0.8
```

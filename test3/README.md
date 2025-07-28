# 🚀 Movement Tracking MuJoCo

**A comprehensive reinforcement learning platform for humanoid movement simulation, biomechanical analysis, and muscle modeling using MuJoCo physics engine.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Interactive Dashboards](#-interactive-dashboards)
- [Trained Models](#-trained-models)
- [Muscle Modeling](#-muscle-modeling)
- [Usage Examples](#-usage-examples)
- [Technical Details](#-technical-details)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project is a sophisticated RL research platform that bridges computational neuroscience, biomechanics, and reinforcement learning. It provides tools for studying human movement patterns, developing motor control systems, and analyzing complex biomechanical behaviors.

### Key Applications
- **Human Movement Analysis**: Understanding biomechanical patterns
- **RL Algorithm Comparison**: Benchmarking SAC vs TD3 vs A2C
- **Muscle-Level Control**: Fine-grained neuromuscular simulation  
- **Complex Motor Skills**: Ball handling, walking, coordination tasks

---

## ✨ Features

### 🤖 **Reinforcement Learning**
- **Multi-Algorithm Support**: SAC, TD3, and A2C implementations
- **Extensive Model Library**: 400+ trained models across algorithms
- **Version Compatibility**: Support for Humanoid-v4 and v5 environments
- **Automated Training Pipeline**: Configurable training with checkpointing

### 🎛️ **Interactive Dashboards**
- **Enhanced MuJoCo Viewer**: Real-time model visualization and control
- **Performance Analytics**: Live metrics and comprehensive analysis
- **Video Recording**: Automated performance capture and replay
- **Model Comparison**: Side-by-side algorithm evaluation

### 🧠 **Biomechanical Modeling**
- **Hill Muscle Model**: Physiologically accurate force generation
- **EMG Signal Simulation**: Realistic muscle activation patterns
- **Agonist/Antagonist Coordination**: Muscle pair interactions
- **Length-Tension Relationships**: Biomechanically accurate modeling

### 🎮 **Specialized Simulations**
- **Humanoid Walking**: Advanced locomotion with RL training
- **Ball Control**: Complex object manipulation tasks
- **Muscle Visualization**: Detailed biomechanical animation
- **Custom Control Schemes**: User-defined movement patterns

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- MuJoCo license (free for personal/academic use)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Movement_Tracking_Mujoco/test3

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_version_compatibility.py
```

### Dependencies
```
# Core RL and MuJoCo
mujoco>=2.3.0
gymnasium[mujoco]>=0.29.0
stable-baselines3>=2.0.0

# Scientific Computing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Jupyter Environment
jupyter>=1.0.0
ipywidgets>=8.0.0
jupyterlab>=4.0.0

# ML and Monitoring
tensorboard>=2.13.0
torch>=2.0.0
opencv-python>=4.8.0
```

---

## 🚀 Quick Start

### 1. Interactive Dashboard (Recommended)
```bash
# Launch Jupyter Lab
jupyter lab

# Open either dashboard:
# - enhanced_mujoco_viewer.ipynb (comprehensive UI)
# - model_visualization_dashboard.ipynb (advanced analytics)
```

### 2. Command Line Training
```bash
# Train a new model
python sb3.py Humanoid-v4 SAC -t

# Test existing model
python sb3.py Humanoid-v4 SAC -s models/SAC_100000.zip
```

### 3. Muscle Simulation
```bash
# Basic muscle model
python visualize_muscle.py

# Advanced Hill muscle model
python visualize_hill_muscle.py
```

---

## 📁 Project Structure

```
test3/
├── 📊 Interactive Dashboards
│   ├── enhanced_mujoco_viewer.ipynb      # Main UI for model control
│   └── model_visualization_dashboard.ipynb # Advanced analytics
│
├── 🤖 Core RL Scripts
│   ├── sb3.py                           # Main training/testing script
│   ├── visualize_humanoid_walking.py    # Walking demonstrations
│   └── test_version_compatibility.py    # Environment testing
│
├── 🧠 Muscle Modeling
│   ├── visualize_muscle.py              # Basic muscle simulation
│   ├── visualize_hill_muscle.py         # Advanced Hill model
│   └── muscle/                          # XML model definitions
│       ├── muscle_model.xml
│       └── hill_muscle_model.xml
│
├── 🎮 Specialized Simulations
│   ├── humanoid_ball_control.py         # Ball catching/carrying
│   └── walking_basics_demo.py           # Locomotion fundamentals
│
├── 📦 Models & Data
│   ├── models/                          # Trained RL models (400+)
│   ├── videos/                          # Recorded demonstrations
│   └── logs/                            # Training logs
│
└── 🔧 Configuration
    ├── requirements.txt                 # Dependencies
    └── validation.py                    # System validation
```

---

## 🎛️ Interactive Dashboards

### Enhanced MuJoCo Viewer
**Primary interface for model interaction and visualization**

**Features:**
- 🤖 **Algorithm Selection**: Choose between A2C, SAC, TD3
- 📊 **Model Management**: Browse 400+ trained models
- ⚙️ **Parameter Control**: Episodes, steps, seeds, video recording
- 📈 **Real-time Analytics**: Live performance metrics
- 🎮 **Interactive Controls**: Start, stop, refresh capabilities

**Usage:**
1. Open `enhanced_mujoco_viewer.ipynb`
2. Select algorithm and model from dropdowns
3. Configure execution parameters
4. Click "🚀 Run Viewer" to start simulation
5. Use "📈 Visualize" for detailed analysis

### Model Visualization Dashboard
**Advanced analytics and model comparison platform**

**Features:**
- 📊 **Live Metrics**: Real-time performance tracking during simulation
- 🔍 **Model Comparison**: Side-by-side algorithm evaluation
- 📈 **Advanced Plotting**: Interactive visualizations with Plotly
- 🎥 **Video Recording**: Automated performance capture
- 📋 **Detailed Reports**: Comprehensive performance analysis

---

## 🤖 Trained Models

The project includes an extensive library of pre-trained models:

### A2C Models (325+ models)
- **Training Range**: 25,000 - 8,125,000 steps
- **Best Performance**: Advanced locomotion patterns
- **Use Case**: Quick prototyping and baseline comparisons

### SAC Models (78+ models)
- **Training Range**: 25,000 - 2,050,000 steps  
- **Best Performance**: Complex continuous control tasks
- **Use Case**: High-quality humanoid walking and manipulation

### TD3 Models (22+ models)
- **Training Range**: 25,000 - 550,000 steps
- **Best Performance**: Deterministic control policies
- **Use Case**: Robotic applications requiring precision

### Model Organization
```
models/
├── A2C_[steps].zip     # Advantage Actor-Critic models
├── SAC_[steps].zip     # Soft Actor-Critic models
└── TD3_[steps].zip     # Twin Delayed DDPG models
```

---

## 🧠 Muscle Modeling

### Hill Muscle Model
**Physiologically accurate muscle force generation**

```python
class HillMuscle:
    def __init__(self, fmax=1000, lmin=0.1, lmax=0.3):
        self.fmax = fmax  # Maximum muscle force
        self.lmin = lmin  # Minimum muscle length
        self.lmax = lmax  # Maximum muscle length
        
    def compute_force(self, activation, length, velocity):
        # Length-tension relationship
        l_norm = (length - self.lmin) / (self.lmax - self.lmin)
        fl = np.exp(-((l_norm - 0.5) ** 2) / 0.1)
        
        # Force-velocity relationship  
        v_norm = velocity / self.l0
        fv = (1 - v_norm) / (1 + v_norm) if v_norm > 0 else (1 + v_norm) / (1 - v_norm)
        
        return self.fmax * activation * fl * fv
```

### EMG Signal Simulation
**Realistic muscle activation patterns**
- Noise modeling for biological realism
- Phase control for agonist/antagonist coordination
- Temporal dynamics matching experimental data

### Applications
- **Biomechanical Research**: Accurate muscle force prediction
- **Rehabilitation**: Movement analysis and therapy planning
- **Prosthetics**: Control signal generation for artificial limbs

---

## 📖 Usage Examples

### Training a New Model
```python
# Train SAC model on Humanoid-v4
python sb3.py Humanoid-v4 SAC -t

# Monitor training with TensorBoard
tensorboard --logdir logs/
```

### Testing Model Performance
```python
# Test specific model
python sb3.py Humanoid-v4 SAC -s models/SAC_1000000.zip

# Compare environment versions
python test_version_compatibility.py
```

### Muscle Simulation
```python
# Run Hill muscle simulation
python visualize_hill_muscle.py

# Basic muscle model
python visualize_muscle.py
```

### Ball Control Demonstration
```python
# Advanced humanoid ball control
python humanoid_ball_control.py
```

---

## 🔬 Technical Details

### Algorithms

#### SAC (Soft Actor-Critic)
- **Type**: Maximum entropy off-policy algorithm
- **Best For**: Humanoid walking, complex continuous control
- **Pros**: Highly sample efficient, excellent exploration, stable
- **Cons**: More computationally complex

#### TD3 (Twin Delayed DDPG)  
- **Type**: Improved DDPG with twin critics
- **Best For**: Continuous control, robotic manipulation
- **Pros**: Good for continuous control, more stable than DDPG
- **Cons**: Less exploration than SAC, hyperparameter sensitive

#### A2C (Advantage Actor-Critic)
- **Type**: Policy gradient with value function baseline
- **Best For**: Quick training, simpler tasks
- **Pros**: Fast computation, stable on simple tasks, low memory
- **Cons**: Sample inefficient, struggles with complex control

### Environment Compatibility
- **Humanoid-v4**: Legacy environment, extensive model support
- **Humanoid-v5**: Updated environment, improved physics
- **HumanoidStandup-v4**: Specialized standing/balance tasks

### Performance Metrics
- **Episode Rewards**: Cumulative reward per episode
- **Episode Length**: Steps until termination/truncation
- **Success Rate**: Percentage of episodes with positive rewards
- **Stability**: Reward variance across episodes

---

## 🎥 Video Documentation

All simulations support automatic video recording:

```python
# Enable video recording in dashboards
record_video = True

# Videos saved to: videos/
# Format: [ModelName]-episode-[N].mp4
```

### Available Videos
- `SAC_v5_working-episode-*.mp4`: SAC model demonstrations
- `A2C_1ep_fixed-episode-*.mp4`: A2C baseline performance
- `rl-video-episode-*.mp4`: General RL demonstrations

---

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install black flake8 pytest

# Format code
black *.py

# Run tests
pytest validation.py
```

### Guidelines
- Follow PEP 8 style conventions
- Add docstrings for new functions
- Include performance benchmarks for new models
- Update README for new features

---

## 📚 References

- **MuJoCo Documentation**: [mujoco.readthedocs.io](https://mujoco.readthedocs.io)
- **Stable-Baselines3**: [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io)
- **Gymnasium**: [gymnasium.farama.org](https://gymnasium.farama.org)
- **Hill Muscle Model**: Zajac, F.E. (1989) "Muscle and tendon: properties, models, scaling, and application to biomechanics"

---

## 📄 License

This project is available for academic and research purposes. Please cite appropriately in publications.

---

## 👥 Authors

**Anbar Althaf & Arhaan Girdhar**

*Interactive interfaces and enhanced visualization systems for reinforcement learning in biomechanical simulation.*

---

**🚀 Ready to explore human movement simulation? Start with the Enhanced MuJoCo Viewer notebook!** 
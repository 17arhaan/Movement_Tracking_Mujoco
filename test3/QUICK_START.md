# 🎯 QUICK START - Model Visualization

## Copy-Paste Commands (Run these in order):

### 1. Setup
```bash
cd test3
source ../venv/bin/activate
```

### 2. Quick Model Test
```bash
# See all your trained models
python simple_model_viewer.py --list

# Test a well-trained SAC model (SHORT: 1 episode)
python simple_model_viewer.py --model SAC_325000.zip --episodes 1

# Test with video recording
python simple_model_viewer.py --model SAC_325000.zip --episodes 1 --record
```

### 3. Compare Different Algorithms  
```bash
# SAC vs TD3 vs A2C (pick the highest step models you have)
python simple_model_viewer.py --model SAC_325000.zip --episodes 2
python simple_model_viewer.py --model TD3_350000.zip --episodes 2  
python simple_model_viewer.py --model A2C_1000000.zip --episodes 2
```

### 4. Interactive Dashboard
```bash
# Launch the full Jupyter interface
jupyter lab model_visualization_dashboard.ipynb
```

## What You'll See:

**Command-Line Viewer:**
- Live MuJoCo window with humanoid walking
- Terminal output showing episode progress
- Automatic performance plots after completion
- Videos saved to `videos/` folder

**Jupyter Dashboard:**
- Professional web interface with controls
- Real-time performance charts
- Interactive model selection
- Live metrics dashboard

## Tips:
- Start with 1-2 episodes to see it working quickly
- Use `--record` to save videos of best performances  
- The SAC and TD3 models generally perform better than A2C
- Higher step counts usually mean better performance 
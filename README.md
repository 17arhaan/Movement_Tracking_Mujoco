

# Movement Tracking Mujoco

Movement Tracking Mujoco is a project that demonstrates how to use the [MuJoCo](https://mujoco.org/) physics engine with Python and React to simulate, track, and visualize humanoid movement. It integrates motion tracking, LSTM-based prediction, and an interactive GUI, making it ideal for researchers and developers in robotics, biomechanics, or machine learning.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This project simulates a humanoid model (e.g., `humanoid.xml`) using MuJoCo, tracks its motion, and employs an LSTM model to predict or analyze movement sequences. A React-based GUI allows users to interact with the simulation, load data, train models, and visualize results in real-time or post-processed formats.

## Features

- **Humanoid Simulation:** Realistic movement simulation using MuJoCo.
- **Motion Tracking:** Captures joint angles, velocities, and other metrics.
- **LSTM Prediction:** Trains an LSTM model for motion forecasting or classification.
- **Interactive GUI:** React frontend for controlling and visualizing the simulation.
- **Visualization:** Renders MuJoCo scenes and generates Matplotlib plots.
- **Extensibility:** Modular design for custom models, sensors, or algorithms.

## Requirements

- **Python 3.7+**
- **MuJoCo** (see [installation guide](https://mujoco.org/docs/installation.html))
- **mujoco-py** (Python bindings for MuJoCo)
- **Node.js 16+ and npm** (for React frontend)
- **Python Dependencies**: `numpy`, `tensorflow`, `flask`, `matplotlib`, `pandas` (listed in `backend/requirements.txt`)
- **Node.js Dependencies**: `react`, `react-dom` (listed in `frontend/package.json`)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/17arhaan/Movement_Tracking_Mujoco.git
   cd Movement_Tracking_Mujoco
   ```

2. **Set Up the Backend**

   - Install MuJoCo and configure your license (see [MuJoCo docs](https://mujoco.org/docs/installation.html)).
   - Set up a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install Python dependencies:
     ```bash
     pip install -r backend/requirements.txt
     ```

3. **Set Up the Frontend**

   - Navigate to the frontend directory:
     ```bash
     cd frontend
     ```
   - Install Node.js dependencies:
     ```bash
     npm install
     ```

4. **Verify Setup**

   Ensure `humanoid.xml` is placed in `backend/` or adjust paths in `mujoco_sim.py`.

## Usage

### Running the Backend

Launch the Flask API to manage simulation and LSTM processing:

```bash
cd backend
python api.py
```

The server runs at `http://localhost:5000`.

### Running the Frontend

Start the React GUI:

```bash
cd frontend
npm start
```

Visit `http://localhost:3000` to access the interface.

### Running Standalone Simulation (Optional)

To run a standalone MuJoCo simulation without the GUI:

```bash
cd backend
python mujoco_sim.py
```

This renders the humanoid model with dummy joint angles.

### Training the LSTM Model

The GUI’s "Train LSTM" button triggers training via the API. Alternatively, run standalone:

```bash
cd backend
python lstm_model.py
```

Modify `lstm_model.py` to load your dataset.

### Visualization

The GUI’s "Simulate Motion" button renders MuJoCo output. For standalone plots:

```bash
cd backend
python motion_tracker.py  # Add Matplotlib plotting logic here
```

## Project Structure

```
Movement_Tracking_Mujoco/
├── backend/                  # Python backend for simulation and ML
│   ├── motion_tracker.py     # Tracks and preprocesses motion data
│   ├── lstm_model.py         # Defines and trains the LSTM model
│   ├── mujoco_sim.py         # Runs MuJoCo simulation with humanoid.xml
│   ├── api.py                # Flask API for backend-frontend communication
│   └── requirements.txt      # Python dependencies
├── frontend/                 # React frontend for GUI
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── MotionControl.js  # Controls for motion and simulation
│   │   ├── Visualizer.js    # Displays MuJoCo simulation output
│   │   └── index.js         # React entry point
│   ├── public/
│   │   └── index.html       # HTML template
│   └── package.json         # Node.js dependencies
├── humanoid.xml             # MuJoCo humanoid model (optional placement)
├── LICENSE.md               # MIT License file
└── README.md                # Project documentation
```

## Examples

### Example 1: Simulate Random Motion
- Start the backend (`python backend/api.py`).
- Open the GUI (`npm start` in `frontend/`).
- Click "Simulate Motion" to render a random humanoid pose.

### Example 2: Train LSTM on Dummy Data
- Run `python backend/lstm_model.py` standalone to train on random sequences.
- Check the console for training progress and saved model (`lstm_model.h5`).

### Example 3: Visualize Motion Data
- Modify `motion_tracker.py` to output joint angles to a CSV.
- Use Matplotlib in a custom script (e.g., `visualize.py`):
  ```python
  import matplotlib.pyplot as plt
  import pandas as pd
  data = pd.read_csv("motion_data.csv")
  plt.plot(data["time"], data["joint_1"])
  plt.show()
  ```

## Contributing

Contributions are encouraged! To contribute:

1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Submit a pull request.

For significant changes, open an issue first. Adhere to the project’s coding style and document new features.

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

## Contact

For inquiries or feedback, open an issue on GitHub or email [17arhaan.connect@gmail.com](mailto:17arhaan.connect@gmail.com).

---


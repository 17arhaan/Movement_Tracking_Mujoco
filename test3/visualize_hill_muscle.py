import mujoco
import mujoco.viewer
import numpy as np
import time
import os

class HillMuscle:
    def __init__(self, fmax=1000, lmin=0.1, lmax=0.3):
        self.fmax = fmax  # Maximum muscle force
        self.lmin = lmin  # Minimum muscle length
        self.lmax = lmax  # Maximum muscle length
        self.l0 = (lmin + lmax) / 2  # Optimal muscle length
        
    def compute_force(self, activation, length, velocity):
        # Length-tension relationship
        l_norm = (length - self.lmin) / (self.lmax - self.lmin)
        fl = np.exp(-((l_norm - 0.5) ** 2) / 0.1)  # Gaussian curve
        
        # Force-velocity relationship
        v_norm = velocity / self.l0
        fv = (1 - v_norm) / (1 + v_norm) if v_norm > 0 else (1 + v_norm) / (1 - v_norm)
        
        # Total force
        force = self.fmax * activation * fl * fv
        return force

def generate_emg_signal(t, frequency=1, phase=0, noise_level=0.1):
    """Generate EMG-like signal with noise and phase control"""
    # Base signal with phase control
    base = 0.5 * (1 + np.sin(2 * np.pi * frequency * t + phase))
    
    # Add noise
    noise = np.random.normal(0, noise_level)
    
    # Rectify and smooth
    emg = np.abs(base + noise)
    emg = np.clip(emg, 0, 1)
    
    return emg

# Print current working directory and check if file exists
print(f"Current directory: {os.getcwd()}")
xml_path = "muscle/hill_muscle_model.xml"
print(f"Looking for file: {xml_path}")
print(f"File exists: {os.path.exists(xml_path)}")

try:
    # Load the model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Initialize Hill muscles
    bicep = HillMuscle()
    tricep = HillMuscle()

    # Create viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera position for better view
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20

        # Initialize time
        t = 0
        dt = 0.01

        # Simulation loop
        while viewer.is_running():
            # Generate EMG-like signals with opposite phases for agonist/antagonist
            bicep_emg = generate_emg_signal(t, frequency=1, phase=0, noise_level=0.1)
            tricep_emg = generate_emg_signal(t, frequency=1, phase=np.pi, noise_level=0.1)
            
            # Compute muscle forces using Hill model with varying lengths
            # Length changes based on joint angle
            joint_angle = data.qpos[0]  # Get current joint angle
            length = 0.2 + 0.1 * np.sin(joint_angle)  # Length varies with joint angle
            velocity = data.qvel[0]  # Get current joint velocity
            
            bicep_force = bicep.compute_force(bicep_emg, length, velocity)
            tricep_force = tricep.compute_force(tricep_emg, length, -velocity)
            
            # Apply muscle forces through actuators with increased strength
            data.ctrl[0] = 1.5 * bicep_force / bicep.fmax  # Increased bicep force
            data.ctrl[1] = 1.5 * tricep_force / tricep.fmax  # Increased tricep force
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()
            
            # Update time
            t += dt
            
            # Small delay to make visualization smoother
            time.sleep(dt)

except Exception as e:
    print(f"Error occurred: {str(e)}")
    print("Please make sure:")
    print("1. The hill_muscle_model.xml file exists in the muscle directory")
    print("2. MuJoCo is properly installed")
    print("3. You have the necessary permissions to access the file") 
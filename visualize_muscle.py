import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# Print current working directory and check if file exists
print(f"Current directory: {os.getcwd()}")
xml_path = "muscle/muscle_model.xml"
print(f"Looking for file: {xml_path}")
print(f"File exists: {os.path.exists(xml_path)}")

try:
    # Load the model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Create viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera position for better view
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20

        # Simulation loop
        while viewer.is_running():
            # Create a more dramatic folding motion
            t = time.time()
            # Use a single sine wave with larger amplitude for dramatic folding
            motion = -0.8 * np.sin(t)  # Negative value to fold inward
            
            # Apply motor control with the folding motion
            data.ctrl[0] = motion
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()
            
            # Small delay to make visualization smoother
            time.sleep(0.01)

except Exception as e:
    print(f"Error occurred: {str(e)}")
    print("Please make sure:")
    print("1. The muscle_model.xml file exists in the muscle directory")
    print("2. MuJoCo is properly installed")
    print("3. You have the necessary permissions to access the file") 
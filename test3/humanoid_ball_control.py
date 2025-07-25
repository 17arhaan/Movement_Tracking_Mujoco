import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def compute_distance(pos1, pos2):
    """Compute Euclidean distance between two positions"""
    return np.sqrt(np.sum((pos1 - pos2) ** 2))

def generate_catching_motion(t, ball_pos, hand_pos):
    """Generate motion for catching the ball"""
    # Compute target hand position (slightly above ball)
    target_pos = ball_pos + np.array([0, 0, 0.1])
    
    # Compute direction to target
    direction = target_pos - hand_pos
    distance = compute_distance(target_pos, hand_pos)
    
    # Generate motion based on distance
    if distance > 0.5:
        # Move hands towards ball
        motion = direction / distance
    else:
        # Close hands to catch
        motion = np.array([0, 0, -1])
    
    return motion

def generate_carrying_motion(t, ball_pos, hand_pos):
    """Generate motion for carrying the ball"""
    # Move hands up and slightly forward
    target_pos = np.array([0.5, 0, 1.5])
    direction = target_pos - hand_pos
    distance = compute_distance(target_pos, hand_pos)
    
    if distance > 0.1:
        motion = direction / distance
    else:
        motion = np.zeros(3)
    
    return motion

# Print current working directory and check if file exists
print(f"Current directory: {os.getcwd()}")
xml_path = "humanoid_ball.xml"
print(f"Looking for file: {xml_path}")
print(f"File exists: {os.path.exists(xml_path)}")

try:
    # Load the model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Create viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera position for better view
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

        # Initialize time and state
        t = 0
        dt = 0.01
        state = "catching"  # States: "catching", "carrying"
        
        # Simulation loop
        while viewer.is_running():
            # Get current positions
            ball_pos = data.sensordata[0:3]  # Ball position
            left_hand_pos = data.sensordata[6:9]  # Left hand position
            right_hand_pos = data.sensordata[9:12]  # Right hand position
            
            # Generate motion based on state
            if state == "catching":
                # Generate catching motion
                left_motion = generate_catching_motion(t, ball_pos, left_hand_pos)
                right_motion = generate_catching_motion(t, ball_pos, right_hand_pos)
                
                # Check if ball is caught (close enough to hands)
                if (compute_distance(ball_pos, left_hand_pos) < 0.2 and 
                    compute_distance(ball_pos, right_hand_pos) < 0.2):
                    state = "carrying"
            else:
                # Generate carrying motion
                left_motion = generate_carrying_motion(t, ball_pos, left_hand_pos)
                right_motion = generate_carrying_motion(t, ball_pos, right_hand_pos)
            
            # Apply motion to arms
            # Left arm
            data.ctrl[3:6] = left_motion * 0.5  # Shoulder
            data.ctrl[6] = -0.5  # Elbow
            data.ctrl[7:10] = left_motion * 0.3  # Wrist
            
            # Right arm
            data.ctrl[10:13] = right_motion * 0.5  # Shoulder
            data.ctrl[13] = -0.5  # Elbow
            data.ctrl[14:17] = right_motion * 0.3  # Wrist
            
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
    print("1. The humanoid_ball.xml file exists")
    print("2. MuJoCo is properly installed")
    print("3. You have the necessary permissions to access the file") 
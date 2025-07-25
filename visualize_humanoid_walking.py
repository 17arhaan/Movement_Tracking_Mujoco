import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
import time

def visualize_humanoid():
    # Create the environment with human rendering
    gymenv = gym.make("Humanoid-v4", render_mode="human")
    
    # Create and train a new SAC model
    print("Creating new SAC model...")
    model = SAC("MlpPolicy", gymenv, verbose=1)
    
    # Train the model for a short duration to see some walking behavior
    print("Training model...")
    model.learn(total_timesteps=10000)
    
    # Reset the environment
    obs, _ = gymenv.reset()
    done = False
    truncated = False
    
    # Visualization loop
    while not (done or truncated):
        # Get action from the model
        action, _ = model.predict(obs, deterministic=True)
        
        # Take action in environment
        obs, _, done, truncated, _ = gymenv.step(action)
        
        # Small delay to make visualization smoother
        time.sleep(0.01)
    
    gymenv.close()

if __name__ == "__main__":
    visualize_humanoid() 
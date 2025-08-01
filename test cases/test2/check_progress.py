#!/usr/bin/env python3
"""
Check training progress and test models
"""

import os
import time
from simple_stand_walk import StandingHumanoidEnv, WalkingHumanoidEnv, test_model

def check_training_progress():
    """Check what models exist and test them"""
    print("🔍 CHECKING TRAINING PROGRESS")
    print("=" * 40)
    
    # Check if models exist
    models_to_check = [
        ("standing_model", StandingHumanoidEnv),
        ("walking_model", WalkingHumanoidEnv)
    ]
    
    for model_name, env_class in models_to_check:
        model_file = f"{model_name}.zip"
        if os.path.exists(model_file):
            print(f"✅ Found {model_file}")
            try:
                success = test_model(model_name, env_class, 200)
                if success:
                    print(f"🎉 {model_name} is working well!")
                else:
                    print(f"⚠️  {model_name} needs more training")
            except Exception as e:
                print(f"❌ {model_name} test failed: {e}")
        else:
            print(f"⏳ {model_file} not found yet (still training)")
    
    print("\n📁 Current directory contents:")
    files = [f for f in os.listdir('.') if f.endswith('.zip') or f.endswith('.py')]
    for f in sorted(files):
        size = os.path.getsize(f) / 1024  # KB
        print(f"  {f:30} ({size:6.1f} KB)")

if __name__ == "__main__":
    check_training_progress() 
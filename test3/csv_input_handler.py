import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CSVInputHandler:
    """
    Handles CSV inputs for anthropometric analysis including:
    - Custom anthropometric parameters
    - Trajectory data for comparison
    - Model configurations for batch analysis
    - Motion capture data
    """
    
    def __init__(self):
        self.supported_csv_types = {
            'anthropometric_params': self.load_anthropometric_params,
            'trajectory_data': self.load_trajectory_data,
            'model_config': self.load_model_config,
            'mocap_data': self.load_mocap_data,
            'joint_angles': self.load_joint_angles,
            'gait_data': self.load_gait_data
        }
        
        # Default column mappings for different CSV types
        self.column_mappings = {
            'anthropometric_params': {
                'required': ['parameter', 'value'],
                'optional': ['unit', 'description', 'min_range', 'max_range']
            },
            'trajectory_data': {
                'required': ['time', 'x', 'y', 'z'],
                'optional': ['joint_name', 'velocity_x', 'velocity_y', 'velocity_z']
            },
            'model_config': {
                'required': ['model_name', 'algorithm', 'environment'],
                'optional': ['episodes', 'max_steps', 'seed', 'description']
            },
            'joint_angles': {
                'required': ['time', 'joint_name', 'angle'],
                'optional': ['velocity', 'acceleration', 'torque']
            },
            'gait_data': {
                'required': ['time', 'step_length', 'walking_speed'],
                'optional': ['cadence', 'stance_time', 'swing_time', 'double_support']
            }
        }

    def auto_detect_csv_type(self, csv_path: str) -> str:
        """
        Automatically detect the type of CSV file based on columns.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Detected CSV type or 'unknown'
        """
        try:
            df = pd.read_csv(csv_path, nrows=5)  # Read first few rows
            columns = set(df.columns.str.lower())
            
            # Check for each CSV type
            for csv_type, mapping in self.column_mappings.items():
                required_cols = set([col.lower() for col in mapping['required']])
                if required_cols.issubset(columns):
                    return csv_type
                    
            return 'unknown'
        except Exception as e:
            print(f"❌ Error detecting CSV type: {e}")
            return 'unknown'

    def load_anthropometric_params(self, csv_path: str) -> Dict:
        """
        Load custom anthropometric parameters from CSV.
        
        Expected CSV format:
        parameter,value,unit,description,min_range,max_range
        height,1.75,m,Body height,1.50,2.00
        weight,70.0,kg,Body weight,50.0,120.0
        ...
        """
        try:
            df = pd.read_csv(csv_path)
            
            params = {}
            segment_lengths = {}
            joint_ranges = {}
            
            for _, row in df.iterrows():
                param = row['parameter'].lower()
                value = float(row['value'])
                
                if 'length' in param or 'segment' in param:
                    segment_lengths[param] = value
                elif 'range' in param or 'angle' in param:
                    joint_ranges[param] = value
                else:
                    params[param] = value
            
            anthropometric_data = {
                'basic_params': params,
                'segment_lengths': segment_lengths,
                'joint_ranges': joint_ranges,
                'source': 'CSV',
                'file_path': csv_path
            }
            
            print(f"✅ Loaded {len(df)} anthropometric parameters from CSV")
            return anthropometric_data
            
        except Exception as e:
            print(f"❌ Error loading anthropometric parameters: {e}")
            return {}

    def load_trajectory_data(self, csv_path: str) -> Dict:
        """
        Load trajectory data for comparison analysis.
        
        Expected CSV format:
        time,x,y,z,joint_name,velocity_x,velocity_y,velocity_z
        0.0,0.0,0.0,1.0,head,0.1,0.0,0.0
        0.01,0.001,0.0,1.001,head,0.1,0.0,0.05
        ...
        """
        try:
            df = pd.read_csv(csv_path)
            
            trajectory_data = {}
            
            if 'joint_name' in df.columns:
                # Group by joint
                for joint in df['joint_name'].unique():
                    joint_data = df[df['joint_name'] == joint]
                    trajectory_data[joint] = {
                        'time': joint_data['time'].values,
                        'position': {
                            'x': joint_data['x'].values,
                            'y': joint_data['y'].values,
                            'z': joint_data['z'].values
                        }
                    }
                    
                    # Add velocity if available
                    if all(col in joint_data.columns for col in ['velocity_x', 'velocity_y', 'velocity_z']):
                        trajectory_data[joint]['velocity'] = {
                            'x': joint_data['velocity_x'].values,
                            'y': joint_data['velocity_y'].values,
                            'z': joint_data['velocity_z'].values
                        }
            else:
                # Single trajectory
                trajectory_data['main'] = {
                    'time': df['time'].values,
                    'position': {
                        'x': df['x'].values,
                        'y': df['y'].values,
                        'z': df['z'].values
                    }
                }
            
            print(f"✅ Loaded trajectory data for {len(trajectory_data)} joints/objects")
            return trajectory_data
            
        except Exception as e:
            print(f"❌ Error loading trajectory data: {e}")
            return {}

    def load_model_config(self, csv_path: str) -> List[Dict]:
        """
        Load multiple model configurations for batch analysis.
        
        Expected CSV format:
        model_name,algorithm,environment,episodes,max_steps,seed,description
        A2C_1000000,A2C,Humanoid-v4,5,1000,42,Basic A2C model
        SAC_2000000,SAC,Humanoid-v4,3,1500,123,Advanced SAC model
        ...
        """
        try:
            df = pd.read_csv(csv_path)
            
            model_configs = []
            for _, row in df.iterrows():
                config = {
                    'model_name': row['model_name'],
                    'algorithm': row['algorithm'],
                    'environment': row['environment'],
                    'episodes': int(row.get('episodes', 1)),
                    'max_steps': int(row.get('max_steps', 1000)),
                    'seed': int(row.get('seed', 0)) if pd.notna(row.get('seed')) else None,
                    'description': row.get('description', '')
                }
                model_configs.append(config)
            
            print(f"✅ Loaded {len(model_configs)} model configurations")
            return model_configs
            
        except Exception as e:
            print(f"❌ Error loading model configurations: {e}")
            return []

    def load_joint_angles(self, csv_path: str) -> Dict:
        """
        Load joint angle data for biomechanical analysis.
        
        Expected CSV format:
        time,joint_name,angle,velocity,acceleration,torque
        0.0,hip_flexion,10.5,2.3,0.1,15.2
        0.01,hip_flexion,10.7,2.4,0.1,15.1
        ...
        """
        try:
            df = pd.read_csv(csv_path)
            
            joint_data = {}
            for joint in df['joint_name'].unique():
                joint_angles = df[df['joint_name'] == joint]
                
                joint_data[joint] = {
                    'time': joint_angles['time'].values,
                    'angle': joint_angles['angle'].values
                }
                
                # Add optional data if available
                for col in ['velocity', 'acceleration', 'torque']:
                    if col in joint_angles.columns:
                        joint_data[joint][col] = joint_angles[col].values
            
            print(f"✅ Loaded joint angle data for {len(joint_data)} joints")
            return joint_data
            
        except Exception as e:
            print(f"❌ Error loading joint angle data: {e}")
            return {}

    def load_gait_data(self, csv_path: str) -> Dict:
        """
        Load gait analysis data.
        
        Expected CSV format:
        time,step_length,walking_speed,cadence,stance_time,swing_time
        0.0,0.65,1.2,110,0.6,0.4
        1.0,0.67,1.25,112,0.58,0.42
        ...
        """
        try:
            df = pd.read_csv(csv_path)
            
            gait_data = {
                'time': df['time'].values,
                'step_length': df['step_length'].values,
                'walking_speed': df['walking_speed'].values
            }
            
            # Add optional gait parameters
            for col in ['cadence', 'stance_time', 'swing_time', 'double_support']:
                if col in df.columns:
                    gait_data[col] = df[col].values
            
            print(f"✅ Loaded gait data with {len(df)} time points")
            return gait_data
            
        except Exception as e:
            print(f"❌ Error loading gait data: {e}")
            return {}

    def load_mocap_data(self, csv_path: str) -> Dict:
        """
        Load motion capture data for comparison.
        
        Expected CSV format with marker positions:
        time,marker_name,x,y,z
        0.0,head_top,0.0,0.0,1.75
        0.0,shoulder_left,-0.2,0.0,1.4
        ...
        """
        try:
            df = pd.read_csv(csv_path)
            
            mocap_data = {}
            for marker in df['marker_name'].unique():
                marker_data = df[df['marker_name'] == marker]
                
                mocap_data[marker] = {
                    'time': marker_data['time'].values,
                    'position': {
                        'x': marker_data['x'].values,
                        'y': marker_data['y'].values,
                        'z': marker_data['z'].values
                    }
                }
            
            print(f"✅ Loaded motion capture data for {len(mocap_data)} markers")
            return mocap_data
            
        except Exception as e:
            print(f"❌ Error loading motion capture data: {e}")
            return {}

    def validate_csv_format(self, csv_path: str, csv_type: str) -> bool:
        """
        Validate that CSV has required columns for the specified type.
        
        Args:
            csv_path: Path to CSV file
            csv_type: Type of CSV (anthropometric_params, trajectory_data, etc.)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            df = pd.read_csv(csv_path, nrows=1)
            columns = set(df.columns.str.lower())
            
            if csv_type not in self.column_mappings:
                print(f"❌ Unknown CSV type: {csv_type}")
                return False
            
            required_cols = set([col.lower() for col in self.column_mappings[csv_type]['required']])
            missing_cols = required_cols - columns
            
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                return False
                
            print(f"✅ CSV format is valid for type: {csv_type}")
            return True
            
        except Exception as e:
            print(f"❌ Error validating CSV: {e}")
            return False

    def create_sample_csv_files(self, output_dir: str = "sample_csvs"):
        """
        Create sample CSV files showing the expected format for each type.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Anthropometric parameters sample
        anthro_data = {
            'parameter': ['height', 'weight', 'head_length', 'torso_length', 'hip_flexion_range', 'knee_flexion_range'],
            'value': [1.75, 70.0, 0.23, 0.50, 120.0, 140.0],
            'unit': ['m', 'kg', 'm', 'm', 'degrees', 'degrees'],
            'description': ['Total body height', 'Body weight', 'Head segment length', 'Torso length', 'Hip flexion ROM', 'Knee flexion ROM'],
            'min_range': [1.50, 50.0, 0.20, 0.40, 100.0, 120.0],
            'max_range': [2.00, 120.0, 0.30, 0.70, 140.0, 160.0]
        }
        pd.DataFrame(anthro_data).to_csv(f"{output_dir}/anthropometric_params_sample.csv", index=False)
        
        # 2. Trajectory data sample
        time_points = np.linspace(0, 2, 100)
        joint_heights = {'head': 0.7, 'torso': 0.3, 'hip': 0.0}
        traj_data = []
        for joint in ['head', 'torso', 'hip']:
            for i, t in enumerate(time_points):
                traj_data.append({
                    'time': t,
                    'x': 0.1 * np.sin(2 * np.pi * t),
                    'y': 0.05 * np.cos(2 * np.pi * t),
                    'z': 1.0 + joint_heights[joint] + 0.02 * np.sin(4 * np.pi * t),
                    'joint_name': joint,
                    'velocity_x': 0.1 * 2 * np.pi * np.cos(2 * np.pi * t),
                    'velocity_y': -0.05 * 2 * np.pi * np.sin(2 * np.pi * t),
                    'velocity_z': 0.02 * 4 * np.pi * np.cos(4 * np.pi * t)
                })
        pd.DataFrame(traj_data).to_csv(f"{output_dir}/trajectory_data_sample.csv", index=False)
        
        # 3. Model configuration sample
        model_config_data = {
            'model_name': ['A2C_1000000', 'A2C_2000000', 'SAC_1500000', 'TD3_2500000'],
            'algorithm': ['A2C', 'A2C', 'SAC', 'TD3'],
            'environment': ['Humanoid-v4', 'Humanoid-v4', 'Humanoid-v5', 'Humanoid-v4'],
            'episodes': [5, 3, 4, 2],
            'max_steps': [1000, 1500, 2000, 1200],
            'seed': [42, 123, 456, 789],
            'description': ['Basic A2C model', 'Improved A2C', 'SAC with v5 env', 'TD3 baseline']
        }
        pd.DataFrame(model_config_data).to_csv(f"{output_dir}/model_config_sample.csv", index=False)
        
        # 4. Joint angles sample
        joint_angle_data = []
        joints = ['hip_flexion', 'knee_flexion', 'ankle_flexion']
        for joint in joints:
            for i, t in enumerate(time_points[:50]):
                angle_base = {'hip_flexion': 20, 'knee_flexion': 10, 'ankle_flexion': 5}[joint]
                joint_angle_data.append({
                    'time': t,
                    'joint_name': joint,
                    'angle': angle_base + 30 * np.sin(2 * np.pi * t),
                    'velocity': 30 * 2 * np.pi * np.cos(2 * np.pi * t),
                    'acceleration': -30 * (2 * np.pi)**2 * np.sin(2 * np.pi * t),
                    'torque': 50 + 10 * np.sin(2 * np.pi * t + np.pi/4)
                })
        pd.DataFrame(joint_angle_data).to_csv(f"{output_dir}/joint_angles_sample.csv", index=False)
        
        # 5. Gait data sample
        gait_time = np.linspace(0, 10, 50)  # 10 seconds of gait
        gait_data = {
            'time': gait_time,
            'step_length': 0.65 + 0.05 * np.random.random(len(gait_time)),
            'walking_speed': 1.2 + 0.1 * np.random.random(len(gait_time)),
            'cadence': 110 + 5 * np.random.random(len(gait_time)),
            'stance_time': 0.6 + 0.05 * np.random.random(len(gait_time)),
            'swing_time': 0.4 + 0.03 * np.random.random(len(gait_time))
        }
        pd.DataFrame(gait_data).to_csv(f"{output_dir}/gait_data_sample.csv", index=False)
        
        print(f"✅ Sample CSV files created in '{output_dir}' directory:")
        print(f"   📄 anthropometric_params_sample.csv")
        print(f"   📄 trajectory_data_sample.csv")
        print(f"   📄 model_config_sample.csv")
        print(f"   📄 joint_angles_sample.csv")
        print(f"   📄 gait_data_sample.csv")

# Main interface functions for easy integration
def load_csv_data(csv_path: str, csv_type: str = None) -> Dict:
    """
    Main function to load CSV data of any supported type.
    
    Args:
        csv_path: Path to the CSV file
        csv_type: Type of CSV (auto-detected if None)
        
    Returns:
        Loaded data dictionary
    """
    handler = CSVInputHandler()
    
    if csv_type is None:
        csv_type = handler.auto_detect_csv_type(csv_path)
        print(f"🔍 Auto-detected CSV type: {csv_type}")
    
    if csv_type == 'unknown':
        print(f"❌ Could not determine CSV type. Please specify csv_type parameter.")
        return {}
    
    if not handler.validate_csv_format(csv_path, csv_type):
        return {}
    
    if csv_type in handler.supported_csv_types:
        return handler.supported_csv_types[csv_type](csv_path)
    else:
        print(f"❌ Unsupported CSV type: {csv_type}")
        return {}

def run_anthropometric_analysis_with_csv(model_name: str, 
                                       algorithm: str,
                                       environment: str,
                                       csv_inputs: Dict[str, str] = None,
                                       steps: int = 1000) -> Dict:
    """
    Run anthropometric analysis with custom CSV inputs.
    
    Args:
        model_name: Name of the model to analyze
        algorithm: RL algorithm (A2C, SAC, TD3)
        environment: Environment name
        csv_inputs: Dictionary mapping input types to CSV file paths
                   e.g., {'anthropometric_params': 'params.csv', 'trajectory_data': 'ref_traj.csv'}
        steps: Number of simulation steps
        
    Returns:
        Analysis results dictionary
    """
    from anthropometric_analysis import AnthropometricAnalyzer
    
    print(f"🔬 ENHANCED ANTHROPOMETRIC ANALYSIS WITH CSV INPUTS")
    print(f"{'='*80}")
    print(f"📊 Model: {model_name}")
    print(f"🤖 Algorithm: {algorithm}")
    print(f"🎮 Environment: {environment}")
    print(f"📄 CSV Inputs: {len(csv_inputs) if csv_inputs else 0}")
    print(f"{'='*80}")
    
    # Load CSV inputs
    loaded_data = {}
    if csv_inputs:
        for input_type, csv_path in csv_inputs.items():
            print(f"\n📄 Loading {input_type} from {csv_path}")
            data = load_csv_data(csv_path, input_type)
            if data:
                loaded_data[input_type] = data
    
    # Create analyzer with custom data
    analyzer = AnthropometricAnalyzer()
    
    # Override default parameters with CSV data if available
    if 'anthropometric_params' in loaded_data:
        custom_params = loaded_data['anthropometric_params']
        if 'basic_params' in custom_params:
            analyzer.anthropometric_data.update(custom_params['basic_params'])
        if 'segment_lengths' in custom_params:
            analyzer.anthropometric_data['segment_lengths'].update(custom_params['segment_lengths'])
        print(f"✅ Applied custom anthropometric parameters from CSV")
    
    # Run the standard analysis with loaded model
    try:
        import gymnasium as gym
        from stable_baselines3 import SAC, TD3, A2C
        
        model_path = f"models/{model_name}.zip"
        
        if algorithm == 'SAC':
            model = SAC.load(model_path)
        elif algorithm == 'TD3':
            model = TD3.load(model_path)
        elif algorithm == 'A2C':
            model = A2C.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        env = gym.make(environment, render_mode=None)
        
        # Run simulation and collect data
        obs, _ = env.reset()
        trajectory_data = []
        joint_data = []
        
        for step in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Collect trajectory and joint data
            # This would need to be adapted based on your specific environment
            # For now, we'll create placeholder data
            trajectory_data.append({
                'step': step,
                'reward': reward,
                'action': action,
                'observation': obs
            })
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        env.close()
        
        # Perform analysis
        analysis_results = analyzer.analyze_comprehensive(trajectory_data, loaded_data)
        
        # Add CSV comparison results if reference data was provided
        if 'trajectory_data' in loaded_data:
            comparison_results = analyzer.compare_with_reference(
                trajectory_data, loaded_data['trajectory_data']
            )
            analysis_results['csv_comparison'] = comparison_results
        
        # Generate final report
        print(f"\n🔬 CSV-ENHANCED ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"📊 Overall Score: {analysis_results.get('overall_score', 0):.1f}/100")
        print(f"📈 Compliance Level: {analysis_results.get('compliance_level', 'Unknown')}")
        
        if 'csv_comparison' in analysis_results:
            csv_score = analysis_results['csv_comparison'].get('similarity_score', 0)
            print(f"📄 CSV Reference Match: {csv_score:.1f}/100")
        
        print(f"{'='*60}")
        
        return analysis_results
        
    except Exception as e:
        print(f"❌ Error in CSV-enhanced analysis: {e}")
        return {}

# Utility function to create sample files
def create_sample_csv_files(output_dir: str = "sample_csvs"):
    """Create sample CSV files for testing."""
    handler = CSVInputHandler()
    handler.create_sample_csv_files(output_dir)

if __name__ == "__main__":
    # Create sample CSV files for demonstration
    create_sample_csv_files()
    print("\n💡 Usage examples:")
    print("   # Load anthropometric parameters:")
    print("   data = load_csv_data('params.csv', 'anthropometric_params')")
    print("\n   # Run analysis with CSV inputs:")
    print("   results = run_anthropometric_analysis_with_csv(")
    print("       'A2C_1000000', 'A2C', 'Humanoid-v4',")
    print("       {'anthropometric_params': 'params.csv'})") 
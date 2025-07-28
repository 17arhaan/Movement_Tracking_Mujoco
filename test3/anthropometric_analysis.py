import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import euclidean
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AnthropometricAnalyzer:
    """
    Analyzes humanoid trajectories against human anthropometric parameters
    to determine if generated movements are biomechanically realistic.
    """
    
    def __init__(self):
        # Standard human anthropometric parameters (in meters)
        # Based on 50th percentile adult male data
        self.anthropometric_data = {
            'height': 1.75,  # Total body height
            'weight': 70.0,  # Body weight in kg
            'segment_lengths': {
                'head_neck': 0.25,
                'torso': 0.55,
                'upper_arm': 0.32,
                'forearm': 0.26,
                'hand': 0.19,
                'thigh': 0.43,
                'shank': 0.43,
                'foot': 0.26
            },
            'joint_ranges': {
                'hip_flexion': (-120, 120),      # degrees
                'hip_extension': (-10, 30),
                'knee_flexion': (0, 140),
                'ankle_dorsiflexion': (-20, 30),
                'shoulder_flexion': (0, 180),
                'shoulder_abduction': (0, 180),
                'elbow_flexion': (0, 150),
                'wrist_flexion': (-80, 80)
            },
            'gait_parameters': {
                'step_length': 0.7,      # meters
                'step_width': 0.15,      # meters
                'cadence': 110,          # steps/minute
                'stride_time': 1.1,      # seconds
                'double_support': 0.22   # % of gait cycle
            },
            'kinematic_limits': {
                'max_walking_speed': 1.4,    # m/s
                'max_running_speed': 3.0,    # m/s
                'max_jump_height': 0.5,      # meters
                'max_reach_height': 2.1,     # meters
                'max_reach_distance': 0.8    # meters
            }
        }
        
        # Initialize analysis results
        self.analysis_results = {}
        
    def analyze_joint_angles(self, joint_data: Dict[str, List[float]]) -> Dict[str, Dict]:
        """
        Analyze joint angles against human ROM (Range of Motion) limits.
        
        Args:
            joint_data: Dictionary with joint names as keys and angle trajectories as values
            
        Returns:
            Dictionary with analysis results for each joint
        """
        results = {}
        
        for joint_name, angles in joint_data.items():
            if joint_name in self.anthropometric_data['joint_ranges']:
                min_angle, max_angle = self.anthropometric_data['joint_ranges'][joint_name]
                
                # Convert angles to degrees if in radians
                angles_deg = np.array(angles) * 180 / np.pi if np.max(angles) < 10 else np.array(angles)
                
                # Calculate statistics
                mean_angle = np.mean(angles_deg)
                max_observed = np.max(angles_deg)
                min_observed = np.min(angles_deg)
                range_observed = max_observed - min_observed
                
                # Check if within human limits
                within_limits = (min_observed >= min_angle) and (max_observed <= max_angle)
                
                # Calculate percentage of trajectory within limits
                within_limits_pct = np.mean((angles_deg >= min_angle) & (angles_deg <= max_angle)) * 100
                
                results[joint_name] = {
                    'mean_angle': mean_angle,
                    'max_observed': max_observed,
                    'min_observed': min_observed,
                    'range_observed': range_observed,
                    'human_min': min_angle,
                    'human_max': max_angle,
                    'within_limits': within_limits,
                    'within_limits_percentage': within_limits_pct,
                    'realistic_score': min(100, within_limits_pct)
                }
        
        return results
    
    def analyze_gait_parameters(self, trajectory_data: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Analyze walking gait parameters against human norms.
        
        Args:
            trajectory_data: Dictionary with position data for key body parts
            
        Returns:
            Dictionary with gait analysis results
        """
        results = {}
        
        # Extract relevant trajectory data
        if 'pelvis' in trajectory_data and 'left_foot' in trajectory_data and 'right_foot' in trajectory_data:
            pelvis_pos = trajectory_data['pelvis']
            left_foot_pos = trajectory_data['left_foot']
            right_foot_pos = trajectory_data['right_foot']
            
            # Calculate step length
            step_lengths = []
            for i in range(1, len(pelvis_pos)):
                step_length = euclidean(left_foot_pos[i], right_foot_pos[i-1])
                step_lengths.append(step_length)
            
            mean_step_length = np.mean(step_lengths)
            human_step_length = self.anthropometric_data['gait_parameters']['step_length']
            
            # Calculate walking speed
            if len(pelvis_pos) > 1:
                total_distance = np.sum([euclidean(pelvis_pos[i], pelvis_pos[i-1]) for i in range(1, len(pelvis_pos))])
                total_time = len(pelvis_pos) * 0.01  # Assuming 100Hz sampling
                walking_speed = total_distance / total_time
            else:
                walking_speed = 0
            
            # Analyze against human limits
            max_walking_speed = self.anthropometric_data['kinematic_limits']['max_walking_speed']
            
            results['gait_analysis'] = {
                'mean_step_length': mean_step_length,
                'human_step_length': human_step_length,
                'step_length_realistic': abs(mean_step_length - human_step_length) < 0.2,
                'walking_speed': walking_speed,
                'max_human_speed': max_walking_speed,
                'speed_realistic': walking_speed <= max_walking_speed,
                'gait_realistic_score': self._calculate_gait_score(mean_step_length, walking_speed)
            }
        
        return results
    
    def analyze_reach_kinematics(self, trajectory_data: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Analyze reaching movements against human reach capabilities.
        
        Args:
            trajectory_data: Dictionary with position data for arms and targets
            
        Returns:
            Dictionary with reach analysis results
        """
        results = {}
        
        if 'right_hand' in trajectory_data and 'target' in trajectory_data:
            hand_pos = trajectory_data['right_hand']
            target_pos = trajectory_data['target']
            
            # Calculate reach distances
            reach_distances = []
            for i in range(len(hand_pos)):
                distance = euclidean(hand_pos[i], target_pos[i])
                reach_distances.append(distance)
            
            max_reach_distance = np.max(reach_distances)
            human_max_reach = self.anthropometric_data['kinematic_limits']['max_reach_distance']
            
            # Calculate reach accuracy
            final_distance = reach_distances[-1] if reach_distances else float('inf')
            reach_accuracy = 1.0 / (1.0 + final_distance)  # Higher accuracy for smaller final distance
            
            results['reach_analysis'] = {
                'max_reach_distance': max_reach_distance,
                'human_max_reach': human_max_reach,
                'reach_realistic': max_reach_distance <= human_max_reach,
                'final_distance': final_distance,
                'reach_accuracy': reach_accuracy,
                'reach_realistic_score': min(100, (human_max_reach - max_reach_distance) / human_max_reach * 100)
            }
        
        return results
    
    def analyze_movement_smoothness(self, trajectory_data: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Analyze movement smoothness using jerk analysis.
        
        Args:
            trajectory_data: Dictionary with position data for key body parts
            
        Returns:
            Dictionary with smoothness analysis results
        """
        results = {}
        
        for body_part, positions in trajectory_data.items():
            if len(positions) > 3:
                # Calculate velocity
                velocities = np.diff(positions, axis=0)
                
                # Calculate acceleration
                accelerations = np.diff(velocities, axis=0)
                
                # Calculate jerk (rate of change of acceleration)
                jerks = np.diff(accelerations, axis=0)
                
                # Calculate smoothness metrics
                mean_jerk = np.mean(np.linalg.norm(jerks, axis=1))
                jerk_variance = np.var(np.linalg.norm(jerks, axis=1))
                
                # Human movements typically have low jerk
                jerk_threshold = 10.0  # Arbitrary threshold for smooth movement
                is_smooth = mean_jerk < jerk_threshold
                
                results[f'{body_part}_smoothness'] = {
                    'mean_jerk': mean_jerk,
                    'jerk_variance': jerk_variance,
                    'is_smooth': is_smooth,
                    'smoothness_score': max(0, 100 - mean_jerk * 10)
                }
        
        return results
    
    def analyze_anthropometric_compliance(self, trajectory_data: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Overall analysis of anthropometric compliance.
        
        Args:
            trajectory_data: Dictionary with all trajectory data
            
        Returns:
            Comprehensive analysis results
        """
        # Perform all analyses
        joint_analysis = self.analyze_joint_angles(trajectory_data.get('joint_angles', {}))
        gait_analysis = self.analyze_gait_parameters(trajectory_data)
        reach_analysis = self.analyze_reach_kinematics(trajectory_data)
        smoothness_analysis = self.analyze_movement_smoothness(trajectory_data)
        
        # Calculate overall compliance score
        scores = []
        
        # Joint angle compliance
        if joint_analysis:
            joint_scores = [result['realistic_score'] for result in joint_analysis.values()]
            scores.extend(joint_scores)
        
        # Gait compliance
        if 'gait_analysis' in gait_analysis:
            scores.append(gait_analysis['gait_analysis']['gait_realistic_score'])
        
        # Reach compliance
        if 'reach_analysis' in reach_analysis:
            scores.append(reach_analysis['reach_analysis']['reach_realistic_score'])
        
        # Smoothness compliance
        if smoothness_analysis:
            smoothness_scores = [result['smoothness_score'] for result in smoothness_analysis.values()]
            scores.extend(smoothness_scores)
        
        overall_score = np.mean(scores) if scores else 0
        
        # Determine compliance level
        if overall_score >= 80:
            compliance_level = "Excellent"
        elif overall_score >= 60:
            compliance_level = "Good"
        elif overall_score >= 40:
            compliance_level = "Fair"
        else:
            compliance_level = "Poor"
        
        return {
            'joint_analysis': joint_analysis,
            'gait_analysis': gait_analysis,
            'reach_analysis': reach_analysis,
            'smoothness_analysis': smoothness_analysis,
            'overall_score': overall_score,
            'compliance_level': compliance_level,
            'total_analyses': len(scores)
        }
    
    def _calculate_gait_score(self, step_length: float, walking_speed: float) -> float:
        """Calculate realistic gait score based on step length and speed."""
        human_step_length = self.anthropometric_data['gait_parameters']['step_length']
        max_speed = self.anthropometric_data['kinematic_limits']['max_walking_speed']
        
        step_score = max(0, 100 - abs(step_length - human_step_length) / human_step_length * 100)
        speed_score = max(0, 100 - (walking_speed / max_speed) * 100)
        
        return (step_score + speed_score) / 2
    
    def print_analysis_report(self, analysis_results: Dict[str, Dict]) -> None:
        """
        Print comprehensive anthropometric analysis report.
        
        Args:
            analysis_results: Results from analyze_anthropometric_compliance
        """
        print("\n" + "="*80)
        print("🔬 ANTHROPOMETRIC TRAJECTORY ANALYSIS REPORT")
        print("="*80)
        
        # Overall compliance
        overall_score = analysis_results.get('overall_score', 0)
        compliance_level = analysis_results.get('compliance_level', 'Unknown')
        total_analyses = analysis_results.get('total_analyses', 0)
        
        print(f"\n📊 OVERALL COMPLIANCE:")
        print(f"   🎯 Overall Score: {overall_score:.1f}/100")
        print(f"   📈 Compliance Level: {compliance_level}")
        print(f"   🔍 Total Analyses: {total_analyses}")
        
        # Joint analysis
        if 'joint_analysis' in analysis_results:
            print(f"\n🦴 JOINT ANGLE ANALYSIS:")
            joint_results = analysis_results['joint_analysis']
            for joint, result in joint_results.items():
                status = "✅" if result['within_limits'] else "❌"
                print(f"   {status} {joint}: {result['realistic_score']:.1f}% realistic")
                print(f"      Range: {result['min_observed']:.1f}° to {result['max_observed']:.1f}°")
                print(f"      Human: {result['human_min']:.1f}° to {result['human_max']:.1f}°")
        
        # Gait analysis
        if 'gait_analysis' in analysis_results and 'gait_analysis' in analysis_results['gait_analysis']:
            gait = analysis_results['gait_analysis']['gait_analysis']
            print(f"\n🚶 GAIT ANALYSIS:")
            print(f"   📏 Step Length: {gait['mean_step_length']:.3f}m (Human: {gait['human_step_length']:.3f}m)")
            print(f"   🏃 Walking Speed: {gait['walking_speed']:.2f}m/s (Max: {gait['max_human_speed']:.2f}m/s)")
            print(f"   🎯 Gait Realistic Score: {gait['gait_realistic_score']:.1f}/100")
        
        # Reach analysis
        if 'reach_analysis' in analysis_results and 'reach_analysis' in analysis_results['reach_analysis']:
            reach = analysis_results['reach_analysis']['reach_analysis']
            print(f"\n🤲 REACH ANALYSIS:")
            print(f"   📏 Max Reach: {reach['max_reach_distance']:.3f}m (Human: {reach['human_max_reach']:.3f}m)")
            print(f"   🎯 Reach Accuracy: {reach['reach_accuracy']:.3f}")
            print(f"   📊 Reach Realistic Score: {reach['reach_realistic_score']:.1f}/100")
        
        # Smoothness analysis
        if 'smoothness_analysis' in analysis_results:
            print(f"\n✨ MOVEMENT SMOOTHNESS:")
            smoothness_results = analysis_results['smoothness_analysis']
            for body_part, result in smoothness_results.items():
                status = "✅" if result['is_smooth'] else "❌"
                print(f"   {status} {body_part}: {result['smoothness_score']:.1f}/100")
                print(f"      Mean Jerk: {result['mean_jerk']:.3f}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if overall_score >= 80:
            print("   ✅ Excellent anthropometric compliance! Model generates realistic human-like movements.")
        elif overall_score >= 60:
            print("   ⚠️  Good compliance with minor deviations. Consider fine-tuning joint limits.")
        elif overall_score >= 40:
            print("   ⚠️  Fair compliance with significant deviations. Review movement constraints.")
        else:
            print("   ❌ Poor compliance. Model movements may not be anthropometrically realistic.")
        
        print("\n" + "="*80)
    
    def generate_visualization(self, analysis_results: Dict[str, Dict]) -> None:
        """
        Generate visualization of anthropometric analysis results.
        
        Args:
            analysis_results: Results from analyze_anthropometric_compliance
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Anthropometric Trajectory Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall compliance score
        overall_score = analysis_results.get('overall_score', 0)
        axes[0,0].bar(['Overall Compliance'], [overall_score], color='skyblue', alpha=0.7)
        axes[0,0].set_ylim(0, 100)
        axes[0,0].set_title('Overall Anthropometric Compliance')
        axes[0,0].set_ylabel('Score (%)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Joint angle compliance
        if 'joint_analysis' in analysis_results:
            joints = list(analysis_results['joint_analysis'].keys())
            scores = [analysis_results['joint_analysis'][j]['realistic_score'] for j in joints]
            
            axes[0,1].bar(joints, scores, color='lightgreen', alpha=0.7)
            axes[0,1].set_title('Joint Angle Compliance')
            axes[0,1].set_ylabel('Realistic Score (%)')
            axes[0,1].tick_params(axis='x', rotation=45)
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Gait parameters
        if 'gait_analysis' in analysis_results and 'gait_analysis' in analysis_results['gait_analysis']:
            gait = analysis_results['gait_analysis']['gait_analysis']
            gait_params = ['Step Length', 'Walking Speed']
            human_values = [gait['human_step_length'], gait['max_human_speed']]
            observed_values = [gait['mean_step_length'], gait['walking_speed']]
            
            x = np.arange(len(gait_params))
            width = 0.35
            
            axes[1,0].bar(x - width/2, human_values, width, label='Human Norm', alpha=0.7)
            axes[1,0].bar(x + width/2, observed_values, width, label='Observed', alpha=0.7)
            axes[1,0].set_title('Gait Parameters Comparison')
            axes[1,0].set_ylabel('Value')
            axes[1,0].set_xticks(x)
            axes[1,0].set_xticklabels(gait_params)
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Smoothness analysis
        if 'smoothness_analysis' in analysis_results:
            smoothness_results = analysis_results['smoothness_analysis']
            body_parts = list(smoothness_results.keys())
            smoothness_scores = [smoothness_results[bp]['smoothness_score'] for bp in body_parts]
            
            axes[1,1].bar(body_parts, smoothness_scores, color='orange', alpha=0.7)
            axes[1,1].set_title('Movement Smoothness')
            axes[1,1].set_ylabel('Smoothness Score (%)')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def extract_trajectory_data_from_env(env, model, steps=1000):
    """
    Extract trajectory data from environment for anthropometric analysis.
    
    Args:
        env: Gymnasium environment
        model: Trained RL model
        steps: Number of steps to simulate
        
    Returns:
        Dictionary with trajectory data
    """
    trajectory_data = {
        'joint_angles': {},
        'pelvis': [],
        'left_foot': [],
        'right_foot': [],
        'right_hand': [],
        'target': []
    }
    
    obs, _ = env.reset()
    
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(action)
        
        # Extract joint angles (assuming first 17 values are joint angles)
        if len(obs) >= 17:
            joint_names = [
                'hip_flexion', 'hip_abduction', 'hip_rotation',
                'knee_flexion', 'ankle_flexion', 'ankle_inversion',
                'shoulder_flexion', 'shoulder_abduction', 'shoulder_rotation',
                'elbow_flexion', 'wrist_flexion', 'wrist_abduction',
                'neck_flexion', 'neck_rotation', 'lumbar_flexion',
                'lumbar_rotation', 'lumbar_lateral'
            ]
            
            for i, joint_name in enumerate(joint_names):
                if joint_name not in trajectory_data['joint_angles']:
                    trajectory_data['joint_angles'][joint_name] = []
                trajectory_data['joint_angles'][joint_name].append(obs[i])
        
        # Extract body part positions (simplified - would need actual body part tracking)
        # This is a placeholder - in practice you'd extract from env state
        if len(obs) >= 17:
            # Simulate body part positions based on joint angles
            pelvis_pos = [obs[0], obs[1], obs[2]]  # Simplified
            left_foot_pos = [obs[3], obs[4], obs[5]]
            right_foot_pos = [obs[6], obs[7], obs[8]]
            right_hand_pos = [obs[9], obs[10], obs[11]]
            target_pos = [0.5, 0, 1.0]  # Fixed target position
            
            trajectory_data['pelvis'].append(pelvis_pos)
            trajectory_data['left_foot'].append(left_foot_pos)
            trajectory_data['right_foot'].append(right_foot_pos)
            trajectory_data['right_hand'].append(right_hand_pos)
            trajectory_data['target'].append(target_pos)
        
        if done or truncated:
            break
    
    # Convert to numpy arrays
    for key in trajectory_data:
        if isinstance(trajectory_data[key], list):
            trajectory_data[key] = np.array(trajectory_data[key])
        elif isinstance(trajectory_data[key], dict):
            for joint_key in trajectory_data[key]:
                trajectory_data[key][joint_key] = np.array(trajectory_data[key][joint_key])
    
    return trajectory_data

# Example usage function
def run_anthropometric_analysis(env, model, steps=1000):
    """
    Run complete anthropometric analysis on model-generated trajectories.
    
    Args:
        env: Gymnasium environment
        model: Trained RL model
        steps: Number of steps to simulate
    """
    print("🔬 Starting Anthropometric Analysis...")
    
    # Extract trajectory data
    trajectory_data = extract_trajectory_data_from_env(env, model, steps)
    
    # Initialize analyzer
    analyzer = AnthropometricAnalyzer()
    
    # Perform analysis
    analysis_results = analyzer.analyze_anthropometric_compliance(trajectory_data)
    
    # Print report
    analyzer.print_analysis_report(analysis_results)
    
    # Generate visualization
    analyzer.generate_visualization(analysis_results)
    
    return analysis_results 
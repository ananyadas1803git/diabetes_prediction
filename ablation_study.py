"""
Ablation Study Script for Diabetes Prediction Framework

This script runs three ablation study configurations:
1. XGBoost Only (baseline)
2. Diffusion + XGBoost
3. TSO-HBA + XGBoost (full pipeline)
"""

import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from main import main as run_pipeline


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str) -> None:
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_xgboost_only(config: dict) -> dict:
    """Configuration 1: XGBoost Only (no augmentation, no optimization)"""
    print("\n" + "="*80)
    print("ABLATION STUDY - Configuration 1: XGBoost Only")
    print("="*80)
    
    # Disable all augmentation
    config['diffusion']['use_synthetic_augmentation'] = False
    config['data']['augmentation_method'] = 'none'
    
    # Use default XGBoost parameters (no optimization)
    config['optimization']['optimizer_type'] = 'default'
    
    # Save temporary config
    temp_config = "config_xgboost_only.yaml"
    save_config(config, temp_config)
    
    try:
        # Run pipeline
        run_pipeline(config_path=temp_config, run_baselines=False, optimizer_override='default')
        return {"status": "success", "config": "XGBoost Only"}
    except Exception as e:
        print(f"Error in XGBoost Only: {e}")
        return {"status": "failed", "config": "XGBoost Only", "error": str(e)}
    finally:
        # Cleanup
        if Path(temp_config).exists():
            Path(temp_config).unlink()


def run_diffusion_xgboost(config: dict) -> dict:
    """Configuration 2: Diffusion + XGBoost (with diffusion, no optimization)"""
    print("\n" + "="*80)
    print("ABLATION STUDY - Configuration 2: Diffusion + XGBoost")
    print("="*80)
    
    # Enable diffusion
    config['diffusion']['use_synthetic_augmentation'] = True
    config['data']['augmentation_method'] = 'diffusion'
    
    # Use default XGBoost parameters (no optimization)
    config['optimization']['optimizer_type'] = 'default'
    
    # Save temporary config
    temp_config = "config_diffusion_xgboost.yaml"
    save_config(config, temp_config)
    
    try:
        # Run pipeline
        run_pipeline(config_path=temp_config, run_baselines=False, optimizer_override='default')
        return {"status": "success", "config": "Diffusion + XGBoost"}
    except Exception as e:
        print(f"Error in Diffusion + XGBoost: {e}")
        return {"status": "failed", "config": "Diffusion + XGBoost", "error": str(e)}
    finally:
        # Cleanup
        if Path(temp_config).exists():
            Path(temp_config).unlink()


def run_smote_xgboost(config: dict) -> dict:
    """Configuration 3: SMOTE + XGBoost (with SMOTE, no optimization)"""
    print("\n" + "="*80)
    print("ABLATION STUDY - Configuration 3: SMOTE + XGBoost")
    print("="*80)
    
    # Disable diffusion, enable SMOTE
    config['diffusion']['use_synthetic_augmentation'] = False
    config['data']['augmentation_method'] = 'smote'
    
    # Use default XGBoost parameters (no optimization)
    config['optimization']['optimizer_type'] = 'default'
    
    # Save temporary config
    temp_config = "config_smote_xgboost.yaml"
    save_config(config, temp_config)
    
    try:
        # Run pipeline
        run_pipeline(config_path=temp_config, run_baselines=False, optimizer_override='default')
        return {"status": "success", "config": "SMOTE + XGBoost"}
    except Exception as e:
        print(f"Error in SMOTE + XGBoost: {e}")
        return {"status": "failed", "config": "SMOTE + XGBoost", "error": str(e)}
    finally:
        # Cleanup
        if Path(temp_config).exists():
            Path(temp_config).unlink()


def run_adasyn_xgboost(config: dict) -> dict:
    """Configuration 4: ADASYN + XGBoost (with ADASYN, no optimization)"""
    print("\n" + "="*80)
    print("ABLATION STUDY - Configuration 4: ADASYN + XGBoost")
    print("="*80)
    
    # Disable diffusion, enable ADASYN
    config['diffusion']['use_synthetic_augmentation'] = False
    config['data']['augmentation_method'] = 'adasyn'
    
    # Use default XGBoost parameters (no optimization)
    config['optimization']['optimizer_type'] = 'default'
    
    # Save temporary config
    temp_config = "config_adasyn_xgboost.yaml"
    save_config(config, temp_config)
    
    try:
        # Run pipeline
        run_pipeline(config_path=temp_config, run_baselines=False, optimizer_override='default')
        return {"status": "success", "config": "ADASYN + XGBoost"}
    except Exception as e:
        print(f"Error in ADASYN + XGBoost: {e}")
        return {"status": "failed", "config": "ADASYN + XGBoost", "error": str(e)}
    finally:
        # Cleanup
        if Path(temp_config).exists():
            Path(temp_config).unlink()


def run_tso_hba_xgboost(config: dict, optimizer_type: str = 'tso_hba') -> dict:
    """Configuration 3: TSO-HBA + XGBoost (full pipeline with optimization)"""
    print("\n" + "="*80)
    print(f"ABLATION STUDY - Configuration 3: {optimizer_type.upper()} + XGBoost")
    print("="*80)
    
    # Enable diffusion
    config['diffusion']['use_synthetic_augmentation'] = True
    
    # Use optimization
    config['optimization']['optimizer_type'] = optimizer_type
    
    # Save temporary config
    temp_config = f"config_{optimizer_type}_xgboost.yaml"
    save_config(config, temp_config)
    
    try:
        # Run pipeline
        run_pipeline(config_path=temp_config, run_baselines=False, optimizer_override=optimizer_type)
        return {"status": "success", "config": f"{optimizer_type.upper()} + XGBoost"}
    except Exception as e:
        print(f"Error in {optimizer_type.upper()} + XGBoost: {e}")
        return {"status": "failed", "config": f"{optimizer_type.upper()} + XGBoost", "error": str(e)}
    finally:
        # Cleanup
        if Path(temp_config).exists():
            Path(temp_config).unlink()


def collect_results() -> pd.DataFrame:
    """Collect results from log files or model evaluation outputs."""
    results = []
    
    # This is a placeholder - in practice, you'd parse the actual output
    # For now, we'll create a structure that can be filled manually or via log parsing
    
    results = {
        "Configuration": ["XGBoost Only", "Diffusion + XGBoost", "SMOTE + XGBoost", "ADASYN + XGBoost", "TSO-HBA + XGBoost"],
        "Accuracy": [None, None, None, None, None],
        "Precision": [None, None, None, None, None],
        "Recall": [None, None, None, None, None],
        "F1": [None, None, None, None, None],
        "ROC-AUC": [None, None, None, None, None],
        "Status": ["pending", "pending", "pending", "pending", "pending"]
    }
    
    return pd.DataFrame(results)


def save_ablation_results(results: pd.DataFrame, output_path: str = "results/ablation_study_results.csv") -> None:
    """Save ablation study results to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nAblation study results saved to {output_path}")


def main(config_path: str = "config.yaml", configurations: list = None):
    """Run complete ablation study."""
    
    # Load base configuration
    config = load_config(config_path)
    
    # Define configurations to run
    if configurations is None:
        configurations = ["xgboost_only", "diffusion_xgboost", "smote_xgboost", "adasyn_xgboost", "tso_hba_xgboost"]
    
    # Track results
    study_results = []
    
    # Run each configuration
    for config_name in configurations:
        if config_name == "xgboost_only":
            result = run_xgboost_only(config.copy())
        elif config_name == "diffusion_xgboost":
            result = run_diffusion_xgboost(config.copy())
        elif config_name == "smote_xgboost":
            result = run_smote_xgboost(config.copy())
        elif config_name == "adasyn_xgboost":
            result = run_adasyn_xgboost(config.copy())
        elif config_name == "tso_hba_xgboost":
            result = run_tso_hba_xgboost(config.copy(), optimizer_type='tso_hba')
        elif config_name == "optuna_xgboost":
            result = run_tso_hba_xgboost(config.copy(), optimizer_type='optuna')
        else:
            print(f"Unknown configuration: {config_name}")
            continue
        
        study_results.append(result)
    
    # Collect and save results
    results_df = collect_results()
    
    # Update status based on run results
    for i, result in enumerate(study_results):
        if i < len(results_df):
            results_df.loc[i, "Status"] = result["status"]
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/ablation_study_{timestamp}.csv"
    save_ablation_results(results_df, output_path)
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(results_df.to_string())
    print("="*80)
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study for Diabetes Prediction")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--configs", type=str, nargs='+', 
                       choices=["xgboost_only", "diffusion_xgboost", "smote_xgboost", "adasyn_xgboost", "tso_hba_xgboost", "optuna_xgboost"],
                       default=["xgboost_only", "diffusion_xgboost", "smote_xgboost", "adasyn_xgboost"],
                       help="Configurations to run")
    
    args = parser.parse_args()
    
    main(config_path=args.config, configurations=args.configs)

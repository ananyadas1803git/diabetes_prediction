"""
Main Pipeline Script for Diabetes Prediction Framework

This script orchestrates the complete pipeline:
1. Data preprocessing and EDA
2. Tabular diffusion data augmentation
3. Feature selection
4. TSO-HBA optimization
5. XGBoost model training
6. Model evaluation
7. SHAP explainability
8. Baseline comparisons
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent))
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.preprocessing.preprocessor import DataPreprocessor
from src.preprocessing.eda import EDAAnalyzer
from src.diffusion.tabular_diffusion import TabularDiffusion
from src.feature_selection.feature_selector import FeatureSelector
from src.optimization.tso_hba import TSOHBAOptimizer
from src.models.xgboost_model import XGBoostModel
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.baseline_comparison import BaselineComparison
from src.explainability.shap_explainer import SHAPExplainer
def main(config_path: str = "config.yaml", run_baselines: bool = False):
    config = load_config(config_path)
    logger = setup_logger(
        "diabetes_predictor",
        level=config['logging']['level'],
        log_file=config['logging']['log_file'],
        format_string=config['logging']['format']
    )
    logger.info("=" * 80)
    logger.info("Starting Diabetes Prediction Framework Pipeline")
    logger.info("=" * 80)
    set_seed(config['data']['random_state'])
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Data Preprocessing")
    logger.info("=" * 80)
    preprocessor = DataPreprocessor(config, logger)
    data_path = config['data']['raw_path']
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please place the Pima Indians Diabetes dataset at data/raw/diabetes.csv")
        return
    X_train, X_test, y_train, y_test, train_df, test_df = preprocessor.preprocess(
        input_path=data_path,
        output_path=config['data']['processed_path']
    )
    feature_names = preprocessor.get_feature_names()
    logger.info(f"Features: {feature_names}")
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Exploratory Data Analysis")
    logger.info("=" * 80)
    eda_analyzer = EDAAnalyzer(config, logger)
    eda_analyzer.run_full_eda(train_df)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Tabular Diffusion Data Augmentation")
    logger.info("=" * 80)
    diffusion_model = TabularDiffusion(config, logger)
    diffusion_model.train(X_train, feature_names)
    num_samples = config['diffusion']['num_synthetic_samples']
    synthetic_df = diffusion_model.generate_balanced(num_samples // 2)
    diffusion_model.save_synthetic_data(synthetic_df, config['data']['synthetic_path'])
    X_train_augmented = np.vstack([X_train, synthetic_df[feature_names].values])
    y_train_augmented = np.hstack([y_train, synthetic_df['Outcome'].values])
    logger.info(f"Using augmented training data: {X_train.shape[0]} original + {synthetic_df.shape[0]} synthetic = {X_train_augmented.shape[0]} total")
    X_train_final = X_train_augmented
    y_train_final = y_train_augmented
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Feature Selection")
    logger.info("=" * 80)
    feature_selector = FeatureSelector(config, logger)
    selected_features, X_train_selected = feature_selector.select_features(
        X_train_final, y_train_final, feature_names, method='ensemble'
    )
    logger.info(f"Selected features: {selected_features}")
    feature_report = feature_selector.generate_feature_report()
    report_path = Path(config['evaluation']['plots_path']) / "feature_selection_report.csv"
    feature_report.to_csv(report_path, index=False)
    logger.info(f"Feature selection report saved to {report_path}")
    selected_indices = [feature_names.index(f) for f in selected_features]
    X_test_selected = X_test[:, selected_indices]
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: TSO-HBA Hyperparameter Optimization")
    logger.info("=" * 80)
    optimizer = TSOHBAOptimizer(config, logger)
    best_params = optimizer.optimize(X_train_selected, y_train_final)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: XGBoost Model Training")
    logger.info("=" * 80)
    xgboost_model = XGBoostModel(config, logger)
    validation_fraction = config['model'].get('validation_fraction', 0.1)
    X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(
        X_train_selected, y_train_final,
        test_size=validation_fraction,
        random_state=config['data']['random_state'],
        stratify=y_train_final
    )
    early_stopping_rounds = config['model'].get('early_stopping_rounds', 10)
    model = xgboost_model.train(
        X_train_fit, y_train_fit,
        params=best_params,
        feature_names=selected_features,
        eval_set=[(X_val_fit, y_val_fit)],
        early_stopping_rounds=early_stopping_rounds
    )
    xgboost_model.save_model(config['model']['save_path'])
    if preprocessor.scaler is not None:
        import joblib
        joblib.dump(preprocessor.scaler, config['model']['scaler_path'])
        logger.info(f"Scaler saved to {config['model']['scaler_path']}")
    import joblib
    joblib.dump(feature_selector, config['model']['feature_selector_path'])
    logger.info(f"Feature selector saved to {config['model']['feature_selector_path']}")
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Model Evaluation")
    logger.info("=" * 80)
    evaluator = ModelEvaluator(config, logger)
    y_pred = model.predict(X_test_selected)
    y_proba = model.predict_proba(X_test_selected)
    metrics = evaluator.evaluate_model(
        y_test, y_pred, y_proba,
        model_name="TSO-HBA Optimized XGBoost"
    )
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: SHAP Explainability")
    logger.info("=" * 80)
    shap_explainer = SHAPExplainer(config, logger)
    shap_explainer.fit(model, X_train_selected, selected_features)
    shap_explainer.generate_all_plots()
    importance_ranking = shap_explainer.get_feature_importance_ranking()
    logger.info("SHAP Feature Importance Ranking:")
    logger.info(importance_ranking.to_string())
    if run_baselines:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 9: Baseline Model Comparisons")
        logger.info("=" * 80)
        baseline_comparison = BaselineComparison(config, logger)
        baseline_results = baseline_comparison.run_comparison(X_train_selected, y_train_final, X_test_selected, y_test)
        baseline_results['tso_hba_xgboost'] = metrics
        comparison_df = evaluator.compare_models(baseline_results,save_path=str(Path(config['evaluation']['plots_path']) / "model_comparison.csv"))
        logger.info("\nModel Comparison Results:")
        logger.info(comparison_df.to_string())
        evaluator.plot_comparison(baseline_results, metric='roc_auc')
        evaluator.plot_multiple_roc_curves({
            name: (y_test, result['probabilities'])
            for name, result in baseline_comparison.results.items()})
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Final Model Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info(f"\nModel saved to: {config['model']['save_path']}")
    logger.info(f"Plots saved to: {config['evaluation']['plots_path']}")
    logger.info(f"SHAP explanations saved to: {config['explainability']['save_path']}")
    logger.info("\n" + "=" * 80)
    logger.info("To run the API: uvicorn api.main:app --reload")
    logger.info("To run the frontend: streamlit run frontend/app.py")
    logger.info("To run with Docker: docker-compose up --build")
    logger.info("=" * 80)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diabetes Prediction Framework Pipeline")
    parser.add_argument("--config",type=str,default="config.yaml",help="Path to configuration file")
    parser.add_argument("--baselines",action="store_true",help="Run baseline model comparisons")
    args = parser.parse_args()
    main(config_path=args.config, run_baselines=args.baselines)

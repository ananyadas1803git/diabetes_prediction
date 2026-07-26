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
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
sys.path.insert(0, str(Path(__file__).parent))
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.preprocessing.preprocessor import DataPreprocessor
from src.preprocessing.eda import EDAAnalyzer
from src.diffusion.tabular_diffusion import TabularDiffusion
from src.feature_selection.feature_selector import FeatureSelector
from src.optimization.tso_hba import TSOHBAOptimizer
from src.optimization.optuna_optimizer import OptunaXGBoostOptimizer
from src.models.xgboost_model import XGBoostModel
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.baseline_comparison import BaselineComparison
from src.explainability.shap_explainer import SHAPExplainer
def main(config_path: str = "config.yaml", run_baselines: bool = False, optimizer_override: str | None = None):
    config = load_config(config_path)
    if optimizer_override:
        config['optimization']['optimizer_type'] = optimizer_override
    logger = setup_logger(
        "diabetes_predictor",
        level=config['logging']['level'],
        log_file=config['logging']['log_file'],
        format_string=config['logging']['format'],
    )
    if optimizer_override:
        logger.info(f"Overriding optimizer type with CLI argument: {optimizer_override}")
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
        logger.info(f"Please place the dataset at {data_path}")
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
    if config['diffusion'].get('use_synthetic_augmentation', False):
        synthetic_train_data = X_train
        if config['diffusion'].get('generate_positive_synthetic_only', False):
            synthetic_train_data = X_train[y_train == 1]
            logger.info(
                "Training diffusion model on positive class samples only (%d rows).",
                synthetic_train_data.shape[0],
            )
        diffusion_model.train(synthetic_train_data, feature_names)
        num_samples = config['diffusion']['num_synthetic_samples']
        if config['diffusion'].get('generate_positive_synthetic_only', False):
            synthetic_df = diffusion_model.generate(num_samples, class_label=1)
        else:
            synthetic_df = diffusion_model.generate_balanced(num_samples // 2)
        diffusion_model.save_synthetic_data(synthetic_df, config['data']['synthetic_path'])
        X_train_augmented = np.vstack([X_train, synthetic_df[feature_names].values])
        synthetic_label_col = diffusion_model.target_column
        y_train_augmented = np.hstack([y_train, synthetic_df[synthetic_label_col].values])
        X_train_final, y_train_final = preprocessor.rebalance_data(X_train_augmented, y_train_augmented)
        logger.info(
            f"Using augmented training data: {X_train.shape[0]} original + {synthetic_df.shape[0]} synthetic = {X_train_final.shape[0]} total"
        )
    else:
        logger.info("Skipping synthetic data augmentation.")
        X_train_final, y_train_final = preprocessor.rebalance_data(X_train, y_train)
    imbalance_ratio = float(np.sum(y_train_final == 0)) / float(np.sum(y_train_final == 1)) if np.sum(y_train_final == 1) else 1.0
    logger.info(f"Computed scale_pos_weight={imbalance_ratio:.4f} for class imbalance handling")
    label_counts = dict(zip(*np.unique(y_train_final, return_counts=True)))
    logger.info(f"Training label counts after augmentation/rebalancing: {label_counts}")
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

    use_all_features = config['model'].get('use_all_features_for_final_model', False)
    if use_all_features:
        logger.info("Configured to train the final model on all available features.")
        selected_features = feature_names
        X_train_selected = X_train_final
        X_test_selected = X_test
    else:
        selected_indices = [feature_names.index(f) for f in selected_features]
        X_test_selected = X_test[:, selected_indices]

    if config['evaluation'].get('validate_feature_selection', False):
        from sklearn.model_selection import cross_val_score
        from xgboost import XGBClassifier

        cv_folds = config['evaluation'].get('validation_cv_folds', 5)
        logger.info("Validating feature-selection impact with CV")
        eval_model = XGBClassifier(random_state=config['data']['random_state'], use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        all_scores = cross_val_score(eval_model, X_train_final, y_train_final, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
        selected_scores = cross_val_score(eval_model, X_train_selected, y_train_final, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
        logger.info(f"All-features CV ROC-AUC: {all_scores.mean():.4f} ± {all_scores.std():.4f}")
        logger.info(f"Selected-features CV ROC-AUC: {selected_scores.mean():.4f} ± {selected_scores.std():.4f}")
        if not use_all_features and config['evaluation'].get('auto_select_feature_set', True):
            if all_scores.mean() > selected_scores.mean():
                logger.info("All-features cross-validation performance exceeds selected-features performance. Using all features for final model.")
                selected_features = feature_names
                X_train_selected = X_train_final
                X_test_selected = X_test
            else:
                logger.info("Selected feature subset performs as well or better; continuing with selected features.")

    logger.info("\n" + "=" * 80)
    optimizer_type = config['optimization'].get('optimizer_type', 'tso_hba')
    if optimizer_type == 'optuna':
        logger.info("STEP 5: Optuna Hyperparameter Optimization")
        optimizer = OptunaXGBoostOptimizer(config, logger)
    else:
        logger.info("STEP 5: TSO-HBA Hyperparameter Optimization")
        optimizer = TSOHBAOptimizer(config, logger)
    logger.info("=" * 80)
    best_params = optimizer.optimize(X_train_selected, y_train_final)
    
    # Save convergence curves for TSO-HBA optimizer
    if optimizer_type == 'tso_hba':
        optimizer.plot_tso_convergence_curve(save_path="results/tso_convergence_curve.png")
        optimizer.plot_hba_convergence_curve(save_path="results/hba_convergence_curve.png")
        optimizer.plot_combined_convergence_curves(save_path="results/combined_convergence_curve.png")
    else:
        # For Optuna, we can plot the optimization history
        try:
            optimizer.plot_optimization_history(save_path="results/hba_convergence_curve.png")
        except:
            logger.info("Could not plot Optuna optimization history")
    if "scale_pos_weight" not in best_params:
        best_params["scale_pos_weight"] = float(np.sum(y_train_final == 0)) / float(np.sum(y_train_final == 1)) if np.sum(y_train_final == 1) else 1.0
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
    sample_weight = None
    if config['model'].get('use_class_weight', False):
        class_weight = config['model'].get('class_weight', 'balanced')
        sample_weight = compute_sample_weight(class_weight, y_train_fit)
        logger.info(f"Using class-weighted sample weights with strategy={class_weight}")
    model = xgboost_model.train(
        X_train_fit, y_train_fit,
        params=best_params,
        feature_names=selected_features,
        eval_set=[(X_val_fit, y_val_fit)],
        early_stopping_rounds=early_stopping_rounds,
        sample_weight=sample_weight
    )

    if config['model'].get('calibration', {}).get('enabled', False):
        calibration_method = config['model']['calibration'].get('method', 'sigmoid')
        logger.info(f"Calibrating model probabilities using {calibration_method}")
        calibrator = CalibratedClassifierCV(estimator=model, method=calibration_method, cv='prefit')
        calibrator.fit(X_val_fit, y_val_fit)
        model = calibrator
        logger.info("Calibration completed")

    xgboost_model.model = model
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
    y_val_proba = model.predict_proba(X_val_fit)
    threshold_method = config['evaluation'].get('threshold_method', 'youden')
    threshold_beta = config['evaluation'].get('threshold_beta', 1.0)
    threshold = evaluator.find_best_threshold(
        y_val_fit,
        y_val_proba,
        method=threshold_method,
        beta=threshold_beta,
    )
    logger.info(f"Tuned threshold from validation set ({threshold_method}, beta={threshold_beta}): {threshold:.4f}")

    y_test_proba = model.predict_proba(X_test_selected)
    y_test_pred = (y_test_proba[:, 1] >= threshold).astype(int)
    metrics = evaluator.evaluate_model(
        y_test, y_test_pred, y_test_proba,
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
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--baselines", action="store_true", help="Run baseline model comparisons")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["tso_hba", "optuna"],
        help="Override the optimizer type defined in config.yaml",
    )
    args = parser.parse_args()
    main(config_path=args.config, run_baselines=args.baselines, optimizer_override=args.optimizer)

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, root_mean_squared_log_error

from src.models.regression_models import XGBoostRegressor, FFNNRegressor


def r2_log_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² in logarithmic space for log-normal distributed data.
    
    Formula: R²_log = 1 - [ Σ_i (log(y_i) - log(ŷ_i))² ] / [ Σ_i (log(y_i) - log(ȳ))² ]
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R² value in logarithmic space
    """
    # Remove zeros and negative values to avoid log issues
    mask = (y_true > 0) & (y_pred > 0)
    if np.sum(mask) < 2:
        return np.nan
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Calculate logarithms
    log_y_true = np.log(y_true_clean)
    log_y_pred = np.log(y_pred_clean)
    log_y_mean = np.log(np.mean(y_true_clean))
    
    # Calculate numerator: sum of squared residuals in log space
    numerator = np.sum((log_y_true - log_y_pred) ** 2)
    
    # Calculate denominator: total sum of squares in log space
    denominator = np.sum((log_y_true - log_y_mean) ** 2)
    
    # Avoid division by zero
    if denominator == 0:
        return np.nan
    
    # Calculate R² in log space
    r2_log = 1 - (numerator / denominator)
    
    return r2_log


def male_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Log Error (MALE).
    
    Formula: MALE = (1/n) * Σ (from i=1 to n) |log(1 + ŷ_i) - log(1 + y_i)|
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MALE value
    """
    # Remove negative values to avoid log issues
    mask = (y_true >= 0) & (y_pred >= 0)
    if np.sum(mask) < 1:
        return np.nan
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Use log1p to avoid problems with zero values
    # log1p(x) = log(1 + x) which is numerically stable
    log_diff = np.abs(np.log1p(y_pred_clean) - np.log1p(y_true_clean))
    
    # Calculate mean
    male = np.mean(log_diff)
    
    return male


def load_model_with_scalers(model_path: str) -> Optional[Dict]:
    """
    Load a trained model with its scalers.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing model, scalers, and metadata or None if error
    """
    try:
        if model_path.endswith('.pkl'):
            # Load using our custom model classes
            if 'XGB' in model_path:
                model_instance = XGBoostRegressor.load(model_path)
            elif 'FFNN' in model_path:
                model_instance = FFNNRegressor.load(model_path)
            else:
                print(f"Unknown model type in path: {model_path}")
                return None
            
            # Extract feature names from the model
            feature_names = _get_feature_names_from_model(model_instance)
            
            return {
                'model': model_instance,
                'features': feature_names,
                'model_path': model_path
            }
        else:
            print(f"Unsupported model file format: {model_path}")
            return None
    
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return None


def _get_feature_names_from_model(model_instance) -> List[str]:
    """
    Extract feature names from a trained model.
    
    Args:
        model_instance: Trained model instance
        
    Returns:
        List of feature names
    """
    try:
        # Try to get feature names from the model
        if hasattr(model_instance.model, 'feature_names_in_'):
            return list(model_instance.model.feature_names_in_)
        else:
            # Fallback: try to infer from scaler
            if hasattr(model_instance.scaler_X, 'feature_names_in_'):
                return list(model_instance.scaler_X.feature_names_in_)
            else:
                # If no feature names available, return generic names
                n_features = model_instance.scaler_X.n_features_in_
                return [f"feature_{i}" for i in range(n_features)]
    except:
        # If all else fails, return generic feature names
        return [f"feature_{i}" for i in range(100)]  # Default assumption


def predict_with_model_and_scalers(model_data: Dict, data: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Make predictions using a trained model and its scalers.
    
    Args:
        model_data: Dictionary containing model and metadata
        data: Input DataFrame
        
    Returns:
        Predictions array or None if error
    """
    try:
        model_instance = model_data['model']
        required_features = model_data['features']
        
        # Check if all required features are available
        missing_features = set(required_features) - set(data.columns)
        if missing_features:
            print(f"Missing features: {missing_features}")
            print(f"Available features: {list(data.columns)}")
            return None
        
        # Select only the required features in the correct order
        X = data[required_features]
        
        # Make predictions using the model instance
        predictions = model_instance.predict(X)
        
        return predictions
    
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return None


def load_all_models(models_dir: str) -> Tuple[Dict, List[str]]:
    """
    Load all available models from a directory.
    
    Args:
        models_dir: Directory containing trained models
        
    Returns:
        Tuple of (models_dict, model_names_list)
    """
    models_loaded = {}
    models_found = []
    
    if not os.path.exists(models_dir):
        print(f"Models directory {models_dir} does not exist")
        return models_loaded, models_found
    
    # Find all model files
    model_files = glob.glob(os.path.join(models_dir, "*.pkl"))
    
    for model_path in model_files:
        try:
            # Extract model name from filename
            filename = os.path.basename(model_path)
            model_name = filename.replace('.pkl', '')
            
            print(f"Loading model: {model_name}")
            model_data = load_model_with_scalers(model_path)
            
            if model_data:
                models_loaded[model_name] = model_data
                models_found.append(model_name)
                print(f"✓ Model {model_name} loaded successfully")
            else:
                print(f"✗ Failed to load model {model_name}")
                
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            continue
    
    return models_loaded, models_found


def evaluate_models_save_metrics(predicted_data: pd.DataFrame, 
                                models_found: List[str], 
                                output_dir: str, 
                                dataset_name: str) -> Optional[pd.DataFrame]:
    """
    Evaluate models using multiple metrics and save results.
    
    Args:
        predicted_data: DataFrame with predictions
        models_found: List of model names that were found
        output_dir: Directory to save metrics
        dataset_name: Name of the dataset being predicted
        
    Returns:
        DataFrame with metrics or None if error
    """
    print("\n=== MODEL EVALUATION ===\n")
    
    metrics_list = []
    
    for model_name in models_found:
        prediction_column = f"predicted_COT_{model_name}"
        
        if prediction_column in predicted_data.columns and 'COT' in predicted_data.columns:
            y_pred = predicted_data[prediction_column]
            y_true = predicted_data['COT']
            
            # Remove NaN values for evaluation
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if sum(mask) == 0:
                print(f"No valid data for evaluation of {model_name}")
                continue
                
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            # Calculate metrics
            r2 = r2_score(y_true_clean, y_pred_clean)
            rmse = root_mean_squared_error(y_true_clean, y_pred_clean)
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            r2_log = r2_log_score(y_true_clean, y_pred_clean)
            male = male_score(y_true_clean, y_pred_clean)
            rmsle = root_mean_squared_log_error(y_true_clean, y_pred_clean)
            
            mape_mask = np.abs(y_true_clean) > 1e-6  # Avoid division by very small values
            if np.sum(mape_mask) > 0:
                mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])) * 100
            else:
                mape = np.nan
            
            metrics_list.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'R²': round(r2, 4),
                'R²_log': round(r2_log, 4) if not np.isnan(r2_log) else np.nan,
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'MALE': round(male, 4) if not np.isnan(male) else np.nan,
                'RMSLE': round(rmsle, 4),
                'MAPE': round(mape, 4) if not np.isnan(mape) else np.nan,
                'N_points': len(y_true_clean)
            })
            
            print(f"\nMetrics for {model_name}:")
            print(f"R²: {r2:.4f}")
            print(f"R²_log: {r2_log:.4f}" if not np.isnan(r2_log) else "R²_log: NaN")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"MALE: {male:.4f}" if not np.isnan(male) else "MALE: NaN")
            print(f"RMSLE: {rmsle:.4f}")
            print(f"MAPE: {mape:.4f}%" if not np.isnan(mape) else "MAPE: NaN")
            print(f"N points: {len(y_true_clean)}")
    
    if metrics_list:
        df_metrics = pd.DataFrame(metrics_list)
        metrics_path = os.path.join(output_dir, f'metrics_prediction_{dataset_name}.csv')
        df_metrics.to_csv(metrics_path, index=False)
        print(f"\nMetrics saved to: {metrics_path}")
        return df_metrics
    else:
        print("No metrics calculated")
        return None


def plot_pred_vs_measured(predicted_data: pd.DataFrame, 
                            output_dir: str, 
                            well_name: str,
                            models_found: List[str]) -> None:
    """
    Create and save predicted vs measured scatter plots.
    
    Args:
        predicted_data: DataFrame with predictions
        output_dir: Directory to save plots
        well_name: Name of the well
        models_found: List of model names
    """
    if 'COT' not in predicted_data.columns:
        print("No COT column found for plotting")
        return
    
    for model_name in models_found:
        prediction_column = f"predicted_COT_{model_name}"
        
        if prediction_column in predicted_data.columns:
            y_true = predicted_data['COT']
            y_pred = predicted_data[prediction_column]
            
            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if sum(mask) == 0:
                print(f"No valid data for plotting {model_name}")
                continue
                
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            # Calculate metrics for plot
            r2 = r2_score(y_true_clean, y_pred_clean)
            rmse = root_mean_squared_error(y_true_clean, y_pred_clean)
            rmsle = root_mean_squared_log_error(y_true_clean, y_pred_clean)
            
            mape_mask = np.abs(y_true_clean) > 1e-6  # Avoid division by very small values
            if np.sum(mape_mask) > 0:
                mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])) * 100
            else:
                mape = np.nan
            
            plt.figure(figsize=(10, 8))
            plt.scatter(y_true_clean, y_pred_clean, alpha=0.6, color='blue', s=50)
            
            # Add 1:1 line
            min_val = min(min(y_true_clean), min(y_pred_clean))
            max_val = max(max(y_true_clean), max(y_pred_clean))
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, alpha=0.8)
            
            plt.xlabel("Measured COT (%)", fontsize=12)
            plt.ylabel("Predicted COT (%)", fontsize=12)
            plt.title(f"Predicted vs Measured - {model_name}", fontsize=14)
            plt.grid(True, alpha=0.3)
            
            plt.xlim(0, 15)
            plt.ylim(0, 15)
            
            # Add metrics text
            plt.text(0.05, 0.95, f'RMSE = {rmse:.4f}\nRMSLE = {rmsle:.4f}\nMAPE = {mape:.4f}%', 
                    transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plot_dir = os.path.join(output_dir, "pred_vs_measured")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"pred_vs_measured_{model_name}_{well_name}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Predicted vs measured plot saved to: {plot_path}")
            plt.show()


def plot_predictions(predicted_data: pd.DataFrame, 
                    output_dir: str, 
                    well_name: str,
                    models_found: List[str]) -> None:
    """
    Create and save prediction comparison plots.
    
    Args:
        predicted_data: DataFrame with predictions
        output_dir: Directory to save plots
        well_name: Name of the well
        models_found: List of model names
    """
    if 'DEPTH' not in predicted_data.columns:
        print("No DEPTH column found for plotting")
        return
    
    # Create subplots
    n_models = len(models_found)
    if n_models == 0:
        print("No models to plot")
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Prediction Comparison', fontsize=16)
    
    for i, model_name in enumerate(models_found):
        prediction_column = f"predicted_COT_{model_name}"
        
        if prediction_column in predicted_data.columns:
            ax = axes[i]
            
            # Plotar COT (terceiros)
            if 'COT' in predicted_data.columns:
                mask_cot = ~predicted_data['COT'].isna()
                if mask_cot.any():
                    ax.scatter(predicted_data.loc[mask_cot, 'COT'], predicted_data.loc[mask_cot, 'DEPTH'],
                               label="Measured COT", color='blue', marker='o', s=20, alpha=0.6)
            # Plotar COT_anl (análise própria)
            if 'COT_anl' in predicted_data.columns:
                mask_cot_anl = ~predicted_data['COT_anl'].isna()
                if mask_cot_anl.any():
                    ax.scatter(predicted_data.loc[mask_cot_anl, 'COT_anl'], predicted_data.loc[mask_cot_anl, 'DEPTH'],
                               label="Analytical COT", color='black', marker='*', s=80, alpha=0.9)
            
            ax.plot(predicted_data[prediction_column], predicted_data['DEPTH'], 
                   color="darkred", label=f"Predicted COT", alpha=0.8, linewidth=2)
            
            ax.set_xlim(0, 15)
            ax.set_xlabel("COT (%)")
            ax.set_ylabel("Depth (m)" if i == 0 else "")
            ax.set_title(f"{model_name}")
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"predictions_comparison_{well_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Predictions comparison plot saved to: {plot_path}")
    plt.show()


def plot_predictions_by_combination(predicted_data: pd.DataFrame, 
                                    output_dir: str, 
                                    well_name: str,
                                    combinations: dict) -> None:
    """
    Create and save prediction comparison plots grouped by feature combinations.
    
    Args:
        predicted_data: DataFrame with predictions
        output_dir: Directory to save plots
        well_name: Name of the well
        combinations: Dictionary mapping combination IDs to model names
    """
    if 'DEPTH' not in predicted_data.columns:
        print("No DEPTH column found for plotting")
        return
    
    for combo_id, model_names in combinations.items():
        # Create subplot for this combination (2 models: XGB and FFNN)
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f'Prediction Comparison - Combination {combo_id}', fontsize=16)
        
        for i, model_name in enumerate(model_names):
            prediction_column = f"predicted_COT_{model_name}"
            
            if prediction_column in predicted_data.columns:
                ax = axes[i]
                
                # Plotar COT (terceiros)
                if 'COT' in predicted_data.columns:
                    mask_cot = ~predicted_data['COT'].isna()
                    if mask_cot.any():
                        ax.scatter(predicted_data.loc[mask_cot, 'COT'], predicted_data.loc[mask_cot, 'DEPTH'],
                                   label="Measured COT", color='blue', marker='o', s=20, alpha=0.6)
                # Plotar COT_anl (análise própria)
                if 'COT_anl' in predicted_data.columns:
                    mask_cot_anl = ~predicted_data['COT_anl'].isna()
                    if mask_cot_anl.any():
                        ax.scatter(predicted_data.loc[mask_cot_anl, 'COT_anl'], predicted_data.loc[mask_cot_anl, 'DEPTH'],
                                   label="Analytical COT", color='black', marker='*', s=80, alpha=0.9)
                
                ax.plot(predicted_data[prediction_column], predicted_data['DEPTH'], 
                       color="darkred", label=f"Predicted COT", alpha=0.8, linewidth=2)
                
                ax.set_xlim(0, 15)
                ax.set_xlabel("COT (%)")
                ax.set_ylabel("Depth (m)" if i == 0 else "")
                ax.set_title(f"{model_name}")
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f"predictions_comparison_{combo_id}_{well_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Predictions comparison plot for combination {combo_id} saved to: {plot_path}")
        plt.show()


def plot_residuals(predicted_data: pd.DataFrame, 
                    output_dir: str, 
                    well_name: str,
                    models_found: List[str]) -> None:
    """
    Create and save residual plots for each model.
    
    Args:
        predicted_data: DataFrame with predictions
        output_dir: Directory to save plots
        well_name: Name of the well
        models_found: List of model names
    """
    if 'COT' not in predicted_data.columns:
        print("No COT column found for residual plotting")
        return
    
    for model_name in models_found:
        prediction_column = f"predicted_COT_{model_name}"
        
        if prediction_column in predicted_data.columns:
            y_true = predicted_data['COT']
            y_pred = predicted_data[prediction_column]
            
            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if sum(mask) == 0:
                print(f"No valid data for residual plotting {model_name}")
                continue
                
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            # Calculate residuals
            residuals = y_true_clean - y_pred_clean
            
            # Calculate metrics
            rmse = root_mean_squared_error(y_true_clean, y_pred_clean)
            rmsle = root_mean_squared_log_error(y_true_clean, y_pred_clean)
            
            # Calculate MAPE with protection against division by zero and very small values
            mape_mask = np.abs(y_true_clean) > 1e-6  # Avoid division by very small values
            if np.sum(mape_mask) > 0:
                mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])) * 100
            else:
                mape = np.nan
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16, fontweight='bold')
            
            # 1. Residuals vs Predicted Values
            ax1.scatter(y_pred_clean, residuals, alpha=0.6, color='blue', s=30)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax1.set_xlabel('Predicted COT (%)', fontsize=12)
            ax1.set_ylabel('Residuals (Measured - Predicted)', fontsize=12)
            ax1.set_title('Residuals vs Predicted Values', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 20)
            ax1.set_ylim(-5, 20)
            
            # Add trend line
            z = np.polyfit(y_pred_clean, residuals, 1)
            p = np.poly1d(z)
            ax1.plot(y_pred_clean, p(y_pred_clean), "r--", alpha=0.8, linewidth=2)
            
            # 2. Residuals vs Measured Values
            ax2.scatter(y_true_clean, residuals, alpha=0.6, color='green', s=30)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax2.set_xlabel('Measured COT (%)', fontsize=12)
            ax2.set_ylabel('Residuals (Measured - Predicted)', fontsize=12)
            ax2.set_title('Residuals vs Measured Values', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 20)
            ax2.set_ylim(-5, 20)
            
            # Add trend line
            z = np.polyfit(y_true_clean, residuals, 1)
            p = np.poly1d(z)
            ax2.plot(y_true_clean, p(y_true_clean), "r--", alpha=0.8, linewidth=2)
            
            # 3. Histogram of Residuals
            ax3.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8)
            ax3.set_xlabel('Residuals (Measured - Predicted)', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title('Distribution of Residuals', fontsize=14)
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(-5, 20)
            
            # Add normal distribution curve
            from scipy.stats import norm
            mu, std = norm.fit(residuals)
            xmin, xmax = ax3.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax3.plot(x, p * len(residuals) * (xmax - xmin) / 30, 'r-', linewidth=2, alpha=0.8)
            
            # Add metrics text box INSIDE the histogram panel (top left)
            metrics_text = f'RMSE = {rmse:.4f}\nRMSLE = {rmsle:.4f}\nMAPE = {mape:.4f}%\nMean Residual = {residuals.mean():.4f}\nStd Residual = {residuals.std():.4f}'
            ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes, fontsize=11,
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 4. Q-Q Plot for normality check
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.set_title('Normality Check', fontsize=14)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_dir = os.path.join(output_dir, "residuals")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"residuals_{model_name}_{well_name}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Residual plot for {model_name} saved to: {plot_path}")
            plt.show()


def plot_residuals_by_combination(predicted_data: pd.DataFrame, 
                                output_dir: str, 
                                well_name: str,
                                combinations: dict) -> None:
    """
    Create and save residual plots grouped by feature combinations.
    
    Args:
        predicted_data: DataFrame with predictions
        output_dir: Directory to save plots
        well_name: Name of the well
        combinations: Dictionary mapping combination IDs to model names
    """
    if 'COT' not in predicted_data.columns:
        print("No COT column found for residual plotting")
        return
    
    for combo_id, model_names in combinations.items():
        # Get features from first model name
        if model_names:
            first_model = model_names[0]
            parts = first_model.split('_')
            if len(parts) >= 4:
                features = parts[2:]
                features_str = '_'.join(features)
            else:
                features_str = "unknown_features"
        else:
            features_str = "unknown_features"
        
        # Create subplot for this combination (2 models: XGB and FFNN)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Residual Analysis - Combination {combo_id}\nFeatures: {features_str}', 
                    fontsize=16, fontweight='bold')
        
        for i, model_name in enumerate(model_names):
            prediction_column = f"predicted_COT_{model_name}"
            
            if prediction_column in predicted_data.columns:
                y_true = predicted_data['COT']
                y_pred = predicted_data[prediction_column]
                
                # Remove NaN values
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                if sum(mask) == 0:
                    print(f"No valid data for residual plotting {model_name}")
                    continue
                    
                y_true_clean = y_true[mask]
                y_pred_clean = y_pred[mask]
                
                # Calculate residuals
                residuals = y_true_clean - y_pred_clean
                
                # Calculate metrics
                rmse = root_mean_squared_error(y_true_clean, y_pred_clean)
                rmsle = root_mean_squared_log_error(y_true_clean, y_pred_clean)
                
                # Calculate MAPE with protection against division by zero and very small values
                mape_mask = np.abs(y_true_clean) > 1e-6  # Avoid division by very small values
                if np.sum(mape_mask) > 0:
                    mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])) * 100
                else:
                    mape = np.nan
                
                # Plot residuals vs predicted values
                ax = axes[i, 0]
                ax.scatter(y_pred_clean, residuals, alpha=0.6, color='blue', s=30)
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
                ax.set_xlabel('Predicted COT (%)', fontsize=10)
                ax.set_ylabel('Residuals', fontsize=10)
                ax.set_title(f'{model_name} - Residuals vs Predicted', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 20)
                
                # Add trend line
                z = np.polyfit(y_pred_clean, residuals, 1)
                p = np.poly1d(z)
                ax.plot(y_pred_clean, p(y_pred_clean), "r--", alpha=0.8, linewidth=2)
                
                # Plot histogram of residuals
                ax = axes[i, 1]
                ax.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.8)
                ax.set_xlabel('Residuals', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_title(f'{model_name} - Residual Distribution', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-5, 20)
                
                # Add metrics text INSIDE the histogram panel (top left)
                metrics_text = f'RMSE: {rmse:.4f}\nRMSLE: {rmsle:.4f}\nMAPE: {mape:.4f}%'
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = os.path.join(output_dir, "residuals")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"residuals_by_combination_{combo_id}_{well_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Residuals by combination plot for combination {combo_id} saved to: {plot_path}")
        plt.show()


def plot_predictions_by_combination_window(predicted_data: pd.DataFrame, 
                                  output_dir: str, 
                                  well_name: str,
                                  combinations: dict,
                                  window: int = 5) -> None:
    """
    Cria e salva gráficos de comparação de predição por combinação de features,
    adicionando envelope de desvio padrão do erro em uma janela móvel.
    
    Args:
        predicted_data: DataFrame com predições
        output_dir: Diretório para salvar os plots
        well_name: Nome do poço
        combinations: Dicionário de combinações para modelos
        window: Tamanho da janela móvel (default=5)
    """
    if 'DEPTH' not in predicted_data.columns:
        print("No DEPTH column found for plotting")
        return
    
    for combo_id, model_names in combinations.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f'Prediction Comparison - Combination {combo_id}', fontsize=16)
        
        for i, model_name in enumerate(model_names):
            prediction_column = f"predicted_COT_{model_name}"
            
            if prediction_column in predicted_data.columns:
                ax = axes[i]
                
                # Plotar COT (terceiros)
                if 'COT' in predicted_data.columns:
                    mask_cot = ~predicted_data['COT'].isna()
                    if mask_cot.any():
                        ax.scatter(predicted_data.loc[mask_cot, 'COT'], predicted_data.loc[mask_cot, 'DEPTH'],
                                   label="Measured COT", color='blue', marker='o', s=20, alpha=0.6)
                # Plotar COT_anl (análise própria)
                if 'COT_anl' in predicted_data.columns:
                    mask_cot_anl = ~predicted_data['COT_anl'].isna()
                    if mask_cot_anl.any():
                        ax.scatter(predicted_data.loc[mask_cot_anl, 'COT_anl'], predicted_data.loc[mask_cot_anl, 'DEPTH'],
                                   label="Analytical COT", color='black', marker='*', s=80, alpha=0.9)
                
                ax.plot(predicted_data[prediction_column], predicted_data['DEPTH'], 
                       color="darkred", label=f"Predicted COT", alpha=0.8, linewidth=2)
                
                # Envelope de desvio padrão do erro em janela móvel
                if 'COT' in predicted_data.columns:
                    y_true = predicted_data['COT']
                    y_pred = predicted_data[prediction_column]
                    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                    depth = predicted_data['DEPTH'][mask].values
                    y_pred_valid = y_pred[mask].values
                    y_true_valid = y_true[mask].values
                    error = y_true_valid - y_pred_valid
                    # Calcular std do erro em janela móvel
                    std_error = pd.Series(error).rolling(window, center=True, min_periods=1).std().values
                    lower = y_pred_valid - std_error
                    upper = y_pred_valid + std_error
                    ax.fill_betweenx(depth, lower, upper, color='orange', alpha=0.3, label=f'STD(residuals) window={window}')
                
                ax.set_xlim(0, 20)
                ax.set_xlabel("COT (%)")
                ax.set_ylabel("Depth (m)" if i == 0 else "")
                ax.set_title(f"{model_name}")
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.invert_yaxis()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"predictions_comparison_window_{combo_id}_{well_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Predictions comparison (window STD) plot for combination {combo_id} saved to: {plot_path}")
        plt.show()

# Baseline check sem janela de profundidade
def check_baseline_predictions(df: pd.DataFrame,
                                model_col: str = None,
                                output_dir: str = None,
                                baseline_range: tuple = (0, 1),
                                plot: bool = True) -> None:
    """
    Check and plot the baseline of the measured and predicted TOC by the model.
    Args:
        df: DataFrame with columns 'COT' and the prediction column of the model
        model_col: name of the column of the model to be evaluated
        output_dir: directory to save the plot
        baseline_range: tuple (min, max) to filter baseline
        plot: se True, gera e salva o gráfico
    """
    if model_col is None:
        # Tenta detectar automaticamente a coluna de predição
        pred_cols = [c for c in df.columns if c.startswith('predicted_COT_')]
        if not pred_cols:
            print('No prediction column found!')
            return
        model_col = pred_cols[0]
        print(f'Using prediction column: {model_col}')

    # Filtrar baseline
    min_val, max_val = baseline_range
    baseline = df[(df["COT"] >= min_val) & (df["COT"] <= max_val)].copy()
    if baseline.empty:
        print(f"No samples found in baseline ({min_val} <= COT <= {max_val})!")
        return

    # Estatísticas
    print(f"=== BASELINE (TOC between {min_val} and {max_val}) ===")
    print(f"Total of samples: {len(baseline)}")
    print(f"Measured TOC: mean={baseline['COT'].mean():.3f}, median={baseline['COT'].median():.3f}, std={baseline['COT'].std():.3f}")
    print(f"Predicted TOC: mean={baseline[model_col].mean():.3f}, median={baseline[model_col].median():.3f}, std={baseline[model_col].std():.3f}")
    baseline["erro"] = baseline[model_col] - baseline["COT"]
    print(f"Mean error: {baseline['erro'].mean():.3f}")
    print(f"RMSE: {((baseline['erro']**2).mean())**0.5:.3f}")

    # Quantos valores preditos ficaram acima de 1
    n_pred_acima_1 = (baseline[model_col] > 1).sum()
    perc_pred_acima_1 = 100 * n_pred_acima_1 / len(baseline) if len(baseline) > 0 else 0
    print(f"Predicted TOC > 1: {n_pred_acima_1} samples ({perc_pred_acima_1:.1f}%)")

    if plot:
        plt.figure(figsize=(12, 5))
        algorithm = model_col.split('_')[2] if len(model_col.split('_')) > 2 else model_col
        plt.suptitle(f"Baseline Check {algorithm}", fontsize=16)
        plt.subplot(1, 2, 1)
        plt.hist(baseline["COT"], bins=20, alpha=0.7, label="Measured TOC")
        plt.hist(baseline[model_col], bins=20, alpha=0.7, label="Predicted TOC")
        plt.title("Distribution TOC (Baseline)")
        plt.xlabel("TOC (%)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.scatter(baseline["COT"], baseline[model_col], alpha=0.7)
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
        plt.title("TOC Measured vs Predicted (Baseline)")
        plt.xlabel("TOC Measured")
        plt.ylabel("TOC Predicted")
        plt.legend()
        plt.tight_layout()
        if output_dir is not None:
            plot_dir = os.path.join(output_dir, "baseline")
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"baseline_check_{model_col}.png"), dpi=150)
        plt.show()
    result = {
        "model": model_col,
        "total_samples": int(len(baseline)),
        "measured_toc_mean": baseline['COT'].mean(),
        "measured_toc_median": baseline['COT'].median(),
        "measured_toc_std": baseline['COT'].std(),
        "predicted_toc_mean": baseline[model_col].mean(),
        "predicted_toc_median": baseline[model_col].median(),
        "predicted_toc_std": baseline[model_col].std(),
        "mean_error": baseline['erro'].mean(),
        "rmse": ((baseline['erro']**2).mean())**0.5,
        "n_predicted_toc_above_1": int(n_pred_acima_1),
        "perc_predicted_toc_above_1": perc_pred_acima_1
    }
    if output_dir is not None:
        plot_dir = os.path.join(output_dir, "baseline")
        os.makedirs(plot_dir, exist_ok=True)
        csv_path = os.path.join(plot_dir, "baseline_check_results.csv")
        if os.path.exists(csv_path):
            df_baseline = pd.read_csv(csv_path)
            df_baseline = pd.concat([df_baseline, pd.DataFrame([result])], ignore_index=True)
        else:
            df_baseline = pd.DataFrame([result])
        float_cols = [col for col in df_baseline.columns if df_baseline[col].dtype in ['float32', 'float64', 'float']]
        df_baseline[float_cols] = df_baseline[float_cols].round(3)
        df_baseline.to_csv(csv_path, index=False)
    return None

# Baseline check com janela de profundidade
def check_baseline_predictions_depth(df: pd.DataFrame,
                                        model_col: str = None,
                                        output_dir: str = None,
                                        baseline_range: tuple = (0, 1),
                                        depth_min: float = None,
                                        depth_max: float = None,
                                        plot: bool = True) -> None:
    """
    Check and plot the baseline of measured and predicted TOC by the model, restricted to a depth interval.
    Args:
        df: DataFrame with columns 'COT', 'DEPTH' and the prediction column of the model
        model_col: name of the column of the model to be evaluated
        output_dir: directory to save the plot
        baseline_range: tuple (min, max) to filter baseline
        depth_min: minimum depth (inclusive)
        depth_max: maximum depth (inclusive)
        plot: if True, generates and saves the plot
    """
    if model_col is None:
        pred_cols = [c for c in df.columns if c.startswith('predicted_COT_')]
        if not pred_cols:
            print('No prediction column found!')
            return
        model_col = pred_cols[0]
        print(f'Using prediction column: {model_col}')

    # Filtrar baseline
    min_val, max_val = baseline_range
    baseline = df[(df["COT"] >= min_val) & (df["COT"] <= max_val)].copy()
    if depth_min is not None and depth_max is not None:
        baseline = baseline[(baseline["DEPTH"] >= depth_min) & (baseline["DEPTH"] <= depth_max)]
    if baseline.empty:
        print(f"No samples found in baseline ({min_val} <= COT <= {max_val}) and depth ({depth_min} <= DEPTH <= {depth_max})!")
        return

    print(f"=== BASELINE (TOC between {min_val} and {max_val}), DEPTH {depth_min} to {depth_max} ===")
    print(f"Total of samples: {len(baseline)}")
    print(f"Measured TOC: mean={baseline['COT'].mean():.3f}, median={baseline['COT'].median():.3f}, std={baseline['COT'].std():.3f}")
    print(f"Predicted TOC: mean={baseline[model_col].mean():.3f}, median={baseline[model_col].median():.3f}, std={baseline[model_col].std():.3f}")
    baseline["erro"] = baseline[model_col] - baseline["COT"]
    print(f"Mean error: {baseline['erro'].mean():.3f}")
    print(f"RMSE: {((baseline['erro']**2).mean())**0.5:.3f}")

    # Quantos valores preditos ficaram acima de 1
    n_pred_acima_1 = (baseline[model_col] > 1).sum()
    perc_pred_acima_1 = 100 * n_pred_acima_1 / len(baseline) if len(baseline) > 0 else 0
    print(f"Predicted TOC > 1: {n_pred_acima_1} samples ({perc_pred_acima_1:.1f}%)")

    if plot:
        plt.figure(figsize=(12, 5))
        algorithm = model_col.split('_')[2] if len(model_col.split('_')) > 2 else model_col
        plt.suptitle(f"Baseline Check {algorithm} - Depth {depth_min}-{depth_max}", fontsize=16)
        plt.subplot(1, 2, 1)
        plt.hist(baseline["COT"], bins=20, alpha=0.7, label="Measured TOC")
        plt.hist(baseline[model_col], bins=20, alpha=0.7, label="Predicted TOC")
        plt.title(f"Distribution TOC (Baseline)\nDepth {depth_min}-{depth_max}")
        plt.xlabel("TOC (%)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.scatter(baseline["COT"], baseline[model_col], alpha=0.7)
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
        plt.title(f"TOC Measured vs Predicted (Baseline)\nDepth {depth_min}-{depth_max}")
        plt.xlabel("TOC Measured")
        plt.ylabel("TOC Predicted")
        plt.legend()
        plt.tight_layout()
        if output_dir is not None:
            plot_dir = os.path.join(output_dir, "baseline")
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"baseline_check_{model_col}_depth_{depth_min}_{depth_max}.png"), dpi=150)
        plt.show()
     
    result = {
        "model": model_col,
        "total_samples": int(len(baseline)),
        "measured_toc_mean": baseline['COT'].mean(),
        "measured_toc_median": baseline['COT'].median(),
        "measured_toc_std": baseline['COT'].std(),
        "predicted_toc_mean": baseline[model_col].mean(),
        "predicted_toc_median": baseline[model_col].median(),
        "predicted_toc_std": baseline[model_col].std(),
        "mean_error": baseline['erro'].mean(),
        "rmse": ((baseline['erro']**2).mean())**0.5,
        "n_predicted_toc_above_1": int(n_pred_acima_1),
        "perc_predicted_toc_above_1": perc_pred_acima_1
    }
    if output_dir is not None:
        plot_dir = os.path.join(output_dir, "baseline")
        os.makedirs(plot_dir, exist_ok=True)
        csv_path = os.path.join(plot_dir, "baseline_check_depth_5200_5800_results.csv")
        if os.path.exists(csv_path):
            df_depth = pd.read_csv(csv_path)
            df_depth = pd.concat([df_depth, pd.DataFrame([result])], ignore_index=True)
        else:
            df_depth = pd.DataFrame([result])
        float_cols = [col for col in df_depth.columns if df_depth[col].dtype in ['float32', 'float64', 'float']]
        df_depth[float_cols] = df_depth[float_cols].round(3)
        df_depth.to_csv(csv_path, index=False)
    return None 


def plot_predictions_by_combination_bars_window(predicted_data: pd.DataFrame, 
                                  output_dir: str, 
                                  well_name: str,
                                  combinations: dict) -> None:
    """
    Gráfico simples: para cada profundidade, uma barra horizontal no valor predito.
    Eixo X: 0 a 20. Opcional: pontos pretos para COT medido.
    """
    if 'DEPTH' not in predicted_data.columns:
        print("No DEPTH column found for plotting")
        return
    if 'COT' not in predicted_data.columns:
        print("No COT column found for plotting")
        return
    for combo_id, model_names in combinations.items():
        fig, axes = plt.subplots(1, len(model_names), figsize=(8*len(model_names), 10))
        if len(model_names) == 1:
            axes = [axes]
        for i, model_name in enumerate(model_names):
            prediction_column = f"predicted_COT_{model_name}"
            if prediction_column in predicted_data.columns:
                ax = axes[i]
                mask = ~(predicted_data["COT"].isna() | predicted_data[prediction_column].isna())
                depth = predicted_data.loc[mask, 'DEPTH'].values
                y_pred = predicted_data.loc[mask, prediction_column].values
                y_true = predicted_data.loc[mask, 'COT'].values
                # Barra horizontal para cada valor predito
                ax.barh(depth, y_pred, height=3, color='royalblue', alpha=0.7, label='Predicted COT')
                # Pontos pretos para COT medido
                ax.scatter(y_true, depth, color='black', marker='o', s=20, alpha=0.7, label='Measured COT')
                ax.set_xlabel("COT (%)")
                ax.set_ylabel("Depth (m)")
                ax.set_title(f"{model_name}")
                ax.set_xlim(0, 20)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plot_path = os.path.join(output_dir, f"predictions_bars_pointwise_{combo_id}_{well_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Prediction bars (pointwise) plot for combination {combo_id} saved to: {plot_path}")
        plt.show() 


def plot_predictions_by_combination_bars_window_2(predicted_data: pd.DataFrame, 
                                  output_dir: str, 
                                  well_name: str,
                                  combinations: dict,
                                  window: int = 5) -> None:
    """
    Gráfico simples: para cada profundidade, uma barra horizontal no valor predito.
    Eixo X: 0 a 20. Opcional: pontos pretos para COT medido.
    Adiciona envelope de desvio padrão do erro em janela móvel (window).
    """
    if 'DEPTH' not in predicted_data.columns:
        print("No DEPTH column found for plotting")
        return
    if 'COT' not in predicted_data.columns:
        print("No COT column found for plotting")
        return
    for combo_id, model_names in combinations.items():
        fig, axes = plt.subplots(1, len(model_names), figsize=(8*len(model_names), 10))
        if len(model_names) == 1:
            axes = [axes]
        for i, model_name in enumerate(model_names):
            prediction_column = f"predicted_COT_{model_name}"
            if prediction_column in predicted_data.columns:
                ax = axes[i]
                mask = ~(predicted_data["COT"].isna() | predicted_data[prediction_column].isna())
                depth = predicted_data.loc[mask, 'DEPTH'].values
                y_pred = predicted_data.loc[mask, prediction_column].values
                y_true = predicted_data.loc[mask, 'COT'].values
                # Barra horizontal para cada valor predito
                ax.barh(depth, y_pred, height=3, color='royalblue', alpha=0.7, label='Predicted COT')
                # Pontos pretos para COT medido
                ax.scatter(y_true, depth, color='black', marker='o', s=20, alpha=0.7, label='Measured COT')
                # Envelope de desvio padrão do erro em janela móvel
                erro = y_true - y_pred
                std_erro = pd.Series(erro).rolling(window, center=True, min_periods=1).std().values
                lower = y_pred - std_erro
                upper = y_pred + std_erro
                ax.fill_betweenx(depth, lower, upper, color='orange', alpha=0.3, label=f'STD(residuals) window={window}')
                ax.set_xlabel("COT (%)")
                ax.set_ylabel("Depth (m)")
                ax.set_title(f"{model_name}")
                ax.set_xlim(0, 20)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plot_path = os.path.join(output_dir, f"predictions_bars_pointwise_2_{combo_id}_{well_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Prediction bars (pointwise) plot for combination {combo_id} saved to: {plot_path}")
        plt.show() 
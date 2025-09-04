"""
Regression models with Bayesian optimization for TOC prediction.
"""

import numpy as np
import pandas as pd
import pickle
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, root_mean_squared_log_error
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement
import xgboost as xgb
from sklearn.neural_network import MLPRegressor


class BaseRegressor(ABC):
    """
    Abstract base class for regression models.
    """
    
    def __init__(self, random_state: int = 42, scaler: str = 'robust'):
        """
        Initialize the base regressor.
        
        Args:
            random_state: Random seed for reproducibility
            scaler: Type of scaler to use ('robust' or 'standard')
        """
        self.random_state = random_state
        self.scaler_type = scaler.lower()
        
        # Initialize scalers based on choice
        if self.scaler_type == 'robust':
            self.scaler_X = RobustScaler()
            self.scaler_y = RobustScaler()
        elif self.scaler_type == 'standard':
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
        else:
            raise ValueError(f"Scaler '{scaler}' not supported. Use 'robust' or 'standard'")
        
        self.model = None
        self.best_hyperparameters = None
        self.hyperparameter_history = []
        
    @abstractmethod
    def _get_model(self, **params) -> Any:
        pass
    
    @abstractmethod
    def _get_parameter_bounds(self) -> Dict[str, Tuple]:
        pass
    
    @abstractmethod
    def _evaluate_model(self, **params) -> float:
        """Evaluate model with given parameters using cross-validation."""
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series, optimize: bool = True, 
            n_iter: int = 50, init_points: int = 10) -> Dict[str, Any]:
        """
        Fit the model with optional Bayesian optimization.
        
        Args:
            X: Feature matrix
            y: Target variable
            optimize: Whether to use Bayesian optimization
            n_iter: Number of optimization iterations
            init_points: Number of initial random points
            
        Returns:
            Dictionary with model, scalers, and metadata
        """
        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        if optimize:
            # Bayesian optimization
            acq = ExpectedImprovement(xi=0.01, exploration_decay=0.95, exploration_decay_delay=5)
            optimizer = BayesianOptimization(
                f=self._evaluate_model,
                pbounds=self._get_parameter_bounds(),
                random_state=self.random_state,
                acquisition_function=acq,
                verbose=1
            )
            
            optimizer.maximize(
                init_points=init_points,
                n_iter=n_iter,
            )
            
            # Get best parameters
            best_params = optimizer.max['params']
            self.best_hyperparameters = self._process_best_params(best_params)
            self.hyperparameter_history = optimizer.res
            
        else:
            # Use default parameters
            self.best_hyperparameters = self._get_default_params()
        
        # Train final model with best parameters
        self.model = self._get_model(**self.best_hyperparameters)
        self.model.fit(X_scaled, y_scaled)
        
        return {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'scaler_type': self.scaler_type,
            'best_hyperparameters': self.best_hyperparameters,
            'hyperparameter_history': self.hyperparameter_history
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions in original scale (clipped to >= 0)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler_X.transform(X)
        predictions_scaled = self.model.predict(X_scaled)
        
        # Inverse transform to original scale
        predictions = self.scaler_y.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).ravel()
        
        # Clip predictions to ensure they are >= 0 (TOC cannot be negative)
        predictions = np.clip(predictions, 0, np.inf)
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with R² and RMSE scores
        """
        predictions = self.predict(X)
        
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = root_mean_squared_error(y, predictions)
        
        mape_mask = np.abs(y) > 1e-6  # Avoid division by very small values
        if np.sum(mape_mask) > 0:
            mape = np.mean(np.abs((y[mape_mask] - predictions[mape_mask]) / y[mape_mask])) * 100
        else:
            mape = np.nan
            
        rmsle = root_mean_squared_log_error(y, predictions)

        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'rmsle': rmsle
        }
    
    def save(self, filepath: str) -> None:
        """
        Save model and scalers to file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'scaler_type': self.scaler_type,
            'best_hyperparameters': self.best_hyperparameters,
            'hyperparameter_history': self.hyperparameter_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseRegressor':
        """
        Load model from file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(scaler=model_data.get('scaler_type', 'robust'))
        instance.model = model_data['model']
        instance.scaler_X = model_data['scaler_X']
        instance.scaler_y = model_data['scaler_y']
        instance.best_hyperparameters = model_data['best_hyperparameters']
        instance.hyperparameter_history = model_data['hyperparameter_history']
        
        return instance
    
    @abstractmethod
    def _process_best_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def _get_default_params(self) -> Dict[str, Any]:
        pass


class XGBoostRegressor(BaseRegressor):
    
    def _get_model(self, **params) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_child_weight=int(params['min_child_weight']),
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def _get_parameter_bounds(self) -> Dict[str, Tuple]:
        return {
            'n_estimators': (100, 1000, int),  # Aumentei para capturar mais complexidade
            'max_depth': (3, 12, int),  # Aumentei para dados geológicos complexos
            'min_child_weight': (1, 10, int),  # Reduzi range - valores altos podem underfit
            'learning_rate': (0.005, 0.3),  # Reduzi mínimo para melhor convergência
            'subsample': (0.7, 1.0),  # Aumentei mínimo - dados geológicos são ruidosos
            'colsample_bytree': (0.7, 1.0),  # Aumentei mínimo
            'gamma': (0, 5),  # Aumentei máximo para mais regularização
            'reg_alpha': (0, 20),  # Aumentei para L1 regularization
            'reg_lambda': (0, 20),  # Aumentei para L2 regularization
        }
    
    def _evaluate_model(self, **params) -> float:
        model = self._get_model(**params)
        
        # Use negative RMSE for maximization
        scores = cross_val_score(
            model, 
            self.scaler_X.transform(self.X_train), 
            self.scaler_y.transform(self.y_train.values.reshape(-1, 1)).ravel(),
            cv=5, 
            scoring='neg_root_mean_squared_error'
        )
        
        return scores.mean()
    
    def _process_best_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'min_child_weight': int(params['min_child_weight']),
            'learning_rate': params['learning_rate'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'gamma': params['gamma'],
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
        }
    
    def _get_default_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'min_child_weight': 1,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, optimize: bool = True, 
            n_iter: int = 50, init_points: int = 10) -> Dict[str, Any]:
        # Store training data for evaluation function
        self.X_train = X
        self.y_train = y
        
        return super().fit(X, y, optimize, n_iter, init_points)


class FFNNRegressor(BaseRegressor):

    def _get_model(self, **params) -> MLPRegressor:
        hidden_layer_sizes = self._get_hidden_layer_sizes(int(params['hidden_layer_sizes']))
        activation = self._get_activation(int(params['activation']))
        learning_rate = self._get_learning_rate(int(params['learning_rate']))
        
        return MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            max_iter=int(params['max_iter']),
            alpha=params['alpha'],
            tol=params['tol'],
            random_state=self.random_state,
            solver='adam',
            early_stopping=True,
            beta_1=params['beta_1'],
            beta_2=params['beta_2'],
            validation_fraction=0.1
        )
    
    def _get_parameter_bounds(self) -> Dict[str, Tuple]:
        return {
            'hidden_layer_sizes': (0, 7, int),  # Reduzi para 6 arquiteturas (0-5)
            'activation': (0, 3, int),
            'learning_rate': (0, 2, int),
            'max_iter': (2000, 8000, int),  # Aumentei para convergência melhor
            'alpha': (1e-6, 1e-1),  # Aumentei range para mais regularização
            'tol': (1e-6, 1e-3),  # Ajustei para melhor convergência
            'beta_1': (0.8, 0.999),  # Aumentei para melhor convergência
            'beta_2': (0.8, 0.999)  # Aumentei para melhor convergência
        }
    
    def _evaluate_model(self, **params) -> float:
        model = self._get_model(**params)
        
        # Use negative RMSE for maximization
        scores = cross_val_score(
            model, 
            self.scaler_X.transform(self.X_train), 
            self.scaler_y.transform(self.y_train.values.reshape(-1, 1)).ravel(),
            cv=5, 
            scoring='neg_root_mean_squared_error'
        )
        
        return scores.mean()
    
    def _process_best_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'hidden_layer_sizes': int(params['hidden_layer_sizes']),
            'activation': int(params['activation']),
            'learning_rate': int(params['learning_rate']),
            'max_iter': int(params['max_iter']),
            'alpha': params['alpha'],
            'tol': params['tol'],
            'beta_1': params['beta_1'],
            'beta_2': params['beta_2']
        }
    
    def _get_default_params(self) -> Dict[str, Any]:
        return {
            'hidden_layer_sizes': 1,  # (100,)
            'activation': 1,  # 'logistic'
            'learning_rate': 2,  # 'adaptive'
            'max_iter': 2000,
            'alpha': 1e-4,
            'tol': 1e-4,
            'beta_1': 0.9,
            'beta_2': 0.999
        }
    
    def _get_hidden_layer_sizes(self, index: int) -> Tuple[int, ...]:
        # Map index to hidden layer architecture
        architectures = [
            (50,), (100,), (200,),  # Single layer
            (50, 25), (100, 50), (200, 100),  # Two layers
            (50, 25, 10), (100, 50, 25)  # Three layers
        ]
        return architectures[index]
    
    def _get_activation(self, index: int) -> str:
        #M ap index to activation function
        activations = ['identity', 'logistic', 'tanh', 'relu']
        return activations[index]
    
    def _get_learning_rate(self, index: int) -> str:
        # Map index to learning rate strategy
        strategies = ['constant', 'invscaling', 'adaptive']
        return strategies[index]
    
    def fit(self, X: pd.DataFrame, y: pd.Series, optimize: bool = True, 
            n_iter: int = 50, init_points: int = 10) -> Dict[str, Any]:
        # Store training data for evaluation function
        self.X_train = X
        self.y_train = y
        
        return super().fit(X, y, optimize, n_iter, init_points)
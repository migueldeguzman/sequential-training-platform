"""
Energy Prediction Model for ML Dashboard.

This module implements a machine learning model to predict energy consumption
before running inference, based on model architectural features and prompt characteristics.

Inspired by Caravaca et al. 2025 "From Prompts to Power" which achieves R²=0.92-0.98
using Random Forest with architectural + prompt features.

The predictor enables:
- Pre-inference energy estimation for resource planning
- Model selection based on predicted energy consumption
- What-if analysis for different configurations
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .prompt_features import PromptFeatures, extract_prompt_features

logger = logging.getLogger(__name__)


@dataclass
class EnergyPrediction:
    """Result of energy prediction."""

    predicted_total_energy_mj: float
    predicted_prefill_energy_mj: float
    predicted_decode_energy_mj: float
    predicted_energy_per_token_mj: float
    confidence_interval_95_pct: Tuple[float, float]  # (lower, upper) bounds
    model_accuracy_r2: float  # R² score of the trained model
    features_used: List[str]
    prediction_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "predicted_total_energy_mj": self.predicted_total_energy_mj,
            "predicted_prefill_energy_mj": self.predicted_prefill_energy_mj,
            "predicted_decode_energy_mj": self.predicted_decode_energy_mj,
            "predicted_energy_per_token_mj": self.predicted_energy_per_token_mj,
            "confidence_interval_95_pct": list(self.confidence_interval_95_pct),
            "model_accuracy_r2": self.model_accuracy_r2,
            "features_used": self.features_used,
            "prediction_notes": self.prediction_notes,
        }


class SimpleLinearModel:
    """
    Simple linear regression model with JSON serialization.

    Uses a lightweight implementation to avoid pickle security issues
    while maintaining prediction capabilities.
    """

    def __init__(self):
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit linear model with Ridge regression."""
        # Standardize features
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0) + 1e-10  # Avoid division by zero
        X_scaled = (X - self.feature_means) / self.feature_stds

        # Ridge regression with small regularization
        alpha = 0.1
        n_features = X_scaled.shape[1]
        XtX = X_scaled.T @ X_scaled + alpha * np.eye(n_features)
        Xty = X_scaled.T @ y

        # Solve normal equations
        self.coefficients = np.linalg.solve(XtX, Xty)
        self.intercept = np.mean(y - X_scaled @ self.coefficients)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        X_scaled = (X - self.feature_means) / self.feature_stds
        return X_scaled @ self.coefficients + self.intercept

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for JSON serialization."""
        return {
            "coefficients": self.coefficients.tolist() if self.coefficients is not None else None,
            "intercept": float(self.intercept),
            "feature_means": self.feature_means.tolist() if self.feature_means is not None else None,
            "feature_stds": self.feature_stds.tolist() if self.feature_stds is not None else None,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimpleLinearModel":
        """Load model from dictionary."""
        model = cls()
        model.coefficients = np.array(data["coefficients"]) if data["coefficients"] else None
        model.intercept = data["intercept"]
        model.feature_means = np.array(data["feature_means"]) if data["feature_means"] else None
        model.feature_stds = np.array(data["feature_stds"]) if data["feature_stds"] else None
        model.is_fitted = data["is_fitted"]
        return model


class EnergyPredictor:
    """
    Machine learning model for predicting energy consumption.

    Uses architectural features + prompt features to predict energy.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize energy predictor.

        Args:
            model_path: Path to saved model file (json). If None, model must be trained.
        """
        self.model_path = model_path or "backend/profiling/energy_model.json"
        self.model = None
        self.feature_names: List[str] = []
        self.model_r2: float = 0.0
        self.is_trained: bool = False

        # Try to load existing model
        if Path(self.model_path).exists():
            try:
                self.load_model()
            except Exception as e:
                logger.warning(f"Failed to load model from {self.model_path}: {e}")

    def extract_features(
        self,
        model_features: Dict[str, Any],
        input_tokens: int,
        output_tokens: int,
        batch_size: int = 1,
        prompt_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Extract feature vector for prediction.

        Args:
            model_features: Dictionary of model architectural features
            input_tokens: Number of input tokens (prompt length)
            output_tokens: Number of output tokens to generate
            batch_size: Batch size for inference
            prompt_text: Optional raw prompt text for linguistic feature extraction

        Returns:
            Feature vector as numpy array
        """
        # Model architecture features
        num_layers = model_features.get("num_layers", 0)
        hidden_size = model_features.get("hidden_size", 0)
        intermediate_size = model_features.get("intermediate_size", 0)
        num_attention_heads = model_features.get("num_attention_heads", 0)
        num_key_value_heads = model_features.get("num_key_value_heads", num_attention_heads)
        total_params = model_features.get("total_params", 0)

        # Attention mechanism encoding (one-hot)
        attention_mechanism = model_features.get("attention_mechanism", "MHA")
        is_mha = 1 if attention_mechanism == "MHA" else 0
        is_gqa = 1 if attention_mechanism == "GQA" else 0
        is_mqa = 1 if attention_mechanism == "MQA" else 0

        # MoE flag
        is_moe = 1 if model_features.get("is_moe", False) else 0

        # Derived features (based on paper findings)
        # Paper: layers scale linearly, dimensionality scales quadratically
        layer_complexity = num_layers
        dimension_complexity = hidden_size**2
        total_complexity = num_layers * (hidden_size**2)

        # Prompt features
        total_tokens = input_tokens + output_tokens
        output_ratio = output_tokens / total_tokens if total_tokens > 0 else 0

        # Interaction features
        params_per_million = total_params / 1_000_000
        tokens_times_layers = total_tokens * num_layers
        context_complexity = input_tokens * num_layers * hidden_size

        # Batch size feature
        batch_feature = batch_size

        # Extract prompt linguistic features if text is provided
        pf: Optional[PromptFeatures] = None
        if prompt_text:
            try:
                pf = extract_prompt_features(prompt_text)
            except Exception as e:
                logger.warning(f"Prompt feature extraction failed: {e}")

        # Feature vector (order matters for trained model)
        features = [
            # Basic architecture
            num_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            params_per_million,
            # Attention mechanism
            is_mha,
            is_gqa,
            is_mqa,
            is_moe,
            # Complexity metrics
            layer_complexity,
            dimension_complexity / 1e6,  # Scale down
            total_complexity / 1e9,  # Scale down
            # Prompt features (token-level)
            input_tokens,
            output_tokens,
            total_tokens,
            output_ratio,
            # Interaction features
            tokens_times_layers / 1e6,  # Scale down
            context_complexity / 1e9,  # Scale down
            # Batch size
            batch_feature,
            # Prompt linguistic features (EP-101)
            pf.type_token_ratio if pf else 0.0,
            pf.hapax_ratio if pf else 0.0,
            pf.avg_sentence_length if pf else 0.0,
            pf.instruction_density if pf else 0.0,
            pf.avg_token_entropy if pf else 0.0,
            pf.punctuation_density if pf else 0.0,
            pf.long_word_ratio if pf else 0.0,
            float(pf.code_block_count) if pf else 0.0,
            float(pf.question_count) if pf else 0.0,
        ]

        self.feature_names = [
            "num_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "params_millions",
            "is_mha",
            "is_gqa",
            "is_mqa",
            "is_moe",
            "layer_complexity",
            "dimension_complexity_scaled",
            "total_complexity_scaled",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "output_ratio",
            "tokens_times_layers_scaled",
            "context_complexity_scaled",
            "batch_size",
            # Prompt linguistic features
            "prompt_type_token_ratio",
            "prompt_hapax_ratio",
            "prompt_avg_sentence_length",
            "prompt_instruction_density",
            "prompt_avg_token_entropy",
            "prompt_punctuation_density",
            "prompt_long_word_ratio",
            "prompt_code_block_count",
            "prompt_question_count",
        ]

        return np.array(features).reshape(1, -1)

    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train the energy prediction model on historical profiling data.

        Args:
            training_data: List of dicts with 'features' and 'target_energy_mj' keys

        Returns:
            Dictionary with training metrics (r2_score, mae, rmse)
        """
        if len(training_data) < 5:
            raise ValueError("Need at least 5 training samples. Please run more profiling runs.")

        # Extract features and targets
        X = []
        y = []
        for sample in training_data:
            features = sample["features"]
            if isinstance(features, np.ndarray):
                X.append(features.flatten())
            else:
                X.append(np.array(features).flatten())
            y.append(sample["target_energy_mj"])

        X = np.array(X)
        y = np.array(y)

        # Train simple linear model
        self.model = SimpleLinearModel()
        logger.info(f"Training linear model on {len(X)} samples...")
        self.model.fit(X, y)

        # Evaluate
        y_pred = self.model.predict(X)

        # Calculate R² score
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # MAE and RMSE
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        self.model_r2 = r2
        self.is_trained = True

        logger.info(f"Model trained: R²={r2:.4f}, MAE={mae:.2f} mJ, RMSE={rmse:.2f} mJ")

        metrics = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "n_samples": len(training_data),
        }

        return metrics

    def predict(
        self,
        model_features: Dict[str, Any],
        input_tokens: int,
        output_tokens: int,
        batch_size: int = 1,
        prompt_text: Optional[str] = None,
    ) -> EnergyPrediction:
        """
        Predict energy consumption for a given configuration.

        Args:
            model_features: Dictionary of model architectural features
            input_tokens: Number of input tokens (prompt length)
            output_tokens: Number of output tokens to generate
            batch_size: Batch size for inference
            prompt_text: Optional raw prompt text for richer linguistic features

        Returns:
            EnergyPrediction with predicted values and confidence intervals

        Raises:
            ValueError: If model is not trained or loaded
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Model not trained or loaded. Train the model first or load from file.")

        # Extract features (with optional prompt text analysis)
        features = self.extract_features(
            model_features, input_tokens, output_tokens, batch_size, prompt_text
        )

        # Predict total energy
        predicted_total = float(self.model.predict(features)[0])

        # Estimate prefill vs decode split
        # Based on paper: output tokens ~11x more costly than input tokens
        # E_prefill ≈ input_tokens × cost_per_input
        # E_decode ≈ output_tokens × cost_per_output
        # where cost_per_output ≈ 11 × cost_per_input

        total_weighted = input_tokens + (11 * output_tokens)
        prefill_fraction = input_tokens / total_weighted if total_weighted > 0 else 0.5
        decode_fraction = (11 * output_tokens) / total_weighted if total_weighted > 0 else 0.5

        predicted_prefill = predicted_total * prefill_fraction
        predicted_decode = predicted_total * decode_fraction

        # Energy per token
        total_tokens = input_tokens + output_tokens
        predicted_per_token = predicted_total / total_tokens if total_tokens > 0 else 0

        # Confidence interval (approximate using ±20% for linear model)
        ci_lower = predicted_total * 0.8
        ci_upper = predicted_total * 1.2

        # Notes
        notes = None
        if input_tokens > 4096:
            notes = "Long context: actual energy may be higher due to KV cache pressure"
        elif model_features.get("is_moe", False):
            notes = "MoE model: prediction assumes average expert activation"

        return EnergyPrediction(
            predicted_total_energy_mj=predicted_total,
            predicted_prefill_energy_mj=predicted_prefill,
            predicted_decode_energy_mj=predicted_decode,
            predicted_energy_per_token_mj=predicted_per_token,
            confidence_interval_95_pct=(ci_lower, ci_upper),
            model_accuracy_r2=self.model_r2,
            features_used=self.feature_names,
            prediction_notes=notes,
        )

    def save_model(self, path: Optional[str] = None):
        """
        Save trained model to disk.

        Args:
            path: Path to save model (json file). If None, uses self.model_path.
        """
        if self.model is None:
            raise ValueError("No model to save")

        save_path = path or self.model_path

        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model.to_dict(),
            "feature_names": self.feature_names,
            "model_r2": self.model_r2,
            "is_trained": self.is_trained,
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: Optional[str] = None):
        """
        Load trained model from disk.

        Args:
            path: Path to load model from (json file). If None, uses self.model_path.
        """
        load_path = path or self.model_path

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        with open(load_path, "r") as f:
            data = json.load(f)

        self.model = SimpleLinearModel.from_dict(data["model"])
        self.feature_names = data["feature_names"]
        self.model_r2 = data["model_r2"]
        self.is_trained = data["is_trained"]

        logger.info(f"Model loaded from {load_path} (R²={self.model_r2:.4f})")


def prepare_training_data_from_database(db) -> List[Dict[str, Any]]:
    """
    Prepare training data from profiling database.

    Args:
        db: ProfileDatabase instance

    Returns:
        List of training samples with features and target energy
    """
    # Get all completed runs
    runs = db.get_runs(filters={"status": "completed"})

    training_data = []
    predictor = EnergyPredictor()  # Just for feature extraction

    for run in runs:
        # Skip if missing required data
        if not run.get("total_energy_mj") or not run.get("input_token_count") or not run.get("output_token_count"):
            continue

        # Reconstruct model features from run
        model_features = {
            "num_layers": run.get("num_layers", 0),
            "hidden_size": run.get("hidden_size", 0),
            "intermediate_size": run.get("intermediate_size", 0),
            "num_attention_heads": run.get("num_attention_heads", 0),
            "num_key_value_heads": run.get("num_key_value_heads"),
            "total_params": run.get("total_params", 0),
            "attention_mechanism": run.get("attention_mechanism", "MHA"),
            "is_moe": run.get("is_moe", False),
        }

        # Extract features
        features = predictor.extract_features(
            model_features,
            run["input_token_count"],
            run["output_token_count"],
            run.get("batch_size", 1),
        )

        training_data.append(
            {
                "features": features,
                "target_energy_mj": run["total_energy_mj"],
                "run_id": run["run_id"],
            }
        )

    return training_data

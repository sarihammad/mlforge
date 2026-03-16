"""Central configuration for MLForge using Pydantic Settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MLflow configuration
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "adult-income"
    model_name: str = "adult-income-classifier"

    # Model aliases used in MLflow registry
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"

    # Canary deployment
    canary_traffic_split: float = 0.1

    # Drift detection thresholds
    psi_threshold: float = 0.2
    ks_pvalue_threshold: float = 0.05

    # Data
    data_dir: str = "data"

    # Infrastructure
    redis_url: str = "redis://localhost:6379"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"

    # Train/val/test split ratios
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

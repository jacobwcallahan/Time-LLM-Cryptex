from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceConfig:
    """Dataclass holding all CLI arguments for run_inference.py inference script.

    All fields have defaults. You can create instances manually by passing
    only the values you want to override:

        config = InferenceConfig(model_id="my_run", llm_model="LLAMA")
        config = InferenceConfig(data_path="custom.csv", save_path="./output")
    """

    # Required for loading from MLflow
    model_id: str = "test"
    llm_model: str = "LLAMA"

    # Optional overrides
    mlflow_tracking_uri: Optional[str] = None
    data_path: Optional[str] = None
    save_path: Optional[str] = None
    pipeline: bool = False
    experiment_name: Optional[str] = None

    @classmethod
    def from_namespace(cls, ns) -> "InferenceConfig":
        """Build InferenceConfig from argparse.Namespace (e.g. parse_args() output)."""
        return cls(**{k: v for k, v in vars(ns).items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict) -> "InferenceConfig":
        """Build InferenceConfig from a dict (e.g. from YAML/JSON config). Ignores unknown keys."""
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def parse(cls) -> "InferenceConfig":
        """Parse CLI arguments and return an InferenceConfig."""
        import argparse

        parser = argparse.ArgumentParser(description="Time-LLM Inference Script")
        parser.add_argument('--model_id', type=str, required=True, help='MLflow run name/model ID to load model and config from')
        parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM backbone name (should match training)')
        parser.add_argument('--mlflow_tracking_uri', type=str, default=None, help='Optional MLflow tracking URI')
        parser.add_argument('--data_path', type=str, default=None, help='Optional override for input data CSV')
        parser.add_argument('--save_path', type=str, default=None, help='Optional override for output location of inference.csv')
        parser.add_argument('--pipeline', type=bool, default=False, help='Informs that the inference is being run in the pipeline')
        parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name to use for MLflow experiment tracking and logging.')

        args = parser.parse_args()
        return cls.from_namespace(args)

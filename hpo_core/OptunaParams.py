import yaml
from pathlib import Path

class OptunaParams:
    """
    This class is responsible for storing the optuna parameters.
    """
    def __init__(self, yaml_file: str):
        self.yaml_file = yaml_file

    def get_params(self):
        with open(self.yaml_file, "r") as f:
            self.params = yaml.safe_load(f)

    def get_params(self):
        return self.params

    def set_optuna_vars(trial, args, data_path, yaml_file):
        """Sets the optuna variables for the trial into a dictionary.
        The dictionary is then used as arguments for the run_main.py script.
        The values are pulled from the optuna_vars.yaml (or given) 
        file.

        Args:
            trial: Optuna trial object
            data_path: Path to the data
            args: Arguments

        Returns:
            params: Dictionary of trial parameters
        """

        with open(Path("config") / args.yaml_file, "r") as f:
            config = yaml.safe_load(f)

        params = {}

        # Categorical parameters
        for name, values in config.get("categorical", {}).items():
            # If there is only one value, use it twice
            # This is because Optuna requires two values for categorical parameters
            if len(values) == 1:
                params[name] = trial.suggest_categorical(name, values * 2)
                
            else:
                params[name] = trial.suggest_categorical(name, values)

        # Int parameters
        for name, cfg in config.get("int", {}).items():
            # If step is provided, use it to suggest the int parameter
            if "step" in cfg:
                params[name] = trial.suggest_int(
                name,
                int(cfg["low"]),
                int(cfg["high"]),
                step=int(cfg.get("step", 1))
            )
            else:
                params[name] = trial.suggest_int(
                    name,
                    int(cfg["low"]),
                    int(cfg["high"])
                )

        # Float parameters
        for name, cfg in config.get("float", {}).items():
            # If step is provided, use it to suggest the float parameter
            if "step" in cfg:
                params[name] = trial.suggest_float(
                    name,
                    float(cfg["low"]),
                    float(cfg["high"]),
                    step=float(cfg.get("step", 1))
                )
            else:
                # If step is not provided, use the log flag to suggest the float parameter
                params[name] = trial.suggest_float(
                    name,
                    float(cfg["low"]),
                    float(cfg["high"]),
                    log=cfg.get("log", False)
                )

        for name, cfg in config.get("log_float", {}).items():
            params[name] = trial.suggest_float(
                name,
                float(cfg["low"]),
                float(cfg["high"])
                )


        params["target"] = "returns" if args.returns else "close"
        params['target'] = "volatility" if args.volatility else params["target"]
        params["metric"] = "MSE"
        #params["dates"] = f"{ARGS["start"]}_{ARGS["end"]}"
        params["experiment_name"] = args.experiment_name

        #trial.set_user_attr("dates", f"{ARGS["start"]}_{ARGS["end"]}")
        trial.set_user_attr("granularity", args.granularity)
        trial.set_user_attr("aggregate", args.aggregate)
        trial.set_user_attr("target", params["target"])
        trial.set_user_attr("data_type", "returns" if args.returns else "ohlcv")
        trial.set_user_attr("metric", "MSE")
        
        print("--------------------------------\n")
        print("Trial Parameters:")
        for key, value in params.items():
            print(f"{key}: {value}", end=" | ")
        
        print("\n\n--------------------------------")

        return params

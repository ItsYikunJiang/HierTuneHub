# HPSearchSpace

HPSearchSpace is a Python library for defining search spaces for hyperparameter optimization problem.

## Key Features

- Support YAML file for defining search space.
- Support hierarchical search space definition.
- Support optimization libraries including `Hyperopt`, `Optuna` and `FLAML`.

## Usage

[//]: # (TODO: Add usage examples)

First, you should define a search space in a YAML file.
The following is an example a search space for `SVC` and `RandomForestClassifier` in `sklearn` package.

```yaml
---
estimators_group: # Start with an estimator group name
  sklearn.svm.SVC: # estimator full name, which contains the package name and class name
    C: # hyperparameter name
      range: [ 1.0e-10, 1.0 ]  # hyperparameter range, from low to high. For scientific notation,
      # 1e-10 should be written as 1.0e-10 so that YAML parser can parse it as numeric type correctly.
      # For quantized search space, range should be a list consisting of low, high and step such as [ 1.0e-10, 1.0, 1.0e-10 ]
      sampler: "loguniform"  # sampler type
      default: 1.0  # default value, optional
    kernel:
      values: [ "linear", "rbf" ]  # categorical choices
      sampler: "choice"
      default: "rbf"
  sklearn.ensemble.RandomForestClassifier:
    n_estimators:
      range: [ 10, 1000 ]
      sampler: "uniformint"
      default: 10
    max_depth:
      range: [ 2, 32 ]
      sampler: "uniformint"
      default: 5
```

The YAML file may consist of multiple estimator groups. 
In each estimator group, you can have multiple estimators. 
The estimators must be specified with their full name (e.g., `sklearn.svm.SVC`).
For each estimator, you can define hyperparameters with their search space.
For continuous hyperparameters, you can specify `range` and `sampler`.
For categorical hyperparameters, you can specify `values`. `sampler` must be set to `choice`, or you can omit it.
`default` is optional, it is used as the default value when initializing the hyperparameter tuning process.

There are several types of samplers supported in HPSearchSpace:
- `uniform`: Uniform distribution
- `loguniform`: Log-uniform distribution
- `quniform`: Quantized uniform distribution
- `qloguniform`: Quantized log-uniform distribution
- `uniformint`: Uniform integer distribution
- `quniformint`: Quantized uniform integer distribution
- `loguniformint`: Log-uniform integer distribution
- `qloguniformint`: Quantized log-uniform integer distribution
- `choice`: Categorical choices

Then, you can load the search space from the YAML file and use it with hyperparameter optimization libraries.

```python
from HPSearchSpace import SearchSpace
search_space = SearchSpace(config_file="search_space.yaml")
```

An example objective function for hyperparameter optimization can be defined as follows:

```python
def objective(config):
    sampled_estimator_config = config["estimator_group"]
    estimator = sampled_estimator_config["estimator_class"](**sampled_estimator["params"])
    score = cross_val_score(estimator, X, y, cv=5).mean()
    return score
```


Finally, you can use the search space with hyperparameter optimization libraries.

- For `Hyperopt`:
```python
hp_space = search_space.get_hyperopt_space()

from hyperopt import fmin
best = fmin(fn=objective, space=hp_space)
```

- For `Optuna`: 
A conversion is required because optuna's objective function takes in a trial object.
```python
# conversion
def objective_optuna(trial):
    config = search_space.get_optuna_space(trial)
    return objective(config)

# run optimization
import optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective_optuna, n_trials=100)
```

- For `FLAML`:
```python
flaml_space = search_space.get_flaml_space()

# run optimization
from flaml.tune import tune
best_config = tune.run(objective, config=flaml_space)
```

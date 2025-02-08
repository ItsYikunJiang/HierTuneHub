# HPSearchSpace

HPSearchSpace is a Python library for defining search spaces for hyperparameter optimization problem.

## Key Features

- Support YAML file for defining search space.
- Support hierarchical search space definition.
- Support optimization libraries including `Hyperopt`, `Optuna` and `FLAML`.

## Usage

First, you should define a search space in a YAML file.
The following is an example a search space for `SVC` and `RandomForestClassifier` in `sklearn` package.

```yaml
---
estimators_group: # Start with an estimator group name. You can define multiple estimator groups.
  sklearn.svm.SVC: # estimator full name, which contains the package name and class name if you want to directly use it.
    C: # hyperparameter name
      range: [ 1.0e-10, 1.0 ]  # hyperparameter range, from low to high. For scientific notation,
      # 1e-10 should be written as 1.0e-10 so that YAML parser can parse it as numeric type correctly.
      # For quantized search space, range should be a list consisting of low, high and step such as [ 1.0e-10, 1.0, 1.0e-10 ]
      sampler: "loguniform"  # sampler type
    kernel:
      values: [ "linear", "rbf" ]  # categorical choices
      sampler: "choice"
  sklearn.ensemble.RandomForestClassifier:
    n_estimators:
      range: [ 10, 1000 ]
      sampler: "uniformint"
    max_depth:
      range: [ 2, 32 ]
      sampler: "uniformint"
```

The YAML file may consist of multiple estimator groups. 
In each estimator group, you can have multiple estimators. 
The estimators must be specified with their full name (e.g., `sklearn.svm.SVC`).
For each estimator, you can define hyperparameters with their search space.
For continuous hyperparameters, you can specify `range` and `sampler`.
For categorical hyperparameters, you can specify `values`. `sampler` must be set to `choice`, or you can omit it.

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
    estimator_name = sampled_estimator_config["estimator_name"]
    estimator_class = sampled_estimator_config["estimator_class"]
    model = estimator_class(**sampled_estimator_config["params"])
    score = cross_val_score(model, X, y, cv=5).mean()
    return score
```

`config` is a dictionary containing the sampled configuration.
`config["estimator_group"]` is a dictionary containing the sampled estimator configuration in the estimator group.
A raw string of the estimator name is stored in the `estimator_name` field. `estimator_class` is the actual class object of the estimator.
`params` is a dictionary containing the hyperparameters sampled from the search space.



Finally, you can use the search space with hyperparameter optimization libraries.

- For `Hyperopt`:

```python
hp_space = search_space.to_hyperopt()

from hyperopt import fmin

best = fmin(fn=objective, space=hp_space)
```

- For `Optuna`: 
A conversion is required because optuna's objective function takes in a trial object.

```python
# conversion
def objective_optuna(trial):
    config = search_space.to_optuna(trial)
    return objective(config)


# run optimization
import optuna

study = optuna.create_study(direction="minimize")
study.optimize(objective_optuna, n_trials=100)
```

- For `FLAML`:

```python
flaml_space = search_space.to_flaml()

# run optimization
from flaml.tune import tune

best_config = tune.run(objective, config=flaml_space)
```

## Additional Features

- `search_space.select(estimator_list)`: Select a subset of the search space for the given list of estimator classes.
- `search_space.join(other_search_space)`: Join the search space with another search space.
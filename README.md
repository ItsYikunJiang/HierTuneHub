# HPSearchSpace

HPSearchSpace is a Python library for defining search spaces for hyperparameter optimization problem.

## Key Features

- Support YAML file for defining search space.
- Support hierarchical search space definition.
- Support optimization libraries including `Hyperopt`, `Optuna` and `FLAML`.

## Usage

### Search Space

First, you should define a search space in a YAML file.

- The YAML file may consist of multiple levels of hierarchy.
- If the level is defined using a list, each item in the list should be a dictionary, and each item is considered a possible choice.
In the sampling process, one of the items is randomly selected.
- If the level is defined using a dictionary, all key-value pairs will be included in the sampled configuration.
- For a hyperparameter that needs to be tuned, you need to specify `range`, `values` and/or `sampler` as a dictionary at the lowest level.
  - For continuous hyperparameters, you need to specify `range` and `sampler`. 
  - For categorical hyperparameters, you need to specify `values`. `sampler` must be set to `choice`, or you can omit it.
- For `Hyperopt` and `Optuna`, they need a unique identifier for every possible hyperparameter. Therefore, the defined search space must satisfy the following conditions:
  - A unique identifier key-value pair must be provided in the dictionary if it is a possible choice from a list level. 
  The key name must be the same across the whole file. The value must be unique across all the choices.
  It is required by `Hyperopt` and `Optuna` to identify the configuration.
    - Default key name is `name`, or you can use other identifiers such as `id` or `class` pass `name="id"` or `name="class"` to the `SearchSpace` constructor.
  - A unique character string that is not present in any other keys. Default is `?`.

The following types of samplers are supported in `HPSearchSpace`:
- `uniform`: Uniform distribution
- `loguniform`: Log-uniform distribution
- `quniform`: Quantized uniform distribution
- `qloguniform`: Quantized log-uniform distribution
- `uniformint`: Uniform integer distribution
- `quniformint`: Quantized uniform integer distribution
- `loguniformint`: Log-uniform integer distribution
- `qloguniformint`: Quantized log-uniform integer distribution
- `choice`: Categorical choices

`SearchSpace` class provides the following methods:

- `to_hyperopt`: Convert the search space to a dict for `Hyperopt`.
- `to_optuna`: Convert the search space to `Optuna`. You need to pass in an `Optuna.Trial` object.
- `to_flaml`: Convert the search space to a dict for `FLAML`.

### Hyperparameter Tuning

After you define the search space, you can use it for hyperparameter optimization. `HPSearchSpace` provides a unified interface for hyperparameter optimization libraries including `Hyperopt`, `Optuna`, and `FLAML`.

You need to define your own objective function that takes in a sampled configuration and returns either a single score or a dictionary containing scores and other information.

You can call `create_tuner` function to create a `Tuner` class with the search space and the optimization library you want to use.

```python
def create_tuner(
        objective: Callable,
        search_space: SearchSpace,
        mode: str = "min",
        metric: Optional[str] = None,
        framework: str = "hyperopt",
        framework_params: dict = None,
        **kwargs
) -> Tuner:
    """
    Create a tuner object based on the specified framework.
    :param objective: user-defined objective function. The objective function can ouput a single float value, or a
    dictionary containing the metric value and other values.
    :param search_space: A SearchSpace object containing the search space for the hyperparameters.
    :param mode: The optimization mode. Either 'min' or 'max'.
    :param metric: If the objective function returns a dictionary, the metric key specifies the key to be used for
    optimization.
    :param framework: The framework to be used for optimization. Supported frameworks are "hyperopt", "optuna", and
    "flaml".
    :param framework_params: Additional parameters to be passed to the framework tuning function.
    :param kwargs: Additional parameters to be passed to the framework tuning function.
    :return: A Tuner object based on the specified framework.
    """
```

`Tuner` class provides `run` method to start the optimization process. 
After the optimization process is finished, you can get the best hyperparameters and the best result by calling `best_params` and `best_result` properties.

You can also call `trials` property to get all the trials evaluated during the optimization process. 
It is a list of `Trial` objects which contains `params` and `result` attributes corresponding to the sampled configuration and the result of objective function.

## Example

The following is an example to tune hyperparameters for different classifiers using unified interface provided by `HPSearchSpace`.

The search space is defined in a YAML file as follows:

```yaml
---
estimators:
  - name: "sklearn.svm.SVC" # estimator full name
    C: # hyperparameter name
      range: [ 1.0e-10, 1.0 ]  # hyperparameter range, from low to high. For scientific notation,
      # 1e-10 should be written as 1.0e-10 so that YAML parser can parse it as numeric type correctly.
      sampler: "loguniform"  # sampler type
    kernel:
      - name: "linear"
      - name: "poly"
        degree:
          range: [ 2, 5 ]
          sampler: "uniformint"
        gamma:
          values: [ "auto", "scale" ]
      - name: "rbf"
        gamma:
          values: [ "auto", "scale" ]
          sampler: "loguniform"
  - name: "sklearn.ensemble.RandomForestClassifier"
    n_estimators:
      range: [ 10, 1000 ]
      sampler: "uniformint"
    max_depth:
      range: [ 2, 32 ]
      sampler: "uniformint"
  - name: "sklearn.ensemble.GradientBoostingClassifier"
    n_estimators:
      range: [ 10, 1000 ]
      sampler: "uniformint"
    max_depth:
      range: [ 2, 32 ]
      sampler: "uniformint"
  - name: "sklearn.neighbors.KNeighborsClassifier"
    n_neighbors:
      range: [ 2, 10 ]
      sampler: "uniformint"
```

The objective function is defined as follows:

```python
import time

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
x, y = iris.data, iris.target

def objective(config):
    config = config['estimators']
    name = config.pop("name")
    if name == "sklearn.svm.SVC":
        c = config.pop("C")
        kernel = config['kernel'].pop("name")
        kernel_params = config['kernel']
        model = SVC(C=c, kernel=kernel, **kernel_params)
    elif name == "sklearn.ensemble.RandomForestClassifier":
        model = RandomForestClassifier(**config)
    elif name == "sklearn.ensemble.GradientBoostingClassifier":
        model = GradientBoostingClassifier(**config)
    elif name == "sklearn.neighbors.KNeighborsClassifier":
        model = KNeighborsClassifier(**config)
    else:
        raise ValueError(f"Unknown estimator: {config['estimator']}")
    
    t_start = time.time()
    acc = cross_val_score(model, x, y, cv=5).mean()
    t_end = time.time()
    return {
        'acc': acc, 
        'time': t_end - t_start
    }
```

Now, we can create a tuner object and run the optimization process:

```python
search_space = SearchSpace("example.yaml")
hyperopt_tuner = create_tuner(objective,
                              search_space,
                              mode="max",
                              metric="acc",
                              framework="hyperopt",
                              max_evals=10  # number of evaluation times
                              )
hyperopt_tuner.run()
```

In the end, we can get the best hyperparameters and the best result:

```python
best_params = hyperopt_tuner.best_params
best_result = hyperopt_tuner.best_result

print(best_params)
print(best_result)

# Output (may vary):
# {'estimators': {'C': 1.7454438588621903, 'kernel': {'degree': np.int64(2), 'gamma': 'scale', 'name': 'poly'}, 'name': 'sklearn.svm.SVC'}}
# {'acc': np.float64(-0.9866666666666667), 'time': 0.008997917175292969}
```
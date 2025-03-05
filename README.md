# HierTuneHub

Are you tired of babysitting hyperparameter tuning process? 
Are you confused by all the hyperparameter optimization libraries out there and don't know which one to choose?
Are you struggling to translate your hierarchical search space (dependencies of different hyperparameters) into a format that can be used by different optimization libraries?
Don't worry! `HierTuneHub` is here to help you!

By simply defining a search space in a YAML file and black-box objective function,
you can use `HierTuneHub` to tune hyperparameters with popular optimization libraries including `Hyperopt`, `Optuna`, and `FLAML` (more to come!) with ease.

## Key Features

- Search space definition in a YAML file
  - Support for multiple levels of hierarchy in the search space
  - Support all types of hyperparameters including continuous, integer, and categorical
- Unified interface for hyperparameter optimization libraries
- Easy-to-use API
- Support for optimization libraries including `Hyperopt`, `Optuna`, and `FLAML` (more to come!)

## Installation

Currently, `HierTuneHub` is not available on PyPI. You can install it from GitHub:

```bash
pip install git+https://github.com/ItsYikunJiang/HierTuneHub.git
```

After installation, you can import the package in your Python code:

```python
import hiertunehub
```

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
    - Default key name is `name`, or you can use other identifiers such as `id` or `class` and pass `name="id"` or `name="class"` to the `SearchSpace` constructor.
  - A unique character string that is not present in any other keys. It is used to concatenate the keys to form a unique identifier. 
  Default is `?`, or you can use other characters such as `!` and pass `sep="!"` to the `SearchSpace` constructor.

The following types of samplers are supported in `HierTuneHub`:
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

`SearchSpace` also provides the following class methods:
- `from_dict`: Create a `SearchSpace` object from a dictionary resembling the YAML configuration.
- `from_flaml`: Create a `SearchSpace` object from a FLAML configuration.

### Hyperparameter Tuning

After you define the search space, you can use it for hyperparameter optimization. 
`HierTuneHub` provides a unified interface for hyperparameter optimization libraries including `Hyperopt`, `Optuna`, and `FLAML`.

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

`Tuner` class also provides `results` property to get all the results evaluated during the optimization process in a list.

Note for passing additional parameters (`framework_params` and `kwargs`) to the optimization libraries:
- `optuna`:
  - `sampler`, `pruner` and `study_name` are passed to `optuna.create_study` function. Other parameters are passed to `optuna.study.optimize` function.
- `hyperopt`:
  - all parameters are passed to `hyperopt.fmin` function.
- `FLAML`:
  - all parameters are passed to `flaml.tune.run` function.

## Example

The following is an example of iris dataset classification problem. 
We want to tune hyperparameters for different classifiers using unified interface provided by `HierTuneHub` to find the best hyperparameters that maximize the accuracy.

The search space is defined in a YAML file as follows. 
It consists of four classifiers: `SVC`, `RandomForestClassifier`, `GradientBoostingClassifier`, and `KNeighborsClassifier`.
Each classifier has its own hyperparameters to be tuned.

```yaml
---
estimators:
  - name: "sklearn.svm.SVC" # estimator name
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
          values: [ "auto", "scale" ]  # categorical choices
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

The objective function is defined as follows. The function takes in a configuration dictionary and returns a dictionary containing the accuracy and the time taken to evaluate the model.

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

Now, we can load the search space, create a tuner object and run the optimization process:

```python
from hiertunehub import SearchSpace, create_tuner

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
# {'estimators': {'C': 1.9537341171427107, 'kernel': {'gamma': 'auto', 'name': 'rbf'}, 'name': 'sklearn.svm.SVC'}}
# {'acc': np.float64(0.9800000000000001), 'time': 0.012115001678466797}
```
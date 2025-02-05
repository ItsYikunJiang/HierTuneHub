from hyperopt import hp
from hyperopt.pyll import scope
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

def convert_to_hyperopt_space(param_dict):
    model_mapping = {
        "sklearn.svm.SVC": SVC,
        "sklearn.ensemble.RandomForestClassifier": RandomForestClassifier,
        "sklearn.ensemble.GradientBoostingClassifier": GradientBoostingClassifier,
        "sklearn.neighbors.KNeighborsClassifier": KNeighborsClassifier
    }
    
    def get_hp_sampler(name, param):
        if param["sampler"] == "loguniform":
            return hp.loguniform(name, param["range"][0], param["range"][1])
        elif param["sampler"] == "uniformint":
            return scope.int(hp.quniform(name, param["range"][0], param["range"][1], 1))
        elif param["sampler"] == "choice":
            return param["values"][0]  # Single value, so it's fixed
        else:
            raise ValueError(f"Unknown sampler type: {param['sampler']}")
    
    space = hp.choice("classifier", [
        {
            "model": model_mapping[model],
            "params": {
                param: get_hp_sampler(f"{model.split('.')[-1].lower()}_{param}", details)
                for param, details in params.items()
            }
        }
        for model, params in param_dict["estimators"].items()
    ])
    
    return space

def suggest_classifier(trial_, param_config):
    classifier_map = {
        'sklearn.svm.SVC': ('SVC', SVC),
        'sklearn.ensemble.RandomForestClassifier': ('RandomForest', RandomForestClassifier),
        'sklearn.ensemble.GradientBoostingClassifier': ('GradientBoosting', GradientBoostingClassifier),
        'sklearn.neighbors.KNeighborsClassifier': ('KNeighbors', KNeighborsClassifier)
    }
    
    classifier_name = trial_.suggest_categorical("classifier", [v[0] for v in classifier_map.values()])
    sk_classifier = next(k for k, v in classifier_map.items() if v[0] == classifier_name)
    params = param_config['estimators'][sk_classifier]
    
    param_values = {}
    for param, config in params.items():
        if config['sampler'] == 'loguniform':
            param_values[param] = trial_.suggest_float(param, *config['range'], log=True)
        elif config['sampler'] == 'uniformint':
            param_values[param] = trial_.suggest_int(param, *config['range'])
        elif config['sampler'] == 'choice':
            param_values[param] = trial_.suggest_categorical(param, config['values'])
        else:
            param_values[param] = config['default']
    
    return classifier_map[sk_classifier][1](**param_values)

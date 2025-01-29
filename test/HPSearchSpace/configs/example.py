config = {
    'estimators': {
        'sklearn.ensemble.RandomForestRegressor': {
            'n_estimators': {'range': [10, 100], 'sampler': 'uniformint', 'default': 10},
            'max_depth': {'range': [1, 10], 'sampler': 'uniformint', 'default': 5}
        },
        'sklearn.linear_model.LogisticRegression': {
            'C': {'range': [0.1, 1.0], 'sampler': 'uniform', 'default': 1.0},
            'penalty': {'values': ['l1', 'l2', 'elasticnet'], 'sampler': 'choice', 'default': 'l2'}
        }
    }
}

config = {
    'estimators': {
        'sklearn.ensemble.RandomForestRegressor': {
            'n_estimators': {'range': [10, 100], 'sampler': 'uniformint'},
            'max_depth': {'range': [1, 10], 'sampler': 'uniformint'}
        },
        'sklearn.linear_model.LogisticRegression': {
            'C': {'range': [0.1, 1.0], 'sampler': 'uniform'},
            'penalty': {'values': ['l1', 'l2', 'elasticnet'], 'sampler': 'choice'}
        }
    }
}

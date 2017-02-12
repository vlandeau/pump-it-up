import time

from hyperopt import hp, fmin, tpe
from hyperopt.base import Trials, STATUS_OK
from hyperopt.fmin import space_eval
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from stacked_generalization.lib.stacking import StackedClassifier

import utils

RANDOM_FOREST_LABEL = 'rf'
RANDOM_FOREST_PARAMETERS_SPACE = {
    'rf__max_depth': hp.choice('max_depth', list(range(2, 10)) + [None]),
    'rf__max_features': hp.choice('max_features', [None, 'sqrt', 'log2']),
    'rf__n_estimators': hp.choice('n_estimators', range(50, 1000)),
    'rf__criterion': hp.choice('criterion', ["gini", "entropy"]),
    'rf__bootstrap': hp.choice('bootstrap', [False, True])
}


class ParameterTuner:
    def __init__(self, df_train, y_train):
        self.tuned_models_params = dict()
        self.stacked_model = None
        self.tuned_pipeline_params = dict()
        self.df_train = df_train
        self.x_train = None
        self.y_train = y_train

    def tune_pipeline_parameters(self,
                                 pipeline,
                                 space_pipeline,
                                 cv=5, max_evals=15,
                                 n_jobs=4,
                                 space_rf=RANDOM_FOREST_PARAMETERS_SPACE,
                                 best_params_filepath='best_params.json'
                                 ):
        rf = RandomForestClassifier()
        pipeline_to_optimize = Pipeline(pipeline.steps + [(RANDOM_FOREST_LABEL, rf)])

        space = space_pipeline.copy()
        space.update(space_rf)
        space[RANDOM_FOREST_LABEL + '__n_jobs'] = n_jobs

        best_params, trials = self._get_best_params(pipeline_to_optimize,
                                                    space=space,
                                                    cv=cv,
                                                    max_evals=max_evals)

        pipeline_to_optimize.set_params(**best_params)
        (rf_name, tuned_rf) = pipeline_to_optimize.steps.pop(-1)
        self.tuned_pipeline_params = pipeline_to_optimize.get_params()
        self.tuned_models_params[tuned_rf.__class__.__name__] = tuned_rf.get_params()
        utils.save_dict_to_file(dict=best_params,
                                filepath=best_params_filepath)
        return best_params, trials

    def tune_model_parameters(self, model, space, cv=3, max_evals=15):
        best_params, trials = self._get_best_params(model, space, cv, max_evals)
        model_name = model.__class__.__name__
        self.tuned_models_params[model_name] = best_params
        return best_params, trials

    def stack_models(self):
        classifiers = [model(self.tuned_models_params)
                       for model in self.tuned_models_params.keys()]
        base_classifier = LogisticRegression(random_state=1)
        self.stacked_model = StackedClassifier(base_classifier,
                                               classifiers,
                                               verbose=1)
        self.stacked_model.fit(self.df_train, self.y_train)
        return self.stacked_model

    def _get_best_params(self, estimator_to_optimize, space, cv, max_evals):
        objective = self._get_objective_function(estimator_to_optimize, cv)
        trials = Trials()
        best_space_params = fmin(fn=objective,
                                 space=space,
                                 algo=tpe.suggest,
                                 max_evals=max_evals,
                                 trials=trials)
        best_params = space_eval(space, best_space_params)
        return best_params, trials

    def _get_objective_function(self, estimator_to_optimize, cv):
        def objective(current_space):
            estimator = estimator_to_optimize.set_params(**current_space)
            scores = cross_val_score(estimator, self.df_train, self.y_train, cv=cv)
            print "Iteration %s: score %s with std %s for params %s" % (objective.current_iteration,
                                                                        scores.mean(),
                                                                        scores.std(),
                                                                        str(current_space))
            objective.current_iteration += 1
            return {'loss': 1 - scores.mean(),
                    'status': STATUS_OK,
                    'eval_time': time.time()}

        objective.current_iteration = 0
        return objective

    def set_x_train(self, x_train):
        self.x_train = x_train

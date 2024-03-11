import pandas as pd
import numpy as np
import joblib
import time
import logging
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import (
    train_test_split,
    KFold,
    ShuffleSplit,
    RandomizedSearchCV,
)

from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import optuna
from optuna.samplers import TPESampler  # , RandomSampler
from optuna.pruners import MedianPruner
import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation, record_evaluation

from ml_visualize import plot_confusion_matrix, plot_multiple_subplots_calc_pred
from ml_analyse import compute_scores_regression


def scale_data(df, scaler):
    """_summary_
    Scales the data using the provided scaler.
    If the data is scaled, then the custom metric should be used to evaluate the model

    Args:
        df (_type_): _description_
        scaler (_type_): _description_

    Returns:
        _type_: _description_
    """
    # standard_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    # quantile_transformer = QuantileTransformer(random_state=42)
    # power_transformer = PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
    # robust_scaler = RobustScaler(quantile_range=(25, 75), with_centering=True, with_scaling=True, copy=True)
    df_scaled = df.copy()
    df_scaled.iloc[:, :-4] = scaler.fit_transform(df_scaled.iloc[:, :-4])
    return df_scaled


# scale target data?
def scale_target_data(df, scaler, reverse=False):
    """_summary_
    Only scales the taget data, i.e the pKa values
    Args:
        df (_type_): _description_
        scaler (_type_): _description_
        reverse (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # qt = QuantileTransformer(output_distribution='normal')
    df_scaled = df.copy()
    if reverse:
        df_scaled["label"] = scaler.inverse_transform(
            df_scaled["label"].values.reshape(-1, 1)
        )
    else:
        df_scaled["label"] = scaler.fit_transform(
            df_scaled["label"].values.reshape(-1, 1)
        )
    return df_scaled


def process_data(
    indices,
    df_source,
    col_name,
    feature_col,
    label_col,
    atom_index_col,
    regression,
):
    data = {
        "features": [],
        "labels": [],
        "atom_indices": [],
        "idx_names": [],
        "names": [],
    }

    df_loc = df_source.loc[indices, [col_name, feature_col, label_col, atom_index_col]]

    for idx, row in df_loc.iterrows():
        for desc, label, atom_idx in zip(
            row[feature_col], row[label_col], row[atom_index_col]
        ):
            if regression and label == float("inf"):
                continue
            elif not regression and label == -1:
                continue

            data["features"].append(desc)
            data["labels"].append(label)
            data["atom_indices"].append(atom_idx)
            data["idx_names"].append(idx)
            data["names"].append(row[col_name])

    return data


def create_folds_molecular(df, n_splits=5, test_size=0.2, random_state=42):
    """_summary_

    Args:
        df (_type_): _description_
        n_splits (int, optional): _description_. Defaults to 5.
        test_size (float, optional): _description_. Defaults to 0.2.
        random_state (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
        a list of train folds indexes,
        a list of val folds indexes,
        a list of test indices

    Example:
    # Create DataFrame
    df = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50),
        'target': np.random.rand(50)
    })

    train_folds, val_folds, test_indices = create_folds(df, n_splits=5, test_size=0.2, random_state=42)

    print("Train folds:", train_folds)
    print("Validation folds:", val_folds)
    print("Test indices:", test_indices)
    """
    # Add empty 'train_test' column to df
    df["train_test"] = np.nan

    # Split into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, shuffle=True, random_state=random_state
    )

    # Mark the test set in the original DataFrame
    df.loc[train_df.index, "train_test"] = "train"
    df.loc[test_df.index, "train_test"] = "test"

    # Prepare for KFold
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kf = ShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=random_state)

    # Initialize empty lists to store train and val indices
    train_folds = [[] for _ in range(n_splits)]
    val_folds = [[] for _ in range(n_splits)]

    # Iterate through KFold indices
    for fold, (train_index, val_index) in enumerate(kf.split(train_df)):
        train_folds[fold] = train_df.iloc[train_index].index.tolist()
        val_folds[fold] = train_df.iloc[val_index].index.tolist()

    return train_folds, val_folds, train_df.index.tolist(), test_df.index.tolist()


def create_train_test_df_molecular(
    indices,
    df_source,
    col_name="names",
    feature_col="descriptor_vector_conf20",
    label_col="lst_pka_sp_lfer",
    atom_index_col="atom_indices_conf20",
    regression=True,
):
    """_summary_

    Args:
        test_indices (_type_): _description_
        df_source (_type_): _description_
        feature_col (str, optional): _description_. Defaults to 'descriptor_vector_conf20'.
        label_col (str, optional): _description_. Defaults to 'lst_pka_sp_lfer'.
        atom_index_col (str, optional): _description_. Defaults to 'atom_indices_conf20'.

    Returns:
        _type_: _description_
    """
    data = process_data(
        indices, df_source, col_name, feature_col, label_col, atom_index_col, regression
    )

    df = pd.DataFrame(
        data["features"],
        columns=[f"feature_{i}" for i in range(len(data["features"][0]))],
    )
    df["label"] = data["labels"]
    df["atom_index"] = data["atom_indices"]
    df["idx_name"] = data["idx_names"]
    df["names"] = data["names"]

    return df


def create_folds_df_molecular(
    folds,
    df_source,
    col_name="names",
    feature_col="descriptor_vector_conf20",
    label_col="lst_pka_sp_lfer",
    atom_index_col="atom_indices_conf20",
    regression=True,
):
    """_summary_

    Args:
        folds (_type_): _description_
        df_source (_type_): _description_
        feature_col (str, optional): _description_. Defaults to 'descriptor_vector_conf20'.
        label_col (str, optional): _description_. Defaults to 'lst_pka_sp_lfer'.
        atom_index_col (str, optional): _description_. Defaults to 'atom_indices_conf20'.

    Returns:
        _type_: _description_
        A list of DataFrames containing the folds
    """

    df_folds_list = []

    for fold in folds:
        data = process_data(
            fold,
            df_source,
            col_name,
            feature_col,
            label_col,
            atom_index_col,
            regression,
        )

        fold_df = pd.DataFrame(
            data["features"],
            columns=[f"feature_{i}" for i in range(len(data["features"][0]))],
        )
        fold_df["label"] = data["labels"]
        fold_df["atom_index"] = data["atom_indices"]
        fold_df["idx_name"] = data["idx_names"]
        fold_df["names"] = data["names"]

        df_folds_list.append(fold_df)

    return df_folds_list


def weighted_rmse(preds, train_data):
    labels = train_data.get_label()

    weights = np.ones_like(labels)
    weights[labels <= 40] = (
        2  # Double the weight for entries with labels in the first 40 bins
    )

    # Calculate weighted RMSE
    weighted_square_diffs = weights * ((labels - preds) ** 2)
    mean_weighted_square_diff = np.sum(weighted_square_diffs) / np.sum(weights)
    weighted_rmse_val = np.sqrt(mean_weighted_square_diff)

    return "Weighted_RMSE", weighted_rmse_val, False


def original_scale_metrics(preds, train_data, transformer):
    """_summary_
    Used it as a feval function in LGBM if the data has been transformed
    Args:
        preds (_type_): _description_
        train_data (_type_): _description_
        transformer (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Convert predictions back to original scale
    original_preds = transformer.inverse_transform(preds.reshape(-1, 1)).flatten()
    original_labels = transformer.inverse_transform(
        train_data.get_label().reshape(-1, 1)
    ).flatten()

    # Calculate MAE and RMSE on original scale
    mae_value = mean_absolute_error(original_labels, original_preds)
    rmse_value = mean_squared_error(original_labels, original_preds, squared=False)

    return [
        ("Original_Scale_MAE", mae_value, False),
        ("Original_Scale_RMSE", rmse_value, False),
    ]


def train_single_fold(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    params=None,
    n_iter=10000,
    verbose_eval=1,
    early_stop=True,
    custom_metric=None,
    transformer=None,
):
    """
    Trains a single fold of data using LightGBM.

    if verbose_eval > 0 then the evals_result is returned.
    if verbose_eval = 1 then the evals_result is returned every 1 iteration.
    """
    # Prepare datasets
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = (
        lgb.Dataset(X_val, y_val, reference=lgb_train) if X_val is not None else None
    )

    valid_sets, valid_names = prepare_validation_sets(lgb_train, lgb_val)

    evals_result = {}
    callbacks = prepare_callbacks(verbose_eval, early_stop, evals_result)

    # Use transformer if provided
    if transformer:
        custom_metric = lambda preds, train_data: original_scale_metrics(
            preds, train_data, transformer
        )

    # Train the model
    if not params:
        params = {}

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=n_iter,
        valid_sets=valid_sets,
        valid_names=valid_names,
        feval=custom_metric,
        callbacks=callbacks,
    )

    return model, evals_result


def prepare_validation_sets(lgb_train, lgb_val):
    """
    Prepares validation sets for training.
    """
    valid_sets = [lgb_train]
    valid_names = ["train"]
    if lgb_val:
        valid_sets.append(lgb_val)
        valid_names.append("valid")
    return valid_sets, valid_names


def prepare_callbacks(verbose_eval, early_stop, evals_result):
    """
    Prepares callbacks for training.
    """
    callbacks = []
    if early_stop:
        callbacks.append(
            early_stopping(stopping_rounds=250, first_metric_only=False, verbose=True)
        )
    if verbose_eval > 0:
        callbacks.append(log_evaluation(period=verbose_eval))
    callbacks.append(record_evaluation(evals_result))
    return callbacks


def train_lgbm_generator(
    X,
    y,
    params,
    n_folds=5,
    seed=42,
    cv="kfold",
    verbose_eval=-1,
    n_iter=10000,
    early_stop=True,
    df_train_folds=None,
    df_val_folds=None,
    custom_metric=None,
    transformer=None,
):
    """
    Trains a LightGBM model using various cross-validation strategies.
    """

    if cv == "kfold":
        yield from train_using_kfold_cv(
            X,
            y,
            params,
            n_folds,
            seed,
            verbose_eval,
            n_iter,
            early_stop,
            custom_metric,
            transformer,
        )

    elif cv == "molecular":
        yield from train_using_molecular_cv(
            df_train_folds,
            df_val_folds,
            params,
            n_iter,
            verbose_eval,
            early_stop,
            custom_metric,
            transformer,
        )

    else:
        yield from train_without_cv(
            X,
            y,
            params,
            seed,
            verbose_eval,
            n_iter,
            early_stop,
            custom_metric,
            transformer,
        )


def train_using_kfold_cv(
    X,
    y,
    params,
    n_folds,
    seed,
    verbose_eval,
    n_iter,
    early_stop,
    custom_metric,
    transformer,
):
    """
    Trains using KFold cross-validation.
    Need to check if this takes the correct columns
    """

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model, evals_result = train_single_fold(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params=params,
            n_iter=n_iter,
            verbose_eval=verbose_eval,
            early_stop=early_stop,
            custom_metric=custom_metric,
            transformer=transformer,
        )

        yield i, model, evals_result, model.best_score, model.best_iteration, X_train, y_train, X_val, y_val


def train_using_molecular_cv(
    df_train_folds,
    df_val_folds,
    params,
    n_iter,
    verbose_eval,
    early_stop,
    custom_metric,
    transformer,
):
    """
    Trains using molecular cross-validation.
    X_train and X_val removes the last 4 columns, which are the idx_name, comp_name, atom_index, and label
    """

    for i, (train_fold_df, val_fold_df) in enumerate(zip(df_train_folds, df_val_folds)):
        # Extract features and labels
        X_train = train_fold_df.iloc[:, :-4]
        y_train = train_fold_df["label"]
        X_val = val_fold_df.iloc[:, :-4]
        y_val = val_fold_df["label"]

        model, evals_result = train_single_fold(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params=params,
            n_iter=n_iter,
            verbose_eval=verbose_eval,
            early_stop=early_stop,
            custom_metric=custom_metric,
            transformer=transformer,
        )

        yield i, model, evals_result, model.best_score, model.best_iteration, X_train, y_train, X_val, y_val


def train_without_cv(
    X, y, params, seed, verbose_eval, n_iter, early_stop, custom_metric, transformer
):
    """
    Trains without cross-validation.
    if early_stop = True, then a validation set is needed
    and splits the data into train and val sets
    """

    if early_stop:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=seed, shuffle=True
        )

        model, evals_result = train_single_fold(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params=params,
            n_iter=n_iter,
            verbose_eval=verbose_eval,
            early_stop=early_stop,
            custom_metric=custom_metric,
            transformer=transformer,
        )

        yield 0, model, evals_result, model.best_score, model.best_iteration, X_train, y_train, X_val, y_val

    else:
        model, evals_result = train_single_fold(
            X_train=X,
            y_train=y,
            X_val=None,
            y_val=None,
            params=params,
            n_iter=n_iter,
            verbose_eval=verbose_eval,
            early_stop=False,
            custom_metric=custom_metric,
            transformer=transformer,
        )

        yield 0, model, evals_result, model.best_score, model.best_iteration, X, y, None, None


def objective_optuna(
    trial,
    metric=["l1", "rmse"],
    cv="molecular",
    transformer=None,
    reg_obj="regression",
    boost="gbdt",
):
    """_summary_

    max depth should be np.log2(n) where == #n nodes. i.e. np.log2(3121) = 11. conseqently 10 to avoid overfitting
    num_leaves should be 2**max_depth
    """

    boosting_types = [boost]  # ["gbdt", "rf", "dart"], ["gbdt", "dart"]
    boosting_type = trial.suggest_categorical("boosting", boosting_types)
    # early_stopping_rounds = 50
    # if cv == "molecular":
    #     X_train = None
    #     y_train = None

    # "metric": ['mae', 'rmse'],
    # for scaled data set custom metric to None, transformer=qt, "metric": ['custom']
    # if weighted RMSE is used, then custom metric should be weigthed_rmse

    custom_metric = None

    if metric == ["custom", "l1"]:
        custom_metric = weighted_rmse
    elif metric == ["custom"] and transformer:
        custom_metric = original_scale_metrics

    FIXED_PARAMS = {
        "objective": reg_obj,
        "metric": metric,
        "random_state": 42,
    }

    # Search space for regression
    if reg_obj == "regression":
        SEARCH_PARAMS = {
            "boosting_type": boosting_type,
            "num_leaves": trial.suggest_int("num_leaves", 8, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 300),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-5, 10, log=True
            ),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "max_bin": trial.suggest_int("max_bin", 100, 1000),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "verbosity": -1,
        }

    # Search space for binary classification
    if reg_obj == "binary":
        pos_samples = np.sum(y_train == 1)
        neg_samples = np.sum(y_train == 0)
        SEARCH_PARAMS = {
            "boosting_type": boosting_type,
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                np.sqrt(neg_samples / pos_samples) + 1,
                neg_samples / pos_samples,
            ),
            "num_leaves": trial.suggest_int("num_leaves", 8, 2000),
            "max_depth": trial.suggest_int("max_depth", 2, 11),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 300),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-5, 10, log=True
            ),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "max_bin": trial.suggest_int("max_bin", 100, 1000),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "verbosity": -1,
        }

    params = {**FIXED_PARAMS, **SEARCH_PARAMS}

    # perform cross-validation
    lst_mean_RMSE = []
    lst_mean_weighted_RMSE = []
    lst_mean_org_RMSE = []
    lst_mean_binary_logloss = []

    # for fold, model, _, best_score_fold, best_iteration, _, _, val_X, val_y in train_lgbm_generator(X=np.array(X_train),y=np.array(y_train), n_folds=5, cv='kfold', seed=42, n_iter=1000, verbose_eval=-1, params=params, early_stop=True):
    for (
        fold,
        model,
        _,
        best_score_fold,
        best_iteration,
        _,
        _,
        val_X,
        val_y,
    ) in train_lgbm_generator(
        X=np.array(X_train),
        y=np.array(y_train),
        n_folds=5,
        cv=cv,
        seed=42,
        n_iter=10000,
        verbose_eval=-1,
        params=params,
        early_stop=True,
        df_train_folds=df_train_folds,
        df_val_folds=df_val_folds,
        custom_metric=custom_metric,
        transformer=transformer,
    ):
        # lst_mean_org_RMSE.append(best_score_fold['valid']['Original_Scale_RMSE'])
        # print(best_score_fold['valid']['Weighted_RMSE'])
        if metric == ["custom"] and transformer:
            lst_mean_org_RMSE.append(best_score_fold["valid"]["Original_Scale_RMSE"])
            return np.mean(lst_mean_org_RMSE)
        elif metric == ["custom", "l1"]:
            lst_mean_weighted_RMSE.append(best_score_fold["valid"]["Weighted_RMSE"])
            return np.mean(lst_mean_weighted_RMSE)
        elif reg_obj == "binary":
            print(f"binary log loss: {best_score_fold['valid']['binary_logloss']}")
            print(f"AUC: {best_score_fold['valid']['auc']}")
            lst_mean_binary_logloss.append(best_score_fold["valid"]["binary_logloss"])
            return np.mean(lst_mean_binary_logloss)
        else:
            print(best_score_fold["valid"]["rmse"])
            lst_mean_RMSE.append(best_score_fold["valid"]["rmse"])
            return np.mean(lst_mean_RMSE)


def random_search_optimizer(reg, X, y, search_params, scoring, cv, n_trials, seed=42):
    RandomSearch = RandomizedSearchCV(
        estimator=reg,
        param_distributions=search_params,
        n_iter=n_trials,
        cv=cv,
        scoring=scoring,
        refit=scoring[0],
        return_train_score=True,
        random_state=seed,
        n_jobs=-1,
        verbose=2,
    )
    RandomSearch.fit(X, y)
    results = pd.DataFrame(RandomSearch.cv_results_)
    best_params = RandomSearch.best_params_
    return results, best_params


def optuna_optimizer(objective, n_trials, seed=42):
    """_summary_
    Wrapper for optuna optimization

    samplers:
        TPESampler(seed=seed)
        RandomSampler(seed=seed)
    directions:
        ["minimize"] for single-objective optimization
        ["minimize", "minimize"] for multi-objective optimization

    Args:
        objective (_type_): _description_
        n_trials (_type_): _description_
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: _description_
    """
    # RandomSampler(seed=seed)
    # directions=["minimize", "minimize"]
    study = optuna.create_study(
        directions=["minimize"],
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_warmup_steps=10),
    )
    # study_name='study_name', storage='sqlite:///example.db'
    # optimize(func[, n_trials, timeout, n_jobs, ...])
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()

    best_params = study.best_trial.params
    # joblib.dump(study, "optuna_study.pkl")
    # study.trials_dataframe().to_pickle(f"optuna_trials_{date}.pkl")
    return study.trials_dataframe(), best_params


def find_best_regressor(
    reg_type,
    X,
    y,
    search_params,
    fixed_params,
    scoring,
    optimizer,
    cv="kfold",
    n_folds=5,
    n_trials=2,
    seed=42,
    metric="",
    transformer=None,
    reg_obj="regression",
):
    """_summary_

    Args:
        reg_type (_type_): choose lightgbm, only for randomized serach
        X (_type_): _description_
        y (_type_): _description_
        search_params (_type_): _description_
        fixed_params (_type_): _description_
        scoring (_type_): _description_
        optimizer (_type_): _description_
        cv (str, optional): _description_. Defaults to "kfold".
        n_folds (int, optional): _description_. Defaults to 5.
        n_trials (int, optional): _description_. Defaults to 2.
        seed (int, optional): _description_. Defaults to 42.
        metric (str, optional): _description_. Defaults to "".
        transformer (_type_, optional): _description_. Defaults to None.
        reg_obj (str, optional): _description_. Defaults to "regression".

    Raises:
        ValueError: raises an error if the optimizer is not supported

    Returns:
        _type_: _description_

    To do:
        - RandomizedSearchCV is currently experimental
        - implement xgboost and catboost under reg dict
            "xgb": xgb.XGBRegressor,
            "catboost": cb.CatBoostRegressor,

    """
    reg = {
        "lgbm": lgb.LGBMRegressor,
    }

    estimator = reg[reg_type](**fixed_params)
    dict_cross_val = {
        "kfold": KFold(n_splits=n_folds, shuffle=True, random_state=seed),
    }

    if optimizer == "random":
        results, best_params = random_search_optimizer(
            estimator,
            X,
            y,
            search_params,
            scoring,
            dict_cross_val[cv],
            n_trials=n_trials,
            seed=seed,
        )

    elif optimizer == "optuna":
        # results, best_params = optuna_optimizer(objective_optuna(metric=''), n_trials, seed=seed)
        results, best_params = optuna_optimizer(
            lambda trial: objective_optuna(
                trial,
                cv=cv,
                metric=metric,
                transformer=transformer,
                reg_obj=reg_obj,
                boost=boost,
            ),
            n_trials,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    all_params = {**fixed_params, **best_params}

    return results, all_params


if __name__ == "__main__":
    date_current = datetime.now().strftime("%Y%m%d")
    ml_opt_path = Path.cwd() / "temp"

    # Type in parameters needed for optimization and training
    search_type = "molecular"  # atomic, molecular, molecular_scaled
    cv = "molecular"  # molecular, kfold
    metric = [
        "l1",
        "rmse",
    ]  # ['custom', 'l1'] ['custom'] ['l1', 'rmse'] ["binary_logloss", "auc"]
    custom_metric = None  # None, original_scale_metrics, weighted_rmse
    transformer = None  # None, qt
    reg_obj = "regression"  # regression, binary
    boost = "gbdt"  # dart, gbdt

    # dataset = pd.read_pickle(
    #     "ML_bordwellch_ibond_shen_with_outliers_without_noch_final_20231021.pkl"
    # )

    # dataset = pd.read_pickle(
    #     "/Users/borup/Nextcloud/Education/phd_theoretical_chemistry/project_CH/pkalc_ml/datasets/ML_bordwellch_ibond_shen_with_outliers_without_noch_final_20231106.pkl"
    # )

    dataset = pd.read_pickle(
        "/Users/borup/Nextcloud/Education/phd_theoretical_chemistry/project_CH/pkalc_ml/datasets/full_dataset_20231221.pkl"
    )

    if ml_opt_path.exists():
        log_path = Path(f"{ml_opt_path}/ml_opt_{reg_obj}_{boost}_{date_current}.log")
    else:
        # make a temp directory to save files into
        ml_opt_path = Path("temp/")
        ml_opt_path.mkdir(parents=True, exist_ok=True)
        log_path = Path(f"{ml_opt_path}/ml_opt_{reg_obj}_{boost}_{date_current}.log")
        # raise FileNotFoundError("Could not find the directory {ml_val_path} or {root_path}")

    log_level = logging.DEBUG

    # Print to the terminal
    logging.root.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger("pythonConfig")
    if not log.hasHandlers():
        log.setLevel(log_level)
        log.addHandler(stream)

    # file handler:
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    # supress logging warnings
    # optuna.logging.set_verbosity(optuna.logging.WARNING)

    start_time = time.time()
    # ---------------------------------------------------
    # DATA PREPARATION
    # split the data into train and test sets. test_size = 0.2 means 20% of the data is used for testing
    # the validation set is created from 10 % of the training set
    train_folds, val_folds, train_indices, test_indices = create_folds_molecular(
        df=dataset, n_splits=5, test_size=0.2, random_state=42
    )

    # # print("Train folds:", train_folds)
    # print("Validation folds:", val_folds)
    # print("Test indices:", test_indices)

    train_df = create_train_test_df_molecular(
        indices=train_indices,
        df_source=dataset,
        col_name="names",
        feature_col="descriptor_vector",
        label_col="lst_pka_lfer",  # classification: atom_lowest_lfer, atom_lowest_lfer_05,	atom_lowest_lfer_1	atom_lowest_lfer_2, regression: lst_pka_lfer
        atom_index_col="atom_indices",
        regression=True,  # classification : False, regression : True
    )

    test_df = create_train_test_df_molecular(
        indices=test_indices,
        df_source=dataset,
        col_name="names",
        feature_col="descriptor_vector",
        label_col="lst_pka_lfer",  # classification: atom_lowest_lfer, regression: lst_pka_lfer
        atom_index_col="atom_indices",
        regression=True,  # classification : False, regression : True
    )

    # Create a list of dataframes of training folds
    df_train_folds = create_folds_df_molecular(
        folds=train_folds,
        df_source=dataset,
        col_name="names",
        feature_col="descriptor_vector",
        label_col="lst_pka_lfer",  # classification: atom_lowest_lfer, regression: lst_pka_lfer
        atom_index_col="atom_indices",
        regression=True,  # classification : False, regression : True
    )

    # Create a list of dataframes of validation folds
    df_val_folds = create_folds_df_molecular(
        folds=val_folds,
        df_source=dataset,
        col_name="names",
        feature_col="descriptor_vector",
        label_col="lst_pka_lfer",  # classification: atom_lowest_lfer, regression: lst_pka_lfer
        atom_index_col="atom_indices",
        regression=True,  # classification : False, regression : True
    )

    if transformer is not None:
        qt = QuantileTransformer(output_distribution="normal")
        train_df = scale_target_data(df=train_df, scaler=qt)
        test_df = scale_target_data(df=test_df, scaler=qt)
        df_train_folds = [
            scale_target_data(df=fold, scaler=qt) for fold in df_train_folds
        ]
        df_val_folds = [scale_target_data(df=fold, scaler=qt) for fold in df_val_folds]

    lst_dfs = [train_df, test_df, df_train_folds, df_val_folds]
    joblib.dump(
        lst_dfs,
        Path(ml_opt_path / f"lst_dfs_optuna_{boost}_{search_type}_{date_current}.pkl"),
    )

    # ---------------------------------------------------

    fixed_params = {
        "random_state": 42,
        "objective": reg_obj,
        "metric": metric,
    }

    print("using the following for search")
    print(f"metric: {metric}")
    print(f"optimizer: optuna")
    print(f"cv: {cv}")
    print(f"custom metric: {custom_metric}")
    print(f"transformer: {transformer}")
    print(f"reg obj: {reg_obj}")
    print(f"boost: {boost}")
    print("--" * 30)
    time.sleep(5)
    log.info("##" * 50)
    log.info("HYPERPARAMETER SEARCH")
    log.info("Optimizer: Optuna")
    log.info(f"Objective: {reg_obj}")
    log.info(f"Boosting type: {boost}")
    log.info(f"Data transformation: {transformer}")
    log.info(f"CV: {cv}")
    log.info(f"Metric: {metric} with custom metric: {custom_metric}")

    # ---------------------------------------------------
    # OPTUNA OPTIMIZATION
    # 1. find best parameters
    # 2. train model with best parameters
    # 3. save model and results
    # 4. plot results
    # ---------------------------------------------------

    X_train = train_df.iloc[:, :-4]  # Drop 'label', 'atom_index', 'idx_name'
    y_train = train_df["label"]
    X_test = test_df.iloc[:, :-4]  # Drop 'label', 'atom_index', 'idx_name'
    y_test = test_df["label"]
    # # # print(test_df.head())

    study_results, all_params_optuna = find_best_regressor(
        reg_type="lgbm",
        X=X_train,
        y=y_train,
        search_params="",
        fixed_params=fixed_params,
        scoring=[""],
        optimizer="optuna",
        cv=cv,
        n_folds=5,
        n_trials=100,
        seed=42,
        metric=metric,
        transformer=transformer,
        reg_obj=reg_obj,
    )

    joblib.dump(
        all_params_optuna,
        Path(
            ml_opt_path / f"all_params_optuna_{boost}_{search_type}_{date_current}.pkl"
        ),
    )
    joblib.dump(
        study_results,
        Path(
            ml_opt_path
            / f"study_results_optuna_{boost}_{search_type}_{date_current}.pkl"
        ),
    )

    # ---------------------------------------------------
    gen = train_lgbm_generator(
        X=np.array(X_train),
        y=np.array(y_train),
        params=all_params_optuna,
        n_folds=5,
        seed=42,
        cv="",
        verbose_eval=-1,
        n_iter=10000,
        early_stop=True,
        custom_metric=custom_metric,
        transformer=transformer,
    )
    fold, model, evals_result, best_score_fold, best_iteration, _, _, _, _ = next(gen)

    model.save_model(
        Path(
            ml_opt_path / f"final_model_optuna_{boost}_{search_type}_{date_current}.txt"
        ),
        num_iteration=model.best_iteration,
    )
    joblib.dump(
        model,
        Path(
            ml_opt_path / f"final_model_optuna_{boost}_{search_type}_{date_current}.pkl"
        ),
    )
    joblib.dump(
        evals_result,
        Path(
            ml_opt_path
            / f"evals_result_optuna_{boost}_{search_type}_{date_current}.pkl"
        ),
    )

    predictions_from_optuna = model.predict(np.array(X_test))
    if transformer is not None:
        predictions_from_optuna = transformer.inverse_transform(
            predictions_from_optuna.reshape(-1, 1)
        ).flatten()

    # TRAIN without early stopping rounds
    all_params_optuna_no_early = {
        k: v for k, v in all_params_optuna.items() if k != "early_stopping_rounds"
    }
    gen_no_early = train_lgbm_generator(
        X=np.array(X_train),
        y=np.array(y_train),
        params=all_params_optuna,
        n_folds=5,
        seed=42,
        cv="",
        verbose_eval=-1,
        n_iter=10000,
        early_stop=False,
        custom_metric=custom_metric,
        transformer=transformer,
    )

    (
        fold_no_early,
        model_no_early,
        evals_result_no_early,
        best_score_fold_no_early,
        best_iteration_no_early,
        _,
        _,
        _,
        _,
    ) = next(gen_no_early)

    model_no_early.save_model(
        Path(
            ml_opt_path
            / f"final_model_optuna_no_early_{boost}_{search_type}_{date_current}.txt",
            num_iteration=model_no_early.best_iteration,
        )
    )
    joblib.dump(
        model_no_early,
        Path(
            ml_opt_path
            / f"final_model_optuna_no_early_{boost}_{search_type}_{date_current}.pkl"
        ),
    )
    joblib.dump(
        evals_result_no_early,
        Path(
            ml_opt_path
            / f"evals_result_optuna_no_early_{boost}_{search_type}_{date_current}.pkl",
        ),
    )
    predictions_from_optuna_no_early = model_no_early.predict(np.array(X_test))
    if transformer is not None:
        predictions_from_optuna_no_early = transformer.inverse_transform(
            predictions_from_optuna_no_early.reshape(-1, 1)
        ).flatten()

    joblib.dump(
        {
            "pred_testset_with_early": predictions_from_optuna,
            "pred_testset_no_early": predictions_from_optuna_no_early,
        },
        Path(
            ml_opt_path
            / f"predictions_testset_optuna_{boost}_{search_type}_{date_current}.pkl"
        ),
    )

    # ---------------------------------------------------

    if reg_obj == "regression":  # regression, binary
        log.info("--" * 50)
        log.info("Results Regression search")
        log.info(f"hyperparameters: {all_params_optuna}")
        log.info("With early stopping")
        log.info(compute_scores_regression(y_test, predictions_from_optuna))
        log.info("Without early stopping")
        log.info(compute_scores_regression(y_test, predictions_from_optuna_no_early))
        log.info("--" * 50)

        print("--" * 30)
        print("Results Regression search")
        print("With early stopping")
        print(compute_scores_regression(y_test, predictions_from_optuna))
        print("Without early stopping")
        print(compute_scores_regression(y_test, predictions_from_optuna_no_early))
        print("--" * 30)

        plot_multiple_subplots_calc_pred(
            n=1,
            y_preds_list=[np.array(y_test)],
            y_test=np.array(predictions_from_optuna),
            titles_list=[""],
            save_fig=True,
            fig_name=Path(
                f"{ml_opt_path}/plot_{boost}_{search_type}_{date_current}.svg"
            ),
            lst_textstr=None,
            outliers=False,
        )

        plot_multiple_subplots_calc_pred(
            n=1,
            y_preds_list=[np.array(y_test)],
            y_test=np.array(predictions_from_optuna_no_early),
            titles_list=[""],
            save_fig=True,
            fig_name=Path(
                f"{ml_opt_path}/plot_{boost}_{search_type}_no_early_{date_current}.svg"
            ),
            lst_textstr=None,
            outliers=False,
        )

    elif reg_obj == "binary":
        print("--" * 30)
        print("Results Binary classification search")
        print("--" * 30)
        predictions_from_optuna = (predictions_from_optuna > 0.5).astype("int")
        predictions_from_optuna_no_early = (
            predictions_from_optuna_no_early > 0.5
        ).astype("int")

        cm = confusion_matrix(y_test, predictions_from_optuna)
        cm_no_early = confusion_matrix(y_test, predictions_from_optuna_no_early)

        log.info("--" * 50)
        log.info("Results Binary classification search")
        log.info("With early stopping")
        log.info(f"binary logloss: {best_score_fold['valid']['binary_logloss']}")
        log.info(f"Accuracy: {accuracy_score(y_test, predictions_from_optuna)}")
        log.info(f"MCC: {matthews_corrcoef(y_test, predictions_from_optuna)}")
        log.info(f"F1 score: {f1_score(y_test, predictions_from_optuna)}")
        log.info(f"Confusion matrix:\n{cm}")
        log.info("--" * 50)
        log.info("Without early stopping")
        log.info(
            f"Accuracy: {accuracy_score(y_test, predictions_from_optuna_no_early)}"
        )
        log.info(f"MCC: {matthews_corrcoef(y_test, predictions_from_optuna_no_early)}")
        log.info(f"f1 score: {f1_score(y_test, predictions_from_optuna_no_early)}")
        log.info(f"Confusion matrix:\n{cm_no_early}")

        print("With early stopping")
        print(f"binary logloss: {best_score_fold['valid']['binary_logloss']}")
        # cm = confusion_matrix(y_test, predictions_from_optuna)
        print(train_df["label"].value_counts(), test_df["label"].value_counts())
        print(f"Accuracy: {accuracy_score(y_test, predictions_from_optuna)}")
        print(f"MCC: {matthews_corrcoef(y_test, predictions_from_optuna)}")
        print(f"F1 score: {f1_score(y_test, predictions_from_optuna)}")
        print(f"Confusion matrix:\n{cm}")
        print("--" * 50)
        print("Without early stopping")
        print("--" * 50)

        print(f" binary logloss: {best_score_fold_no_early}")

        print(
            f"Train value counts:{train_df['label'].value_counts()},\nTest value counts:{test_df['label'].value_counts()}"
        )
        print(f"Accuracy: {accuracy_score(y_test, predictions_from_optuna_no_early)}")
        print(f"MCC: {matthews_corrcoef(y_test, predictions_from_optuna_no_early)}")
        print(f"Confusion matrix:\n{cm_no_early}")
        print(f"F1 score: {f1_score(y_test, predictions_from_optuna_no_early)}")

        plot_confusion_matrix(
            cm=cm,
            fig_name=Path(
                ml_opt_path / f"cm_optuna_{boost}_{search_type}_{date_current}.svg"
            ),
            save_fig=True,
        )

        plot_confusion_matrix(
            cm=cm_no_early,
            fig_name=Path(
                ml_opt_path
                / f"cm_optuna_no_early_{boost}_{search_type}_{date_current}.svg"
            ),
            save_fig=True,
        )

    # # lgb.plot_metric(evals_result, metric='auc')
    # lgb.plot_metric(evals_result_no_early, metric="binary_logloss")
    # plt.show()

    end_time = time.time()

    log.info(f"Time taken in seconds: {end_time - start_time}")
    log.info(f"Time taken in minutes: {round((end_time - start_time)/60, 2)}")
    log.info("##" * 50)

    # ---------------------------------------------------

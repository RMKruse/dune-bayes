# import basic stuff
import sklearn
import pandas as pd
import numpy as np
import itertools

# Reproducibility
import random
random.seed(0)
import torch
torch.manual_seed(0)

# import tensorflow and keras stuff
import tensorflow_probability as tfp
import tensorflow as tf
tf.random.set_seed(0)

tfd = tfp.distributions
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.losses import *
from keras.regularizers import *
import keras.backend as K

# import kflod stuff
from sklearn.model_selection import KFold

# import preprocessing functions
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# import comparison models
from xgboost import XGBClassifier, XGBRegressor
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
# from nodegam.sklearn import NodeGAMRegressor, NodeGAMClassifier 

# plotting
import matplotlib.pyplot as plt


########################################### Preprocessing func

# GPU Update, da Tensorflow SCHEISSE IST UND 110% der Grafikkarte nutzt.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # beschränken auf 14GB.
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=14*1024)]    # 10 * 1024MB setzen (obere Grenze ist 16GB)
        )
        logical_gpus = tf.config.list_physical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

########################################### Preprocessing func
class CustomPipeline(Pipeline):
    """Custom sklearn Pipeline to transform data."""

    def apply_transformation(self, x):
        """Applies all transforms to the data, without applying last estimator.

        Args:
          x: Iterable data to predict on. Must fulfill input requirements of first
            step of the pipeline.

        Returns:
          xt: Transformed data.
        """
        xt = x
        for _, transform in self.steps[:-1]:
            xt = transform.fit_transform(xt)
        return xt


def transform_data(df):
    """Apply a fixed set of transformations to the pd.Dataframe `df`.

    Args:
      df: Input dataframe containing features.

    Returns:
      Transformed dataframe and corresponding column names. The transformations
      include (1) encoding categorical features as a one-hot numeric array, (2)
      identity `FunctionTransformer` for numerical variables. This is followed by
      scaling all features to the range (-1, 1) using min-max scaling.
    """
    column_names = df.columns
    new_column_names = []
    is_categorical = np.array([dt.kind == "O" for dt in df.dtypes])
    categorical_cols = df.columns.values[is_categorical]
    numerical_cols = df.columns.values[~is_categorical]
    for index, is_cat in enumerate(is_categorical):
        col_name = column_names[index]
        if is_cat:
            new_column_names += [
                "{}: {}".format(col_name, val) for val in set(df[col_name])
            ]
        else:
            new_column_names.append(col_name)
    cat_ohe_step = ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"))

    cat_pipe = Pipeline([cat_ohe_step])
    num_pipe = Pipeline([("identity", FunctionTransformer(validate=True))])
    transformers = [
        ("cat", cat_pipe, categorical_cols),
        ("num", num_pipe, numerical_cols),
    ]
    column_transform = ColumnTransformer(transformers=transformers)

    pipe = CustomPipeline(
        [
            ("column_transform", column_transform),
            ("min_max", MinMaxScaler((-1, 1))),
            ("dummy", None),
        ]
    )
    df = pipe.apply_transformation(df)
    return df, new_column_names


####################################### NAM EXU-activation Layer
class ExuLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(ExuLayer, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.clip_by_value(tf.matmul(inputs, tf.exp(self.w)) + self.b, 0, 1)


######################################################### Model builder


############################### Helper functions for building MLP, NAM and NAMLSS
def built_DNN(input, output_activation="linear", output_num=1):
    x = Dense(1000, "relu")(input)
    x = Dropout(0.5)(x)
    x = Dense(500, "relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(50, "relu")(x)
    x = Dense(25, "relu")(x)
    x = Dense(output_num, activation=output_activation, use_bias=False)(x)
    model_dnn = Model(inputs=input, outputs=x)
    model_dnn.reset_states()
    return model_dnn


def LINEAR(x):
    return x


################# MLP
def MLP(
    features_train,
    labels_train,
    features_test,
    labels_test,
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mse"],
    output_activation="linear",
):
    inps = Input(shape=(features_train.shape[1],))
    model = built_DNN(inps, output_activation=output_activation)

    model.compile(
        loss=POINT_LOSS, metrics=metrics, optimizer=Adam(learning_rate=LEARNING_RATE)
    )

    history = model.fit(
        x=features_train,
        y=labels_train,
        epochs=NUM_EPOCHS,
        callbacks=[EARLY_STOPPING, REDUCE_LR],
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    loc_pred = model.predict(features_test)
    loc_pred = np.array(
        [loc_pred[i][0] for i in range(len(loc_pred))], dtype=np.float64
    )
    likelihood = LL_EVAL(loc_pred, labels_test)

    return likelihood


######################## Distributional DNN


def DDNN(
    features_train,
    labels_train,
    features_test,
    labels_test,
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mse"],
    distribution=tfd.Normal,
    loc_activation=LINEAR,
    scale_activation=tf.math.softplus,
    output_num=2,
):
    # Create inputs
    inps = Input(shape=(features_train.shape[1],))
    ms = built_DNN(inps, "linear", output_num)
    z = ms.output

    # built distributional layer
    # Change for when dist params have different names
    p_y = tfp.layers.DistributionLambda(
        lambda x: distribution(
            loc=loc_activation(x[:, 0]), scale=scale_activation(x[:, 1])
        )
    )(z)

    model = Model(inputs=ms.input, outputs=p_y)

    def NLL(y_true, y_hat):
        return -y_hat.log_prob(y_true)

    model.compile(
        loss=NLL, metrics=metrics, optimizer=Adam(learning_rate=LEARNING_RATE)
    )

    history = model.fit(
        x=features_train,
        y=labels_train,
        epochs=NUM_EPOCHS,
        callbacks=[EARLY_STOPPING, REDUCE_LR],
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    # Evaluate model
    preds = ms(features_test)
    mu_preds = np.array(loc_activation(preds[:, 0]))
    sigma_preds = np.array([scale_activation(preds[:, 1])])

    ll = LL_EVAL(mu_preds, labels_test, sigma_preds)

    return ll


######################################### NAM


def NAM(
    features_train,
    labels_train,
    features_test,
    labels_test,
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mse"],
    output_activation="linear",
):
    inps = [Input(shape=(1,)) for _ in range(features_train.shape[1])]

    # define submodels
    # same architecture as for DNN and MLP
    ms = [
        built_DNN(inps[i], output_activation=output_activation)
        for i in range(features_train.shape[1])
    ]
    z = sum([m.output for m in ms])
    model = Model(inputs=[m.input for m in ms], outputs=z)

    model.compile(
        loss=POINT_LOSS, metrics=metrics, optimizer=Adam(learning_rate=LEARNING_RATE)
    )

    training_data = [features_train[:, i] for i in range(features_train.shape[1])]
    eval_data = [features_test[:, i] for i in range(features_test.shape[1])]

    history = model.fit(
        x=training_data,
        y=labels_train,
        epochs=NUM_EPOCHS,
        callbacks=[EARLY_STOPPING, REDUCE_LR],
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    loc_pred = model.predict(eval_data)
    loc_pred = np.array(
        [loc_pred[i][0] for i in range(len(loc_pred))], dtype=np.float64
    )
    likelihood = LL_EVAL(loc_pred, labels_test)

    return likelihood


############################################ NAMLSS


def define_models_scale(input):
    x = Dense(50, activation="relu")(input)
    x = Dense(25, activation="relu")(x)
    x = Dense(1, activation="linear", use_bias=False)(x)
    x = Model(inputs=input, outputs=x)
    # x.reset_states()
    return x


def NAMLSS(
    features_train,
    labels_train,
    features_test,
    labels_test,
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mse"],
    distribution=tfd.Normal,
    loc_activation=LINEAR,
    scale_activation=tf.math.softplus,
    output_num=1,
):
    training_data = 2 * [features_train[:, i] for i in range(features_train.shape[1])]
    eval_data = 2 * [features_test[:, i] for i in range(features_test.shape[1])]

    inps = [Input(shape=(1,)) for _ in range(2 * features_train.shape[1])]

    ms = [built_DNN(inps[i]) for i in range(features_train.shape[1])]
    ms += [
        define_models_scale(inps[i + features_train.shape[1]])
        for i in range(features_train.shape[1])
    ]

    z1 = sum([m.output for m in ms[: features_train.shape[1]]])
    z2 = sum([m.output for m in ms[features_train.shape[1] :]])

    z = concatenate([z1, z2])

    # Change for when dist params have different names
    p_y = tfp.layers.DistributionLambda(
        lambda x: distribution(
            loc=loc_activation(x[:, 0]), scale=scale_activation(x[:, 1])
        )
    )(z)

    model = Model(inputs=[m.input for m in ms], outputs=p_y)

    def NLL(y_true, y_hat):
        return -y_hat.log_prob(y_true)

    model.compile(
        loss=NLL, metrics=metrics, optimizer=Adam(learning_rate=LEARNING_RATE)
    )

    history = model.fit(
        x=training_data,
        y=labels_train,
        epochs=NUM_EPOCHS,
        callbacks=[EARLY_STOPPING, REDUCE_LR],
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    mu_preds = [
        sum(
            loc_activation(ms[idx].predict(eval_data[idx], verbose=0))
            for idx in range(features_test.shape[1])
        )
    ]
    sigma_preds = [
        sum(
            scale_activation(
                ms[idx + features_test.shape[1]].predict(eval_data[idx], verbose=0)
            )
            for idx in range(features_test.shape[1])
        )
    ]

    preds = [
        ms[idx].predict(eval_data[idx], verbose=0) for idx in range(len(eval_data))
    ]
    preds_mu = preds[: features_test.shape[1]]
    preds_sigma = preds[features_test.shape[1] :]

    mu = sum(preds_mu)
    sigma = sum(preds_sigma)
    mu_preds = loc_activation(mu)
    sigma_preds = scale_activation(sigma)

    mu = np.array([mu_preds[i][0] for i in range(len(mu_preds))])
    sigma = np.array([sigma_preds[i][0] for i in range(len(sigma_preds))])

    ll = LL_EVAL(mu, labels_test, sigma)

    return ll


#### NAMLSS 2
def NA2MLSS(
    features_train,
    labels_train,
    features_test,
    labels_test,
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mse"],
    distribution=tfd.Normal,
    loc_activation=LINEAR,
    scale_activation=tf.math.softplus,
    output_num=2,
):
    training_data = [features_train[:, i] for i in range(features_train.shape[1])]
    eval_data = [features_test[:, i] for i in range(features_test.shape[1])]

    inps = [Input(shape=(1,)) for _ in range(features_train.shape[1])]

    ms = [
        built_DNN(inps[i], output_num=output_num)
        for i in range(features_train.shape[1])
    ]

    z = sum([m.output for m in ms])

    # Change for when dist params have different names
    p_y = tfp.layers.DistributionLambda(
        lambda x: distribution(
            loc=loc_activation(x[:, 0]), scale=scale_activation(x[:, 1])
        )
    )(z)

    model = Model(inputs=[m.input for m in ms], outputs=p_y)

    def NLL(y_true, y_hat):
        return -y_hat.log_prob(y_true)

    model.compile(
        loss=NLL, metrics=metrics, optimizer=Adam(learning_rate=LEARNING_RATE)
    )

    history = model.fit(
        x=training_data,
        y=labels_train,
        epochs=NUM_EPOCHS,
        callbacks=[EARLY_STOPPING, REDUCE_LR],
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    preds = [
        ms[idx].predict(eval_data[idx], verbose=0) for idx in range(len(eval_data))
    ]

    preds = sum(preds)
    mu, sigma = preds[:, 0], preds[:, 1]
    mu_preds = loc_activation(mu)
    sigma_preds = scale_activation(sigma)

    ll = LL_EVAL(mu_preds, labels_test, sigma_preds)

    return ll


###################### XGBoost
def XGB(features_train, labels_train, features_test, labels_test, regression=True):
    if regression:
        model = XGBRegressor()
    else:
        model = XGBClassifier()
    model.fit(features_train, labels_train)

    preds = model.predict(features_test)

    ll = LL_EVAL(np.float64(preds), labels_test)

    return ll


############## EBM
def EBM(features_train, labels_train, features_test, labels_test, regression=True):
    if regression:
        model = ExplainableBoostingRegressor()
    else:
        model = ExplainableBoostingClassifier()

    model.fit(features_train, labels_train)

    preds = model.predict(features_test)

    ll = LL_EVAL(np.float64(preds), labels_test)

    return ll


############################## NODEGAM
# def NODEGAM(features_train, labels_train, features_test, labels_test, regression=True):
#     if regression:
#         model = NodeGAMRegressor(
#             in_features=features_train.shape[1], verbose=0, seed=141, ga2m=0
#         )
#     else:
#         model = NodeGAMClassifier(
#             in_features=features_train.shape[1],
#             objective="ce_loss",
#             verbose=0,
#             seed=141,
#             ga2m=0,
#         )

#     X_train = pd.DataFrame(np.vstack([features_train])).reset_index(drop=True)
#     X_test = pd.DataFrame(np.vstack([features_test])).reset_index(drop=True)
#     record = model.fit(X_train, np.array(labels_train))
#     preds = model.predict(X_test)

#     ll = LL_EVAL(np.float64(preds), labels_test)

#     return ll


if __name__ == "__main__":
    # define task: if not REGRESSION, for EBM, XGBoost, Nodegam they will use cross_entropy as loss:
    REGRESSION = True
    # Loss func for MLP, NAM ->
    POINT_LOSS = "mse"

    # general arguments
    BATCH_SIZE = 512
    NUM_EPOCHS = 2000
    LEARNING_RATE = 0.001
    NUM_FOLDS = 5
    EARLY_STOPPING = EarlyStopping(
        patience=150, restore_best_weights=True, min_delta=1e-05, monitor="loss"
    )
    REDUCE_LR = ReduceLROnPlateau(
        monitor="loss", factor=0.95, patience=10, min_delta=1e-05
    )

    # Metrics for Neural Networks... Not important
    METRICS = [tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mse"]

    # Number of folds, still same random_state
    KFOLD = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=101)

    ####!!!!! DISTRIBUTION: For e.g. Logistic use tfd.Logistic
    DISTRIBUTION = tfd.Normal

    # If logistic use:
    # def log_likelihood(mu, data):
    #   sigma=np.std(data)
    #   mu = np.float64([mu[i][0] for i in range(len(mu))])
    #   dist = tfd.Logistic(mu, sigma)
    #   return tf.reduce_sum(dist.log_prob(value=tf.cast(data, dtype=tf.float64)))

    # Define distribution that is modelled
    def LL_EVAL(loc, y_true, scale=None):
        if scale is None:
            dist = DISTRIBUTION(loc, scale=tfp.stats.stddev(y_true))
        else:
            dist = DISTRIBUTION(loc, scale=scale)
        return -tf.reduce_sum(dist.log_prob(value=y_true)).numpy()

    # if Logistic use:
    # def log_likelihood(mu, sigma, data):
    #    dist = tfd.Logistic(mu, sigma)
    #    return tf.reduce_sum(dist.log_prob(value=tf.cast(data, dtype=tf.float32)))

    LOC_ACTIVATION = LINEAR
    SCALE_ACTIVATION = tf.math.softplus

    # Custom for Dataset
    from sklearn import datasets
    from scipy import stats

    housing = datasets.fetch_california_housing()

    X = pd.DataFrame(data=housing.data, columns=housing.feature_names) # [0:1500]
    targets = housing.target # [0:1500]
    df = pd.DataFrame(X, columns=housing.feature_names)
    df["targets"] = targets
    df = df[(np.abs(stats.zscore(df)) < 10).all(axis=1)]
    df = df.reset_index(drop=True)

    X = df[housing.feature_names]
    targets = np.array(df["targets"])

    # preprocess data
    features, cols = transform_data(X)

    # Normalize Data, only! for Normal dist
    scaler = StandardScaler().fit(np.array(targets).reshape(-1, 1))
    targets = scaler.transform(np.array(targets).reshape(-1, 1)).flatten()

    model_list = [
        "MLP",
        "NAM",
        "XGBOOST",
        "EBM",
        # "NODEGAM",
        "DDNN",
        "NAMLSS",
        "NA2MLSS",
    ]

    results = pd.DataFrame(columns=["Model", "Likelihood"])

    for mod in model_list:
        print(mod)
        ll_per_fold = []

        for train, test in KFOLD.split(features, targets):
            if mod == "DDNN":
                ll = DDNN(
                    features[train],
                    targets[train],
                    features[test],
                    targets[test],
                    metrics=METRICS,
                    distribution=DISTRIBUTION,
                    loc_activation=LOC_ACTIVATION,
                    scale_activation=SCALE_ACTIVATION,
                    output_num=2,
                )
            elif mod == "MLP":
                ll = MLP(
                    features[train],
                    targets[train],
                    features[test],
                    targets[test],
                    metrics=METRICS,
                    output_activation="linear",
                )

            elif mod == "NAM":
                ll = NAM(
                    features[train],
                    targets[train],
                    features[test],
                    targets[test],
                    metrics=METRICS,
                    output_activation="linear",
                )
            elif mod == "XGBOOST":
                ll = XGB(
                    features[train],
                    targets[train],
                    features[test],
                    targets[test],
                    REGRESSION,
                )
            elif mod == "EBM":
                ll = EBM(
                    features[train],
                    targets[train],
                    features[test],
                    targets[test],
                    REGRESSION,
                )
            elif mod == "NODEGAM":
                ll = NODEGAM(
                    features[train],
                    targets[train],
                    features[test],
                    targets[test],
                    REGRESSION,
                )
            elif mod == "NAMLSS":
                ll = NAMLSS(
                    features[train],
                    targets[train],
                    features[test],
                    targets[test],
                    metrics=METRICS,
                    distribution=DISTRIBUTION,
                    loc_activation=LOC_ACTIVATION,
                    scale_activation=SCALE_ACTIVATION,
                    output_num=1,
                )
            elif mod == "NA2MLSS":
                ll = NA2MLSS(
                    features[train],
                    targets[train],
                    features[test],
                    targets[test],
                    metrics=METRICS,
                    distribution=DISTRIBUTION,
                    loc_activation=LOC_ACTIVATION,
                    scale_activation=SCALE_ACTIVATION,
                    output_num=2,
                )

            ll_per_fold.append(ll)

        name = NUM_FOLDS * [mod]
        temp_df = pd.DataFrame(
            np.array((name, ll_per_fold)).T, columns=["Model", "Likelihood"]
        )
        results = pd.concat([results, temp_df])

    results = results.astype({"Likelihood": float})
    print(results.groupby("Model").mean())
    print(results.groupby("Model").std())

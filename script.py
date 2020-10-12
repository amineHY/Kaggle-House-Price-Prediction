# %%
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

################################################################################
# Definition of methods
################################################################################


def evaluation_metrics_regression(y_true, y_pred):
    '''
    - Mean absolute error(MAE)
    - Mean squared error(MSE)
    - Root mean squared error(RMSE)
    - Root mean squared logarithmic error(RMSLE)
    - Mean percentage error(MPE)
    - Mean absolute percentage error(MAPE)
    - R2
    '''

    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    return mae, mse, r2


################################################################################
# main code starts here
################################################################################

print('[INFO] Import the dataset')
X_full = pd.read_csv('input/train.csv')
X_test = pd.read_csv('input/test.csv')


print('[INFO] Drop entries with missing target')
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
X = X_full.drop(['SalePrice'], axis=1)
y = X_full.SalePrice
print('Dimension of the dataset is {} and target {}:'.format(X.shape, y.shape))


# %%
print('[INFO] Split the dataset: train - validation sets')

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=.2, random_state=0)

print("[INFO] Discard columns with more than 20% empty entries")
train_nan_cols = [col for col in X_train.columns if X_train[col].isnull().sum()*100 /
                  X_train.shape[0] > 20]
valid_nan_cols = [col for col in X_valid.columns if X_valid[col].isnull().sum()*100 /
                  X_valid.shape[0] > 20]

X_train.drop(train_nan_cols, axis=1, inplace=True)
X_valid.drop(valid_nan_cols, axis=1, inplace=True)

categorical_cols = [
    col for col in X_train.columns if X_train[col].dtypes == 'object']
numerical_cols = [
    col for col in X_train.columns if X_train[col].dtypes in ['int64', 'float64']]
print('The dataset has {} categorical columns and {} numerical columns.'.format(
    len(categorical_cols), len(numerical_cols)))

# %%

print('[INFO] Handling missing (numerical and categorical) entries')
imputer = SimpleImputer(strategy='most_frequent')

# drop columns with >20% missing entries

print("[INFO] Handling missing numerical data")

X_train_impute_num = pd.DataFrame(
    imputer.fit_transform(X_train[numerical_cols]))
X_train_impute_num.columns = X_train[numerical_cols].columns
X_train_impute_num.index = X_train[numerical_cols].index

X_valid_impute_num = pd.DataFrame(
    imputer.transform(X_valid[numerical_cols]))
X_valid_impute_num.columns = X_valid[numerical_cols].columns
X_valid_impute_num.index = X_valid[numerical_cols].index

print("[INFO] Handling missing categorical data")
X_train_impute_cat = pd.DataFrame(
    imputer.fit_transform(X_train[categorical_cols]))
X_train_impute_cat.columns = X_train[categorical_cols].columns
X_train_impute_cat.index = X_train[categorical_cols].index

X_valid_impute_cat = pd.DataFrame(
    imputer.transform(X_valid[categorical_cols]))
X_valid_impute_cat.columns = X_valid[categorical_cols].columns
X_valid_impute_cat.index = X_valid[categorical_cols].index

X_train_impute = pd.concat([X_train_impute_num, X_train_impute_cat], axis=1)
X_valid_impute = pd.concat([X_valid_impute_num, X_valid_impute_cat], axis=1)


# %%
print("[INFO] Encoding categorical data")
print("\t Compute cardinality of the categorical data")
low_cardinality_cols = [col for col in X_train[categorical_cols]
                        if X_train[col].nunique() < 10]
high_cardinality_cols = [col for col in X_train[categorical_cols]
                         if X_train[col].nunique() >= 10]

# print('\t Categorical columns that will be one-hot encoded:\n',
#       low_cardinality_cols)
# print('\t Categorical columns that will be dropped from the dataset:\n',
#       high_cardinality_cols)

dropped_X_train = X_train_impute.drop(high_cardinality_cols, axis=1)
dropped_X_valid = X_valid_impute.drop(high_cardinality_cols, axis=1)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(
    dropped_X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(
    dropped_X_valid[low_cardinality_cols]))

OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Add one-hot encoded columns to numerical features
X_train_final = pd.concat(
    [X_train_impute[numerical_cols], OH_cols_train], axis=1)
X_valid_final = pd.concat(
    [X_valid_impute[numerical_cols], OH_cols_valid], axis=1)


# %%
print('[INFO] Modeling part: Machine learning Model')
mae_list = []
preds_list = []

# model_rf = RandomForestRegressor(n_estimators=n_est, random_state=0)

N_LIST = [100*x for x in range(1, 15)]
parameters = {'n_estimators': N_LIST}


# Manual parameter tunning
mae_list = []
mse_list = []
r2_list = []
n_est_list = []

for idx in range(1, 15):
    # defining the model
    n_est = 100*idx
    model_rf = RandomForestRegressor(n_estimators=n_est, random_state=0)

    # training the model
    model_rf.fit(X_train_final, y_train)

    # model validation
    preds = model_rf.predict(X_valid_final)

    # model evaluation
    mae, mse, r2 = evaluation_metrics_regression(y_valid, preds)
    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)
    n_est_list.append(n_est)

    print('N_est={3}, \tMAE={0:2.2f}, MSE={1:2.2f}, R2={2:2.2f}'.format(
        mae, mse, r2, n_est))


def min_list(mae_list):
    for idx, x in enumerate(mae_list):
        if x == min(mae_list):
            return x, idx
            break


v, idx = min_list(mae_list)

print('N_est={3:2.2f} \tMAE={0:2.2f}, MSE={1:2.2f}, R2={2:2.2f}'.format(
    min(mae_list), min(mse_list), min(r2_list), n_est_list(idx)))

# %%
print("[INFO] Grid search for parameter tuning with cross validation")

# Model definition
model_rf_gs = GridSearchCV(
    estimator=RandomForestRegressor(random_state=0),
    param_grid=parameters,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    verbose=2
)


model_rf_gs_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(),
    param_grid=parameters,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    verbose=2
)

# model training
model_rf_gs.fit(X_train_final, y_train)
model_rf_gs_random.fit(X_train_final, y_train)

# model validation
preds = model_rf_gs.predict(X_valid_final)
preds_random = model_rf_gs_random.predict(X_valid_final)

# model evaluation
mae, mse, r2 = evaluation_metrics_regression(y_valid, preds)
print('\tMAE={0:2.2f}, MSE={1:2.2f}, R2={2:2.2f}'.format(
    mae, mse, r2))
mae, mse, r2 = evaluation_metrics_regression(y_valid, preds_random)
print('\tMAE={0:2.2f}, MSE={1:2.2f}, R2={2:2.2f}'.format(
    mae, mse, r2))

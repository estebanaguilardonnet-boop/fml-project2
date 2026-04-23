#this code has the non-NN models train on a pooled sample -> pooled sample was created retriving all the country data csv files in my folder and appending each country (keeping dates alligned)
  #also each country dummy coded (except US as reference category)
# model info stored as .joblibs, code also makes predictions and they stored as well
#again, comment out the models dont want running now. doing so will append results so you wont lose a previously run script with different models
# with preditions it also calculates the metrics (stored in summary)


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from joblib import dump
import os
from os.path import join
import time
import argparse
import subprocess
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightgbm', '-q'])
    import lightgbm as lgb


your_path    = '/Users/valentin/Desktop/FML /group'
DATA_DIR     = join(your_path, 'raw_data', 'cleenerst')
PARAMS_DIR   = join(your_path, 'results', 'model_parameters', 'Pooled')
FORECAST_DIR = join(your_path, 'results', 'forecasts_pooled')
SUMMARY_DIR  = join(your_path, 'results', 'summary')

META_COLS = ['id', 'DATE', 'TARGET', 'me']
OLS3_COLS = ['mvel1', 'bm', 'mom_12']

LASSO_ALPHAS  = np.logspace(-3, 3, 10)
RIDGE_ALPHAS  = np.logspace(-3, 3, 10)
RF_MAX_DEPTHS = [2, 4, 6]
RF_MAX_FEATS  = [3, 5, 10]
RF_N_EST      = 300
GBRT_DEPTHS   = [1]
GBRT_LRS      = [0.1]
GBRT_N_EST    = 200

LGBM_DEPTHS     = [1, 2]
LGBM_LRS        = [0.01, 0.1]
LGBM_N_EST      = 1000
LGBM_EARLY_STOP = 200
LGBM_MIN_CHILD  = 50

MODELS_TO_RUN = [#'ols-3',
                 #'linear',
                 #'lasso',
                 #'ridge',
                 #'rf',
                 'gbrt+h',
                 #'lgbm',
                 ]

MODEL_ORDER = ['ols-3', 'linear', 'lasso', 'ridge', 'rf', 'gbrt+h', 'lgbm']

TRAIN_END = 1979
VALID_END = 1989
END_YEAR  = 2016

ALL_MARKETS = [
    'USA', 'Japan', 'China', 'India', 'Korea', 'Hong_Kong', 'Taiwan',
    'France', 'United_Kingdom', 'Thailand', 'Australia', 'Singapore',
    'Sweden', 'South_Africa', 'Poland', 'Israel', 'Vietnam', 'Italy',
    'Turkey', 'Switzerland', 'Indonesia', 'Greece', 'Philippines',
    'Norway', 'Sri_Lanka', 'Denmark', 'Finland', 'Saudi_Arabia',
    'Jordan', 'Egypt', 'Spain', 'Kuwait'
]

REFERENCE_MARKET = 'USA'

#  Data loading {+ pooling 

def find_csv(market):
    candidates = [f'{market}_clean.csv', f'{market}.csv']
    if market == 'Sri_Lanka':
        candidates.insert(0, 'Sri_lanka.csv')
    for name in candidates:
        path = join(DATA_DIR, name)
        if os.path.exists(path):
            return path
    return None

def load_and_pool_all_markets():
    dfs = []
    for market in ALL_MARKETS:
        filepath = find_csv(market)
        if filepath is None:
            print(f"  {market}: CSV not found, skipping")
            continue

        df = pd.read_csv(filepath)
        df = df.replace([np.inf, -np.inf], np.nan)
        df.dropna(inplace=True, how='any')
        df['id']     = market + '+' + df['id'].astype(str)
        df['market'] = market
        dfs.append(df)

    pooled = pd.concat(dfs, ignore_index=True)
    print(f"Pooled: {len(pooled):,} rows across {len(dfs)} markets")
    return pooled

def add_country_dummies(df):
    dummy_markets = sorted([m for m in df['market'].unique() if m != REFERENCE_MARKET])
    for m in dummy_markets:
        df[f'D_{m}'] = (df['market'] == m).astype(int)
    return df, dummy_markets

def demean_target_by_month(df):
    df['DATE_tmp']        = pd.to_datetime(df['DATE'])
    monthly_mean          = df.groupby(pd.Grouper(key='DATE_tmp', freq='ME'))['TARGET'].transform('mean')
    df['TARGET_original'] = df['TARGET']
    df['TARGET']          = df['TARGET'] - monthly_mean
    df.drop(columns='DATE_tmp', inplace=True)
    return df

def get_features(df, dummy_cols):
    exclude = META_COLS + ['market', 'TARGET_original']
    return df[[c for c in df.columns if c not in exclude]]

def get_features_ols3(df, dummy_cols):
    return df[OLS3_COLS + [f'D_{m}' for m in dummy_cols]]

def split_data(df, train_end, valid_end, add_year):
    train = df[df['DATE'] <= str(train_end + add_year)].copy()
    train.reset_index(drop=True, inplace=True)

    valid = df[(df['DATE'] > str(train_end + add_year)) &
               (df['DATE'] <= str(valid_end + add_year))].copy()
    valid.reset_index(drop=True, inplace=True)

    test = df[(df['DATE'] > str(valid_end + add_year)) &
              (df['DATE'] <= str(valid_end + add_year + 1))].copy()
    test.reset_index(drop=True, inplace=True)

    return train, valid, test

#  Model training (one year) 

def train_model_year(add_year, model_name, df, dummy_cols):
    cur_year = VALID_END + add_year

    train_data, valid_data, test_data = split_data(df, TRAIN_END, VALID_END, add_year)

    if len(train_data) == 0 or len(valid_data) == 0 or len(test_data) == 0:
        return None

    if model_name == 'ols-3':
        train_x = get_features_ols3(train_data, dummy_cols)
        valid_x = get_features_ols3(valid_data, dummy_cols)
        test_x  = get_features_ols3(test_data, dummy_cols)
    else:
        train_x = get_features(train_data, dummy_cols)
        valid_x = get_features(valid_data, dummy_cols)
        test_x  = get_features(test_data, dummy_cols)

    train_y = train_data['TARGET']
    valid_y = valid_data['TARGET']

    if model_name == 'ols-3':
        best_model = LinearRegression()
        best_model.fit(train_x, train_y)

    elif model_name == 'linear':
        best_model = LinearRegression()
        best_model.fit(train_x, train_y)

    elif model_name == 'lasso':
        min_mse, best_model = np.inf, None
        for alpha in LASSO_ALPHAS:
            m = Lasso(alpha=alpha, max_iter=5000)
            m.fit(train_x, train_y)
            mse = mean_squared_error(valid_y, m.predict(valid_x))
            if mse < min_mse:
                min_mse, best_model = mse, m

    elif model_name == 'ridge':
        min_mse, best_model = np.inf, None
        for alpha in RIDGE_ALPHAS:
            m = Ridge(alpha=alpha)
            m.fit(train_x, train_y)
            mse = mean_squared_error(valid_y, m.predict(valid_x))
            if mse < min_mse:
                min_mse, best_model = mse, m

    elif model_name == 'rf':
        min_mse, best_model = np.inf, None
        for maxdp in RF_MAX_DEPTHS:
            for max_ft in RF_MAX_FEATS:
                m = RandomForestRegressor(
                    max_depth=maxdp, n_estimators=RF_N_EST,
                    max_features=min(max_ft, train_x.shape[1]),
                    n_jobs=-1, random_state=42
                )
                m.fit(train_x, train_y)
                mse = mean_squared_error(valid_y, m.predict(valid_x))
                if mse < min_mse:
                    min_mse, best_model = mse, m

    elif model_name == 'gbrt+h':
        min_mse, best_model = np.inf, None
        for maxdp in GBRT_DEPTHS:
            for lr in GBRT_LRS:
                m = GradientBoostingRegressor(
                    max_depth=maxdp, learning_rate=lr,
                    n_estimators=GBRT_N_EST, loss='huber', alpha=0.999
                )
                m.fit(train_x, train_y)
                mse = mean_squared_error(valid_y, m.predict(valid_x))
                if mse < min_mse:
                    min_mse, best_model = mse, m

    elif model_name == 'lgbm':
        min_mse, best_model, best_params = np.inf, None, {}
        for maxdp in LGBM_DEPTHS:
            for lr in LGBM_LRS:
                m = lgb.LGBMRegressor(
                    objective='huber', alpha=0.999,
                    max_depth=maxdp, learning_rate=lr,
                    n_estimators=LGBM_N_EST, num_leaves=2 ** maxdp,
                    min_child_samples=LGBM_MIN_CHILD,
                    n_jobs=-1, random_state=42, verbose=-1,
                )
                m.fit(
                    train_x, train_y,
                    eval_set=[(valid_x, valid_y)],
                    callbacks=[
                        lgb.early_stopping(LGBM_EARLY_STOP, verbose=False),
                        lgb.log_evaluation(period=-1),
                    ],
                )
                mse = mean_squared_error(valid_y, m.predict(valid_x))
                if mse < min_mse:
                    min_mse, best_model = mse, m
                    best_params = {'depth': maxdp, 'lr': lr, 'trees': m.best_iteration_}

        print(f"    year {cur_year} | depth={best_params['depth']} "
              f"lr={best_params['lr']} trees={best_params['trees']}/{LGBM_N_EST}")

    else:
        return None

    params_dir = join(PARAMS_DIR, model_name)
    os.makedirs(params_dir, exist_ok=True)
    dump(best_model, join(params_dir, f'year{cur_year}.joblib'))

    y_pred  = best_model.predict(test_x)
    pred_df = pd.DataFrame({
        'id':              test_data['id'].values,
        'DATE':            test_data['DATE'].values,
        'market':          test_data['market'].values,
        'TARGET':          test_data['TARGET'].values,
        'TARGET_original': test_data['TARGET_original'].values,
        'me':              test_data['me'].values,
        'pred':            y_pred
    })

    print(f"    year {cur_year}: train={len(train_data):,}  valid={len(valid_data):,}  "
          f"test={len(test_data):,}  ({test_data['market'].nunique()} markets)")
    return pred_df

#  Evaluation 

def oos_r2(pred, actual):
    ss_res = np.sum((pred - actual) ** 2)
    ss_tot = np.sum(actual ** 2)
    return 1 - ss_res / ss_tot

def rank_correlation(pred, actual):
    corr, _ = spearmanr(pred, actual)
    return corr * 100

def compute_sharpe(forecast_df, weighting='equal'):
    df = forecast_df.copy()
    df['DATE'] = pd.to_datetime(df['DATE'])

    monthly_returns = []
    for date, month_data in df.groupby(pd.Grouper(key='DATE', freq='ME')):
        if len(month_data) < 20:
            continue
        try:
            month_data = month_data.copy()
            month_data['decile'] = pd.qcut(month_data['pred'], q=10,
                                           labels=False, duplicates='drop')
        except ValueError:
            continue

        if month_data['decile'].nunique() < 2:
            continue

        top = month_data[month_data['decile'] == month_data['decile'].max()]
        bot = month_data[month_data['decile'] == month_data['decile'].min()]

        if weighting == 'equal':
            ls_ret = top['TARGET_original'].mean() - bot['TARGET_original'].mean()
        else:
            if top['me'].sum() > 0 and bot['me'].sum() > 0:
                top_ret = (top['TARGET_original'] * top['me']).sum() / top['me'].sum()
                bot_ret = (bot['TARGET_original'] * bot['me']).sum() / bot['me'].sum()
                ls_ret  = top_ret - bot_ret
            else:
                continue
        monthly_returns.append(ls_ret)

    if len(monthly_returns) < 12:
        return np.nan
    arr = np.array(monthly_returns)
    return np.nan if arr.std() == 0 else arr.mean() * np.sqrt(12) / arr.std()


def evaluate_forecast(forecast, label="GLOBAL"):
    n_obs = len(forecast)
    if n_obs == 0:
        return None

    r2        = oos_r2(forecast['pred'].values, forecast['TARGET'].values)
    rank_corr = rank_correlation(forecast['pred'].values, forecast['TARGET'].values)
    sharpe_ew = compute_sharpe(forecast, weighting='equal')
    sharpe_vw = compute_sharpe(forecast, weighting='value')

    print(f"    {label}: R²={r2:.6f}  RankCorr={rank_corr:.4f}  "
          f"SR_EW={sharpe_ew:.4f}  SR_VW={sharpe_vw:.4f}  n={n_obs:,}")

    return {
        'r2_oos': r2, 'rank_corr': rank_corr,
        'sharpe_ew': sharpe_ew, 'sharpe_vw': sharpe_vw, 'n': n_obs
    }

#  Main 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated, e.g. linear,ridge,lgbm')
    args = parser.parse_args()

    models_to_run = (
        [m.strip() for m in args.models.split(',')]
        if args.models else MODELS_TO_RUN
    )

    print(f"Models: {models_to_run}")

    pooled         = load_and_pool_all_markets()
    pooled, dummy_cols = add_country_dummies(pooled)
    pooled         = demean_target_by_month(pooled)

    n_test_years = END_YEAR - VALID_END
    os.makedirs(FORECAST_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    all_results = {}

    for model_name in models_to_run:
        print(f"\n{model_name.upper()}")
        t0       = time.time()
        pred_dfs = []

        for add_year in range(1, n_test_years + 1):
            pred = train_model_year(add_year, model_name, pooled, dummy_cols)
            if pred is not None:
                pred_dfs.append(pred)

        if not pred_dfs:
            continue

        forecast = pd.concat(pred_dfs, ignore_index=True)
        forecast.to_csv(join(FORECAST_DIR, f'{model_name}_pred.csv'), index=False)
        print(f"  {model_name.upper()} done ({time.time()-t0:.1f}s)")

        global_result = evaluate_forecast(forecast, label="GLOBAL")
        all_results[model_name] = {'GLOBAL': global_result}

        for market in sorted(forecast['market'].unique()):
            mkt_fc = forecast[forecast['market'] == market]
            if len(mkt_fc) > 0:
                all_results[model_name][market] = evaluate_forecast(mkt_fc, label=market)

    # Global summary (append-safe)
    global_rows = [
        {'model': model_name, **all_results[model_name]['GLOBAL']}
        for model_name in models_to_run
        if model_name in all_results and all_results[model_name].get('GLOBAL')
    ]
    if global_rows:
        new_global_df = pd.DataFrame(global_rows)
        global_path   = join(SUMMARY_DIR, 'pooled_global_summary.csv')

        if os.path.exists(global_path):
            existing_global = pd.read_csv(global_path)
            existing_global = existing_global[
                ~existing_global['model'].isin(new_global_df['model'])
            ]
            combined_global = pd.concat([existing_global, new_global_df], ignore_index=True)
            combined_global['_order'] = combined_global['model'].apply(
                lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER)
            )
            combined_global = (combined_global
                               .sort_values('_order')
                               .drop(columns='_order')
                               .reset_index(drop=True))
        else:
            combined_global = new_global_df

        combined_global.to_csv(global_path, index=False)
        print(f"Saved → {global_path}")

    # Per-market summary (append-safe)
    combined_rows = [
        {'model': model_name, 'market': market, **all_results[model_name][market]}
        for model_name in models_to_run if model_name in all_results
        for market in sorted(all_results[model_name].keys())
        if market != 'GLOBAL' and all_results[model_name][market]
    ]
    if combined_rows:
        new_df        = pd.DataFrame(combined_rows)
        combined_path = join(SUMMARY_DIR, 'pooled_permarket_summary.csv')

        if os.path.exists(combined_path):
            existing_df = pd.read_csv(combined_path)
            for _, row in new_df.iterrows():
                mask        = ((existing_df['market'] == row['market']) &
                               (existing_df['model']  == row['model']))
                existing_df = existing_df[~mask]
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df['_order'] = combined_df['model'].apply(
                lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER)
            )
            combined_df = (combined_df
                           .sort_values(['market', '_order'])
                           .drop(columns='_order')
                           .reset_index(drop=True))
        else:
            combined_df = new_df

        combined_df.to_csv(combined_path, index=False)
        print(f"Saved → {combined_path}")

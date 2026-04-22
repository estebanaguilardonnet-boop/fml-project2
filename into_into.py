# this code does mkt specific training for non-NN models
# as usual, results and outputs get saved as files locally and are append safe bc like usual, you can comment out some of the models you dont wanna run right now 
# this code combines the training and making the predictions for each market, also calculates the metrics and they got stored in append safe way
# other than results, model parameters get stored too
# the differences in sample periods for each country are predefined based on the paper dates which we also followed upon data downlaods


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from joblib import dump, load
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
PARAMS_DIR   = join(your_path, 'results', 'model_parameters')
FORECAST_DIR = join(your_path, 'results', 'forecasts')
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
                 'lasso',
                 #'ridge',
                 #'rf',
                 #'gbrt+h',
                 'lgbm',
                 ]

MODEL_ORDER = ['ols-3', 'linear', 'lasso', 'ridge', 'rf', 'gbrt+h', 'lgbm']

MARKET_INFO = {
    'Japan':           (1997, 2008, 2010, 2017),
    'China':           (1999, 2004, 2007, 2017),
    'India':           (2007, 2010, 2012, 2017),
    'Korea':           (1997, 2003, 2007, 2017),
    'Hong_Kong':       (1997, 2003, 2007, 2017),
    'Taiwan':          (2007, 2010, 2012, 2017),
    'France':          (1995, 2001, 2005, 2017),
    'United_Kingdom':  (2005, 2008, 2010, 2017),
    'Thailand':        (1997, 2003, 2007, 2017),
    'Australia':       (2008, 2010, 2011, 2017),
    'Singapore':       (2007, 2010, 2012, 2017),
    'Sweden':          (2001, 2005, 2008, 2017),
    'South_Africa':    (1997, 2003, 2007, 2017),
    'Poland':          (2006, 2009, 2011, 2017),
    'Israel':          (2005, 2008, 2010, 2017),
    'Vietnam':         (2010, 2012, 2013, 2017),
    'Italy':           (2001, 2005, 2008, 2017),
    'Turkey':          (2006, 2009, 2011, 2017),
    'Switzerland':     (2002, 2006, 2009, 2017),
    'Indonesia':       (2005, 2008, 2010, 2017),
    'Greece':          (2006, 2009, 2011, 2017),
    'Philippines':     (2006, 2009, 2011, 2017),
    'Norway':          (2007, 2010, 2012, 2017),
    'Sri_Lanka':       (2010, 2012, 2013, 2017),
    'Denmark':         (2007, 2010, 2012, 2017),
    'Finland':         (2007, 2010, 2012, 2017),
    'Saudi_Arabia':    (2010, 2012, 2013, 2017),
    'Jordan':          (2009, 2011, 2012, 2017),
    'Egypt':           (2010, 2012, 2013, 2017),
    'Spain':           (2011, 2012, 2013, 2017),
    'Kuwait':          (2012, 2013, 2014, 2017),
}


def find_csv(market):
    candidates = [f'{market}_clean.csv', f'{market}.csv']
    if market == 'Sri_Lanka':
        candidates.insert(0, 'Sri_lanka.csv')
    for name in candidates:
        path = join(DATA_DIR, name)
        if os.path.exists(path):
            return path
    return None

def load_market_data(market):
    filepath = find_csv(market)
    if filepath is None:
        return None

    start_year, train_end, valid_end, end_year = MARKET_INFO[market]

    df = pd.read_csv(filepath)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True, how='any')
    df = df[df['DATE'] > str(start_year)]
    df = df[df['DATE'] <= str(end_year + 1)]
    df.reset_index(drop=True, inplace=True)
    return df


def get_features(df):
    return df[[c for c in df.columns if c not in META_COLS]]


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

# model training, redone every year

def train_model_year(add_year, model_name, df, train_end, valid_end, market):
    cur_year = valid_end + add_year

    train_data, valid_data, test_data = split_data(df, train_end, valid_end, add_year)

    if len(train_data) == 0 or len(valid_data) == 0 or len(test_data) == 0:
        return None

    train_x = get_features(train_data)
    train_y = train_data['TARGET']
    valid_x = get_features(valid_data)
    valid_y = valid_data['TARGET']
    test_x  = get_features(test_data)

    if model_name == 'ols-3':
        train_x = train_x[OLS3_COLS]
        valid_x = valid_x[OLS3_COLS]
        test_x  = test_x[OLS3_COLS]
        best_model = LinearRegression()
        best_model.fit(train_x, train_y)

    elif model_name == 'linear':
        best_model = LinearRegression()
        best_model.fit(train_x, train_y)

    elif model_name == 'lasso':
        min_mse, best_model = np.inf, None
        for alpha in LASSO_ALPHAS:
            m = Lasso(alpha=alpha)
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
                    max_features=max_ft, n_jobs=-1, random_state=42
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

    params_dir = join(PARAMS_DIR, market, model_name)
    os.makedirs(params_dir, exist_ok=True)
    dump(best_model, join(params_dir, f'year{cur_year}.joblib'))

    y_pred  = best_model.predict(test_x)
    pred_df = pd.DataFrame({
        'id':     test_data['id'].values,
        'DATE':   test_data['DATE'].values,
        'TARGET': test_data['TARGET'].values,
        'me':     test_data['me'].values,
        'pred':   y_pred
    })

    print(f"    year {cur_year}: train={len(train_data):,}  "
          f"valid={len(valid_data):,}  test={len(test_data):,}")
    return pred_df

# calculate metrics
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
            month_data['decile'] = pd.qcut(month_data['pred'], q=10, labels=False, duplicates='drop')
        except ValueError:
            continue

        if month_data['decile'].nunique() < 2:
            continue

        top = month_data[month_data['decile'] == month_data['decile'].max()]
        bot = month_data[month_data['decile'] == month_data['decile'].min()]

        if weighting == 'equal':
            ls_ret = top['TARGET'].mean() - bot['TARGET'].mean()
        else:
            if top['me'].sum() > 0 and bot['me'].sum() > 0:
                top_ret = (top['TARGET'] * top['me']).sum() / top['me'].sum()
                bot_ret = (bot['TARGET'] * bot['me']).sum() / bot['me'].sum()
                ls_ret  = top_ret - bot_ret
            else:
                continue
        monthly_returns.append(ls_ret)

    if len(monthly_returns) < 12:
        return np.nan
    arr = np.array(monthly_returns)
    return np.nan if arr.std() == 0 else arr.mean() * np.sqrt(12) / arr.std()


#  Train one full market

def train_market(market, models_to_run=None):
    if models_to_run is None:
        models_to_run = MODELS_TO_RUN

    print(f"\n{market}")
    df = load_market_data(market)
    if df is None:
        print(f"  CSV not found, skipping")
        return None

    start_year, train_end, valid_end, end_year = MARKET_INFO[market]
    n_test_years = end_year - valid_end

    market_forecast_dir = join(FORECAST_DIR, market)
    os.makedirs(market_forecast_dir, exist_ok=True)

    results = {}
    for model_name in models_to_run:
        print(f"  {model_name.upper()}")
        t0 = time.time()

        pred_dfs = []
        for add_year in range(1, n_test_years + 1):
            pred = train_model_year(add_year, model_name, df, train_end, valid_end, market)
            if pred is not None:
                pred_dfs.append(pred)

        if not pred_dfs:
            continue

        forecast = pd.concat(pred_dfs, ignore_index=True)
        n_obs     = len(forecast)

        r2        = oos_r2(forecast['pred'].values, forecast['TARGET'].values)
        rank_corr = rank_correlation(forecast['pred'].values, forecast['TARGET'].values)
        sharpe_ew = compute_sharpe(forecast, weighting='equal')
        sharpe_vw = compute_sharpe(forecast, weighting='value')

        print(f"    R²={r2:.6f}  RankCorr={rank_corr:.4f}  "
              f"SR_EW={sharpe_ew:.4f}  SR_VW={sharpe_vw:.4f}  "
              f"n={n_obs:,}  ({time.time()-t0:.1f}s)")

        results[model_name] = {
            'r2_oos': r2, 'rank_corr': rank_corr,
            'sharpe_ew': sharpe_ew, 'sharpe_vw': sharpe_vw, 'n': n_obs
        }

        forecast.to_csv(join(market_forecast_dir, f'{model_name}_pred.csv'), index=False)

    return results


# --- Main 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', type=str, default=None)
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated, e.g. linear,ridge,lgbm')
    args = parser.parse_args()

    models_to_run = (
        [m.strip() for m in args.models.split(',')]
        if args.models else MODELS_TO_RUN
    )
    markets = [args.market] if args.market else sorted(MARKET_INFO.keys())

    print(f"Markets: {len(markets)}  |  Models: {models_to_run}")

    all_results = {}
    for market in markets:
        result = train_market(market, models_to_run)
        if result is not None:
            all_results[market] = result

    os.makedirs(SUMMARY_DIR, exist_ok=True)

    # per-model CSVs
    for model_name in models_to_run:
        rows = [
            {'market': market, **all_results[market][model_name]}
            for market in sorted(all_results.keys())
            if model_name in all_results[market]
        ]
        if rows:
            pd.DataFrame(rows).to_csv(
                join(SUMMARY_DIR, f'local_{model_name}_summary.csv'), index=False
            )

    # combined summary (append-safe)
    combined_rows = [
        {'market': market, 'model': model_name, **r}
        for market in sorted(all_results.keys())
        for model_name, r in all_results[market].items()
    ]
    if combined_rows:
        new_df        = pd.DataFrame(combined_rows)
        combined_path = join(SUMMARY_DIR, 'local_training_summary.csv')

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

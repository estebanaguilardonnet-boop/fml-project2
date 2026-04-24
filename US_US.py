# this code has non-NN models train on US data only, my directory set to a folder, results stored in a newly created folder. US data retrieved from subfolder in directory
# MODELS_TO_RUN allows you to choose which model to run so that dont have to do all at once. 
  #re-running code w/ different models won't overide exisiting results, they get appended (unless rerun something that was alr there, then it will just replaec)
# code also makes predictions on US stocks (which are also stored), with that it also calculates some of the metrics used (R2, rank corr, EW+VW sharpe)
    # these stored in a 'summary' folder
# model parameters stored throughout process


from pathlib import Path
import pandas as pd
import numpy as np
import time
import subprocess
import sys

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from joblib import dump

try:
    import lightgbm as lgb
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightgbm', '-q'])
    import lightgbm as lgb


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'raw_data' / 'cleenerst'
RES_DIR  = BASE_DIR / 'results'
US_CSV   = DATA_DIR / 'USA_clean.csv'

PARAMS_DIR   = RES_DIR / 'model_parameters' / 'USA'
FORECAST_DIR = RES_DIR / 'forecasts' / 'USA'
SUMMARY_DIR  = RES_DIR / 'summary'

for model in ['ols-3', 'linear', 'lasso', 'ridge', 'rf', 'gbrt+h', 'lgbm']:
    (PARAMS_DIR / model).mkdir(parents=True, exist_ok=True)
FORECAST_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

META_COLS   = ['id', 'DATE', 'TARGET', 'me']
OLS3_COLS   = ['mvel1', 'bm', 'mom_12']

START_YEAR  = 1963
TRAIN_SPLIT = 1979
VALID_SPLIT = 1989
END_YEAR    = 2017

LASSO_ALPHAS   = np.logspace(-3, 3, 10)
RIDGE_ALPHAS   = np.logspace(-3, 3, 10)
RF_MAX_DEPTHS  = [2, 4, 6]
RF_MAX_FEATS   = [3, 5, 10]
GBRT_DEPTHS    = [1]
GBRT_LRS       = [0.1]
GBRT_N_EST     = 200

LGBM_DEPTHS     = [1, 2]
LGBM_LRS        = [0.01, 0.1]
LGBM_N_EST      = 1000
LGBM_EARLY_STOP = 200
LGBM_MIN_CHILD  = 50

MODELS_TO_RUN = [
    #'ols-3',
    #'linear',
    #'lasso',
    #'ridge',
    #'rf',
    #'gbrt+h',
    'lgbm',
]

MODEL_ORDER = ['ols-3', 'linear', 'lasso', 'ridge', 'rf', 'gbrt+h', 'lgbm']


def load_and_prep(csv_path, start_year=None, end_year=None):
    df = pd.read_csv(csv_path)
    for col in ['ff49', 'excntry']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df['DATE']   = pd.to_datetime(df['DATE'], format='mixed')
    df['TARGET'] = df['TARGET'].astype(float)

    if start_year:
        df = df[df['DATE'] > pd.to_datetime(f'{start_year}-01-01')]
    if end_year:
        df = df[df['DATE'] <= pd.to_datetime(f'{end_year}-12-31')]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Loaded {len(df):,} rows ({df['DATE'].dt.year.min()}–{df['DATE'].dt.year.max()})")
    return df


def in_output(df):
    feature_cols = [c for c in df.columns if c not in META_COLS]
    return df[feature_cols], df['TARGET']


def split_data(df, add_year):
    t_end = pd.to_datetime(f'{TRAIN_SPLIT + add_year}-12-31')
    v_end = pd.to_datetime(f'{VALID_SPLIT + add_year}-12-31')
    x_end = pd.to_datetime(f'{VALID_SPLIT + add_year + 1}-12-31')

    train = df[df['DATE'] <= t_end].reset_index(drop=True)
    valid = df[(df['DATE'] > t_end) & (df['DATE'] <= v_end)].reset_index(drop=True)
    test  = df[(df['DATE'] > v_end) & (df['DATE'] <= x_end)].reset_index(drop=True)
    return train, valid, test


def train_one_year(data, add_year, model_name):
    cur_year   = VALID_SPLIT + add_year
    model_path = PARAMS_DIR / model_name / f'year{cur_year}.joblib'

    train, valid, test = split_data(data, add_year)
    if len(train) == 0 or len(test) == 0:
        return None

    train_x, train_y = in_output(train)
    valid_x, valid_y = in_output(valid)
    test_x,  _       = in_output(test)

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
        best_model, min_mse = None, np.inf
        for alpha in LASSO_ALPHAS:
            m = Lasso(alpha=alpha, max_iter=10000)
            m.fit(train_x, train_y)
            mse = mean_squared_error(valid_y, m.predict(valid_x))
            if mse < min_mse:
                min_mse, best_model = mse, m

    elif model_name == 'ridge':
        best_model, min_mse = None, np.inf
        for alpha in RIDGE_ALPHAS:
            m = Ridge(alpha=alpha)
            m.fit(train_x, train_y)
            mse = mean_squared_error(valid_y, m.predict(valid_x))
            if mse < min_mse:
                min_mse, best_model = mse, m

    elif model_name == 'rf':
        best_model, min_mse = None, np.inf
        for max_depth in RF_MAX_DEPTHS:
            for max_feat in RF_MAX_FEATS:
                m = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=max_depth,
                    max_features=max_feat,
                    n_jobs=-1,
                    random_state=42
                )
                m.fit(train_x, train_y)
                mse = mean_squared_error(valid_y, m.predict(valid_x))
                if mse < min_mse:
                    min_mse, best_model = mse, m

    elif model_name == 'gbrt+h':
        best_model, min_mse = None, np.inf
        for max_depth in GBRT_DEPTHS:
            for lr in GBRT_LRS:
                m = GradientBoostingRegressor(
                    n_estimators=GBRT_N_EST,
                    max_depth=max_depth,
                    learning_rate=lr,
                    loss='huber',
                    alpha=0.999,
                    random_state=42
                )
                m.fit(train_x, train_y)
                mse = mean_squared_error(valid_y, m.predict(valid_x))
                if mse < min_mse:
                    min_mse, best_model = mse, m

    elif model_name == 'lgbm':
        best_model, min_mse = None, np.inf
        best_params = {}
        for max_depth in LGBM_DEPTHS:
            for lr in LGBM_LRS:
                m = lgb.LGBMRegressor(
                    objective='huber',
                    alpha=0.999,
                    max_depth=max_depth,
                    learning_rate=lr,
                    n_estimators=LGBM_N_EST,
                    num_leaves=2 ** max_depth,
                    min_child_samples=LGBM_MIN_CHILD,
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1,
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
                    min_mse    = mse
                    best_model = m
                    best_params = {'depth': max_depth, 'lr': lr, 'trees': m.best_iteration_}

        print(f"    year {cur_year} | depth={best_params['depth']} lr={best_params['lr']} trees={best_params['trees']}")

    else:
        raise ValueError(f"Unknown model: '{model_name}'")

    dump(best_model, model_path)

    y_hat   = best_model.predict(test_x)
    pred_df = test[['id', 'DATE', 'TARGET', 'me']].copy()
    pred_df['pred']     = y_hat
    pred_df['cur_year'] = cur_year
    return pred_df


def train_all_years(data, model_name):
    print(f"\nTraining: {model_name.upper()} | windows {VALID_SPLIT+1}–{END_YEAR}")
    results     = []
    total_start = time.time()

    for add_year in range(1, END_YEAR - VALID_SPLIT + 1):
        cur_year = VALID_SPLIT + add_year
        t0       = time.time()
        pred     = train_one_year(data, add_year, model_name)
        if pred is not None:
            results.append(pred)
            print(f"  year {cur_year} done ({time.time()-t0:.1f}s)")

    print(f"  {model_name.upper()} complete — {(time.time()-total_start)/60:.1f} min")
    return pd.concat(results).reset_index(drop=True) if results else pd.DataFrame()


def oos_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    return 1 - ss_res / ss_tot


def rank_correlation(y_true, y_pred):
    rho, _ = spearmanr(y_true, y_pred)
    return rho * 100


def long_short_sharpe(forecast_df, top_pct=0.1, annualise=12):
    monthly = []
    for _, grp in forecast_df.groupby('DATE'):
        if len(grp) < 20:
            continue
        n      = max(1, int(len(grp) * top_pct))
        grp_s  = grp.sort_values('pred')
        ls_ret = grp_s.iloc[-n:]['TARGET'].mean() - grp_s.iloc[:n]['TARGET'].mean()
        monthly.append(ls_ret)

    monthly = np.array(monthly)
    if len(monthly) < 2 or monthly.std() == 0:
        return np.nan
    return (monthly.mean() / monthly.std()) * np.sqrt(annualise)


def long_short_sharpe_vw(forecast_df, top_pct=0.1, annualise=12):
    monthly = []
    for _, grp in forecast_df.groupby('DATE'):
        if len(grp) < 20:
            continue
        n     = max(1, int(len(grp) * top_pct))
        grp_s = grp.sort_values('pred')

        top      = grp_s.iloc[-n:]
        top_w    = top['me'] / top['me'].sum()
        long_ret = (top_w * top['TARGET']).sum()

        bot       = grp_s.iloc[:n]
        bot_w     = bot['me'] / bot['me'].sum()
        short_ret = (bot_w * bot['TARGET']).sum()

        monthly.append(long_ret - short_ret)

    monthly = np.array(monthly)
    if len(monthly) < 2 or monthly.std() == 0:
        return np.nan
    return (monthly.mean() / monthly.std()) * np.sqrt(annualise)


def evaluate(forecast_df, model_name):
    if len(forecast_df) == 0:
        return {}

    y_true = forecast_df['TARGET'].values
    y_pred = forecast_df['pred'].values

    r2        = oos_r2(y_true, y_pred)
    rc        = rank_correlation(y_true, y_pred)
    sharpe    = long_short_sharpe(forecast_df)
    sharpe_vw = long_short_sharpe_vw(forecast_df) if 'me' in forecast_df.columns else np.nan

    print(f"\n  {model_name.upper()}")
    print(f"  OOS R²       : {r2:.6f}")
    print(f"  Rank Corr    : {rc:.4f}")
    print(f"  Sharpe EW    : {sharpe:.4f}")
    print(f"  Sharpe VW    : {sharpe_vw:.4f}")

    return {
        'model':     model_name,
        'r2_oos':    round(r2, 8),
        'rank_corr': round(rc, 4),
        'sharpe_ew': round(sharpe, 4),
        'sharpe_vw': round(sharpe_vw, 4) if not np.isnan(sharpe_vw) else np.nan,
        'n':         len(forecast_df),
    }


def build_full_summary():
    pred_files = list(FORECAST_DIR.glob('*_pred.csv'))
    if not pred_files:
        return pd.DataFrame()

    rows = []
    for pred_file in pred_files:
        model_name = pred_file.stem.replace('_pred', '')
        forecasts  = pd.read_csv(pred_file, parse_dates=['DATE'])
        metrics    = evaluate(forecasts, model_name)
        if metrics:
            rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    summary_df    = pd.DataFrame(rows).set_index('model')
    ordered_index = [m for m in MODEL_ORDER if m in summary_df.index]
    remaining     = [m for m in summary_df.index if m not in MODEL_ORDER]
    return summary_df.loc[ordered_index + remaining]


if __name__ == '__main__':
    print(f"Models: {MODELS_TO_RUN}")
    us_data = load_and_prep(US_CSV, start_year=START_YEAR, end_year=END_YEAR)

    for model_name in MODELS_TO_RUN:
        forecasts = train_all_years(us_data, model_name)
        if len(forecasts) > 0:
            save_path = FORECAST_DIR / f'{model_name}_pred.csv'
            forecasts.to_csv(save_path, index=False)
            print(f"  Saved → {save_path}")

    print("\nFULL SUMMARY — USA")
    summary_df = build_full_summary()
    if len(summary_df) > 0:
        print(f"\n{summary_df.to_string()}")
        summary_path = SUMMARY_DIR / 'USA_summary.csv'
        summary_df.to_csv(summary_path)
        print(f"  Saved → {summary_path}")

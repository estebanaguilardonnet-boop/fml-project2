#the US_US code stored model info for each non-NN model after training on US, this code is retrieving that and applying the characteristic values of the other 31 countries to predict their returns
  #basically fitting the US-trained models to characteristic values of stocks for the 31 other countries
# again, you can comment out models dont wanna retrieve from - predictions and metric calcualted are appended so no risk of overriding unless a model was run before
# each country got different time period windows, so its pre-defined based on paper dates (which our raw data alligns with)
# predictions stored and measures calculated - both stored in different places and yea, append safe


import pandas as pd
import numpy as np
import os
from os.path import join
from joblib import load
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightgbm', '-q'])
    import lightgbm


your_path    = '/Users/valentin/Desktop/FML /group'
DATA_DIR     = join(your_path, 'raw_data', 'cleenerst')
PARAMS_DIR   = join(your_path, 'results', 'model_parameters', 'USA')
FORECAST_DIR = join(your_path, 'results', 'forecasts_USmodel')
SUMMARY_DIR  = join(your_path, 'results', 'summary')

META_COLS = ['id', 'DATE', 'TARGET', 'me']
OLS3_COLS = ['mvel1', 'bm', 'mom_12']

US_MODEL_FIRST_YEAR = 1990
US_MODEL_LAST_YEAR  = 2016

MODELS_TO_RUN = ['ols-3', 
                 'linear',
                 'lasso', 
                 'ridge',
                 'rf',
                 #'gbrt+h',
                 #'lgbm']

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


def load_market_data(market):
    possible_names = [f'{market}_clean.csv', f'{market}.csv']
    if market == 'Sri_Lanka':
        possible_names.insert(0, 'Sri_lanka.csv')

    filepath = next(
        (join(DATA_DIR, n) for n in possible_names if os.path.exists(join(DATA_DIR, n))),
        None
    )
    if filepath is None:
        return None, None

    df = pd.read_csv(filepath)
    start_year, train_end, valid_end, end_year = MARKET_INFO[market]

    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True, how='any')
    df = df[df['DATE'] > str(start_year)]
    df = df[df['DATE'] <= str(end_year + 1)]
    df.reset_index(drop=True, inplace=True)

    return df, MARKET_INFO[market]

def get_features(df):
    return df[[c for c in df.columns if c not in META_COLS]]

def oos_r2(pred, actual):
    ss_res = np.sum((pred - actual) ** 2)
    ss_tot = np.sum(actual ** 2)
    return 1 - ss_res / ss_tot

def rank_correlation(pred, actual):
    corr, _ = spearmanr(pred, actual)
    return corr * 100


def compute_sharpe(forecast_df, weighting='equal'):
    forecast_df = forecast_df.copy()
    forecast_df['DATE'] = pd.to_datetime(forecast_df['DATE'])

    monthly_returns = []
    for date, month_data in forecast_df.groupby(pd.Grouper(key='DATE', freq='M')):
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
            top_ret = top['TARGET'].mean()
            bot_ret = bot['TARGET'].mean()
        else:
            if top['me'].sum() > 0 and bot['me'].sum() > 0:
                top_ret = (top['TARGET'] * top['me']).sum() / top['me'].sum()
                bot_ret = (bot['TARGET'] * bot['me']).sum() / bot['me'].sum()
            else:
                continue

        monthly_returns.append(top_ret - bot_ret)

    if len(monthly_returns) < 12:
        return np.nan

    monthly_returns = np.array(monthly_returns)
    std = monthly_returns.std()
    return np.nan if std == 0 else monthly_returns.mean() * np.sqrt(12) / std

def predict_market_year(cur_year, model_name, market_data):
    test_data = market_data[
        (market_data['DATE'] >= str(cur_year)) &
        (market_data['DATE'] < str(cur_year + 1))
    ].copy()
    test_data.reset_index(drop=True, inplace=True)

    if len(test_data) == 0:
        return None

    test_x     = get_features(test_data)
    if model_name == 'ols-3':
        test_x = test_x[OLS3_COLS]

    model_path = join(PARAMS_DIR, model_name, f'year{cur_year}.joblib')
    if not os.path.exists(model_path):
        return None

    y_pred = load(model_path).predict(test_x)
    return pd.DataFrame({
        'id':   test_data['id'].values,
        'DATE': test_data['DATE'].values,
        'pred': y_pred
    })

def predict_market(market):
    print(f"\n{market}")
    market_data, info = load_market_data(market)
    if market_data is None:
        print(f"  CSV not found, skipping")
        return None

    start_year, train_end, valid_end, end_year = info
    test_start = valid_end + 1
    test_end   = min(end_year, US_MODEL_LAST_YEAR)

    market_forecast_dir = join(FORECAST_DIR, market)
    os.makedirs(market_forecast_dir, exist_ok=True)

    actuals = market_data[['id', 'DATE', 'TARGET', 'me']].copy()
    results = {}

    for model_name in MODELS_TO_RUN:
        pred_dfs = [
            predict_market_year(cur_year, model_name, market_data)
            for cur_year in range(test_start, test_end + 1)
        ]
        pred_dfs = [p for p in pred_dfs if p is not None]

        if not pred_dfs:
            continue

        forecast = pd.merge(actuals, pd.concat(pred_dfs, ignore_index=True), on=['id', 'DATE'])
        if len(forecast) == 0:
            continue

        r2        = oos_r2(forecast['pred'].values, forecast['TARGET'].values)
        rank_corr = rank_correlation(forecast['pred'].values, forecast['TARGET'].values)
        sharpe_ew = compute_sharpe(forecast, weighting='equal')
        sharpe_vw = compute_sharpe(forecast, weighting='value')

        print(f"  {model_name:8s} | R²={r2:.4f}  RankCorr={rank_corr:.2f}  "
              f"SR_EW={sharpe_ew:.3f}  SR_VW={sharpe_vw:.3f}  n={len(forecast):,}")

        results[model_name] = {
            'r2_oos': r2, 'rank_corr': rank_corr,
            'sharpe_ew': sharpe_ew, 'sharpe_vw': sharpe_vw,
            'n': len(forecast)
        }

        forecast.to_csv(join(market_forecast_dir, f'{model_name}_pred.csv'), index=False)

    return results

if __name__ == '__main__':
    # check which markets have data available
    available_markets = []
    for market in sorted(MARKET_INFO.keys()):
        possible = [f'{market}_clean.csv', f'{market}.csv']
        if market == 'Sri_Lanka':
            possible.insert(0, 'Sri_lanka.csv')
        if any(os.path.exists(join(DATA_DIR, f)) for f in possible):
            available_markets.append(market)

    print(f"{len(available_markets)}/{len(MARKET_INFO)} markets found")

    all_results = {}
    for market in available_markets:
        result = predict_market(market)
        if result is not None:
            all_results[market] = result

    os.makedirs(SUMMARY_DIR, exist_ok=True)

    # per-model summary tables
    for model_name in MODELS_TO_RUN:
        rows = [
            {'market': market, **all_results[market][model_name]}
            for market in sorted(all_results.keys())
            if model_name in all_results[market]
        ]
        if rows:
            pd.DataFrame(rows).to_csv(
                join(SUMMARY_DIR, f'international_{model_name}_summary.csv'), index=False
            )

    # combined summary
    combined_rows = [
        {'market': market, 'model': model_name, **r}
        for market in sorted(all_results.keys())
        for model_name, r in all_results[market].items()
    ]
    if combined_rows:
        combined_path = join(SUMMARY_DIR, 'international_USmodel_summary.csv')
        pd.DataFrame(combined_rows).to_csv(combined_path, index=False)
        print(f"\nSaved combined summary → {combined_path}")

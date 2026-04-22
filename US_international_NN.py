#code to make predictions on 31 other countries based of us training, but only for the NN models. possible since i stored that info in .pt files 
# ive had some issues loading sri lanka bc it was downloaded a few times, sometimes with the space in name, one or two times not (sri lanka vs sri_lanka)
# the predictioins saved and append safe so can comment out models dont want running
# metrics caclulated and stored, also append safe
# its the same logic as the US_international.py file, but now only for the NN models bc those had been done seperately

# if you want to run this yourself, the first chucnk after packages is part have to be careful - the directory is set locally and sensitive to naming of folders and where things stored

import pandas as pd
import numpy as np
import os
from os.path import join
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from scipy.stats import spearmanr


your_path    = '/Users/valentin/Desktop/FML /group'
DATA_DIR     = join(your_path, 'raw_data', 'cleenerst')
US_PARAMS    = join(your_path, 'results', 'model_parameters', 'USA')
FORECAST_DIR = join(your_path, 'results', 'forecasts_USmodel')
SUMMARY_DIR  = join(your_path, 'results', 'summary')

DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
META_COLS    = ['id', 'DATE', 'TARGET', 'me']
NN_NUM_SEEDS = 10

MODELS_TO_RUN = ['nn1',
                 'nn2',
                 'nn3'
                ]
MODEL_ORDER   = ['nn1', 'nn2', 'nn3']

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

#  Network architectures

class NN1(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1    = nn.Linear(input_size, 32)
        self.bn1    = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 1)
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        return self.output(x)

class NN2(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1    = nn.Linear(input_size, 32)
        self.bn1    = nn.BatchNorm1d(32)
        self.fc2    = nn.Linear(32, 16)
        self.bn2    = nn.BatchNorm1d(16)
        self.output = nn.Linear(16, 1)
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.output(x)

class NN3(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1    = nn.Linear(input_size, 32)
        self.bn1    = nn.BatchNorm1d(32)
        self.fc2    = nn.Linear(32, 16)
        self.bn2    = nn.BatchNorm1d(16)
        self.fc3    = nn.Linear(16, 8)
        self.bn3    = nn.BatchNorm1d(8)
        self.output = nn.Linear(8, 1)
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        return self.output(x)

NN_CLASS_MAP = {'nn1': NN1, 'nn2': NN2, 'nn3': NN3}

# load data 

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


def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]



# load US training model info that were stored as .pt files !

def load_us_ensemble(model_name, year, input_size):
    nn_class  = NN_CLASS_MAP[model_name]
    model_dir = join(US_PARAMS, model_name)
    models    = []

    for seed in range(NN_NUM_SEEDS):
        ckpt = join(model_dir, f'year{year}_seed{seed}.pt')
        if not os.path.exists(ckpt):
            return None
        model = nn_class(input_size)
        model.load_state_dict(
            torch.load(ckpt, map_location=DEVICE, weights_only=True)
        )
        model.to(DEVICE)
        model.eval()
        models.append(model)

    return models

#Prediction (month-by-month due to BatchNorm) 

def predict_year(year_data, ensemble, feature_cols):
    for m in ensemble:
        m.eval()

    year_data = year_data.copy()
    year_data['DATE'] = pd.to_datetime(year_data['DATE'], format='mixed')

    monthly_preds = []
    for _, month_grp in year_data.groupby(year_data['DATE'].dt.to_period('M')):
        if len(month_grp) < 2:
            continue

        x_t = torch.from_numpy(
            month_grp[feature_cols].values
        ).float().to(DEVICE)

        with torch.no_grad():
            preds = torch.cat(
                [model(x_t) for model in ensemble], dim=1
            ).mean(dim=1).cpu().numpy()

        chunk = month_grp[['id', 'DATE', 'TARGET', 'me']].copy()
        chunk['pred'] = preds
        monthly_preds.append(chunk)

    if not monthly_preds:
        return pd.DataFrame()
    return pd.concat(monthly_preds, ignore_index=True)

# metrics calculations

def oos_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    return 1 - ss_res / ss_tot


def rank_correlation(y_true, y_pred):
    rho, _ = spearmanr(y_true, y_pred)
    return rho * 100


def compute_sharpe(forecast_df, weighting='equal'):
    df = forecast_df.copy()
    df['DATE'] = pd.to_datetime(df['DATE'])

    monthly_returns = []
    for _, month_data in df.groupby(pd.Grouper(key='DATE', freq='ME')):
        if len(month_data) < 20:
            continue
        try:
            month_data = month_data.copy()
            month_data['decile'] = pd.qcut(
                month_data['pred'], q=10, labels=False, duplicates='drop')
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


def evaluate_forecast(forecast, label):
    n_obs = len(forecast)
    if n_obs == 0:
        return None

    r2        = oos_r2(forecast['TARGET'].values, forecast['pred'].values)
    rank_corr = rank_correlation(forecast['TARGET'].values, forecast['pred'].values)
    sharpe_ew = compute_sharpe(forecast, weighting='equal')
    sharpe_vw = compute_sharpe(forecast, weighting='value')

    print(f"    R²={r2:.6f}  RankCorr={rank_corr:.4f}  "
          f"Sharpe_EW={sharpe_ew:.4f}  Sharpe_VW={sharpe_vw:.4f}  N={n_obs:,}")

    return {
        'r2_oos':    round(r2, 8),
        'rank_corr': round(rank_corr, 4),
        'sharpe_ew': round(sharpe_ew, 4),
        'sharpe_vw': round(sharpe_vw, 4) if not np.isnan(sharpe_vw) else np.nan,
        'n':         n_obs,
    }

# predict one makt
def predict_market(market, models_to_run):
    print(f"\n{market}")
    df = load_market_data(market)
    if df is None:
        print(f"  CSV not found, skipping")
        return None

    _, train_end, valid_end, end_year = MARKET_INFO[market]
    feature_cols = get_feature_cols(df)
    n_features   = len(feature_cols)

    market_forecast_dir = join(FORECAST_DIR, market)
    os.makedirs(market_forecast_dir, exist_ok=True)

    results = {}

    for model_name in models_to_run:
        t0        = time.time()
        pred_dfs  = []
        n_skipped = 0

        for test_year in range(valid_end + 1, end_year + 1):
            # NOTE: year_data assigned twice here — preserved from original
            year_data = df[(df['DATE'] > str(test_year - 1)) &
                           (df['DATE'] <= str(test_year))].copy()
            year_data = df[(df['DATE'] > str(test_year - 1)) &
                           (df['DATE'] <= str(test_year))].copy()

            if len(year_data) == 0:
                n_skipped += 1
                continue

            ensemble = load_us_ensemble(model_name, test_year, n_features)
            if ensemble is None:
                n_skipped += 1
                continue

            pred = predict_year(year_data, ensemble, feature_cols)
            if len(pred) > 0:
                pred_dfs.append(pred)

        if not pred_dfs:
            continue

        forecast = pd.concat(pred_dfs, ignore_index=True)
        print(f"  {model_name.upper()} ({time.time()-t0:.1f}s, {n_skipped} skipped)")
        metrics = evaluate_forecast(forecast, market)

        if metrics:
            results[model_name] = metrics

        forecast.to_csv(join(market_forecast_dir, f'{model_name}_pred.csv'), index=False)

    return results

# -main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', type=str, default=None)
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated, e.g. nn1,nn3')
    args = parser.parse_args()

    models_to_run = (
        [m.strip().lower() for m in args.models.split(',')]
        if args.models else MODELS_TO_RUN
    )
    markets = [args.market] if args.market else sorted(MARKET_INFO.keys())

    print(f"Device: {DEVICE}  |  Markets: {len(markets)}  |  Models: {models_to_run}")

    all_results = {}
    for market in markets:
        result = predict_market(market, models_to_run)
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
                join(SUMMARY_DIR, f'USmodel_nn_{model_name}_summary.csv'), index=False
            )

    # combined summary (append-safe)
    combined_rows = [
        {'market': market, 'model': model_name, **r}
        for market in sorted(all_results.keys())
        for model_name, r in all_results[market].items()
    ]
    if combined_rows:
        new_df        = pd.DataFrame(combined_rows)
        combined_path = join(SUMMARY_DIR, 'USmodel_nn_summary.csv')

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

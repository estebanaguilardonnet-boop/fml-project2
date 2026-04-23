#like into_into.py , this does mkt specific training but only for NN models
# model 'parameter' (NN equivalent) stored in .pt files throughout process, maybe useful for later? cka metric?
# the code also makes the predictions and stores them, as well as the metrics
# like all other code, you can comment out which model to ignore. and as usual, its all append safe so you will only overide exisitng results if re-run code on a model that was done before

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
import random
from scipy.stats import spearmanr


your_path    = '/Users/valentin/Desktop/FML /group'
DATA_DIR     = join(your_path, 'raw_data', 'cleenerst')
PARAMS_DIR   = join(your_path, 'results', 'model_parameters')
FORECAST_DIR = join(your_path, 'results', 'forecasts')
SUMMARY_DIR  = join(your_path, 'results', 'summary')

DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
META_COLS = ['id', 'DATE', 'TARGET', 'me']

NN_EPOCH_NUM     = 100
NN_LAMBDA1_LIST  = [1e-5, 1e-4, 1e-3]
NN_LEARNING_RATE = 1e-2
NN_NUM_SEEDS     = 10
NN_PATIENCE      = 5
NN_BATCH_SIZE    = 10_000

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

# --- Network architectures 

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

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

# --- Prediction (month-by-month due to BatchNorm) 

def nn_predict_monthly(df, best_model_list, feature_cols):
    for m in best_model_list:
        m.eval()

    df = df.copy()
    df['DATE'] = pd.to_datetime(df['DATE'], format='mixed')

    monthly_preds = []
    for _, month_grp in df.groupby(df['DATE'].dt.to_period('M')):
        if len(month_grp) < 2:
            continue
        x_tensor = torch.from_numpy(month_grp[feature_cols].values).float().to(DEVICE)

        with torch.no_grad():
            preds = torch.cat(
                [model(x_tensor) for model in best_model_list], dim=1
            ).mean(dim=1).cpu().numpy()

        chunk = month_grp[['id', 'DATE', 'TARGET', 'me']].copy()
        chunk['pred'] = preds
        monthly_preds.append(chunk)

    if not monthly_preds:
        return pd.DataFrame()
    return pd.concat(monthly_preds, ignore_index=True)


#annial retraining 

def train_nn_year(add_year, model_name, df, train_end, valid_end, market, feature_cols):
    cur_year = valid_end + add_year

    train_data, valid_data, test_data = split_data(df, train_end, valid_end, add_year)

    if len(train_data) == 0 or len(valid_data) == 0 or len(test_data) == 0:
        return None, None

    train_x = train_data[feature_cols].values
    train_y = train_data['TARGET'].values
    valid_x = valid_data[feature_cols].values
    valid_y = valid_data['TARGET'].values

    nn_class   = NN_CLASS_MAP[model_name]
    input_size = train_x.shape[1]
    batch_size = min(NN_BATCH_SIZE, len(train_x))
    h          = int(len(train_x) / batch_size) + 1

    inputs  = torch.from_numpy(train_x).float()
    targets = torch.from_numpy(train_y).float().view(-1, 1)
    val_x_t = torch.from_numpy(valid_x).float().to(DEVICE)
    val_y_t = torch.from_numpy(valid_y).float().view(-1, 1).to(DEVICE)

    save_dir = join(PARAMS_DIR, market, model_name)
    os.makedirs(save_dir, exist_ok=True)

    best_model_list = []

    for seed in range(NN_NUM_SEEDS):
        torch.manual_seed(seed)

        best_valid_error = np.inf
        best_model_seed  = None

        for lambda1 in NN_LAMBDA1_LIST:
            tmp_model = nn_class(input_size)
            tmp_model.to(DEVICE)
            tmp_model.apply(weights_init)

            optimizer = torch.optim.Adam(tmp_model.parameters(), lr=NN_LEARNING_RATE)
            criterion = nn.MSELoss()

            epoch, no_impr  = 0, 0
            tmp_valid_error = np.inf
            tmp_best_model  = None

            while epoch < NN_EPOCH_NUM and no_impr < NN_PATIENCE:
                tmp_model.train()
                permutation = torch.randperm(len(inputs))

                for _ in range(h):
                    i   = random.randint(0, max(0, len(inputs) - batch_size))
                    idx = permutation[i : i + batch_size]
                    bx  = inputs[idx].to(DEVICE)
                    by  = targets[idx].to(DEVICE)

                    l1_reg = sum(p.abs().sum() for p in tmp_model.parameters())
                    out    = tmp_model(bx)
                    loss   = criterion(out, by) + lambda1 * l1_reg

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                epoch += 1
                tmp_model.eval()
                with torch.no_grad():
                    val_loss = criterion(tmp_model(val_x_t), val_y_t).item()

                if val_loss < tmp_valid_error:
                    no_impr         = 0
                    tmp_valid_error = val_loss
                    tmp_best_model  = tmp_model
                else:
                    no_impr += 1

            if tmp_valid_error < best_valid_error:
                best_valid_error = tmp_valid_error
                best_model_seed  = tmp_best_model
                torch.save(best_model_seed.state_dict(),
                           join(save_dir, f'year{cur_year}_seed{seed}.pt'))

        best_model_list.append(best_model_seed)

    pred_df = nn_predict_monthly(test_data, best_model_list, feature_cols)
    if len(pred_df) == 0:
        return None, None

    print(f"    year {cur_year}: train={len(train_data):,}  "
          f"valid={len(valid_data):,}  test={len(test_data):,}")
    return pred_df, best_model_list

# get them metrics calculated

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

#ssD calculaition

def compute_ssd(model_list, x_np, feature_names, max_obs=50_000):
    if len(x_np) > max_obs:
        idx  = np.random.choice(len(x_np), max_obs, replace=False)
        x_np = x_np[idx]

    n_features = x_np.shape[1]
    ssd_accum  = np.zeros(n_features)

    for model in model_list:
        model.eval()
        x_t = torch.from_numpy(x_np).float().to(DEVICE)
        x_t.requires_grad_(True)

        out = model(x_t)
        out.sum().backward()

        ssd_accum += (x_t.grad.cpu().numpy() ** 2).sum(axis=0)
        x_t.requires_grad_(False)

    ssd_accum /= len(model_list)
    total = ssd_accum.sum()
    if total > 0:
        ssd_accum /= total

    return pd.Series(ssd_accum, index=feature_names)

# train one full market 

def train_market(market, models_to_run=None):
    if models_to_run is None:
        models_to_run = MODELS_TO_RUN

    print(f"\n{market}")
    df = load_market_data(market)
    if df is None:
        print(f"  CSV not found, skipping")
        return None

    _, train_end, valid_end, end_year = MARKET_INFO[market]
    n_test_years = end_year - valid_end
    feature_cols = get_feature_cols(df)

    market_forecast_dir = join(FORECAST_DIR, market)
    os.makedirs(market_forecast_dir, exist_ok=True)

    results        = {}
    last_ensembles = {}

    for model_name in models_to_run:
        print(f"  {model_name.upper()}")
        t0            = time.time()
        pred_dfs      = []
        year_ensemble = None

        for add_year in range(1, n_test_years + 1):
            pred, ensemble = train_nn_year(
                add_year, model_name, df, train_end, valid_end,
                market, feature_cols
            )
            if pred is not None:
                pred_dfs.append(pred)
                year_ensemble = ensemble

        if not pred_dfs:
            continue

        forecast = pd.concat(pred_dfs, ignore_index=True)

        if year_ensemble is not None:
            last_ensembles[model_name] = year_ensemble

        r2        = oos_r2(forecast['pred'].values, forecast['TARGET'].values)
        rank_corr = rank_correlation(forecast['pred'].values, forecast['TARGET'].values)
        sharpe_ew = compute_sharpe(forecast, weighting='equal')
        sharpe_vw = compute_sharpe(forecast, weighting='value')

        print(f"    R²={r2:.6f}  RankCorr={rank_corr:.4f}  "
              f"SR_EW={sharpe_ew:.4f}  SR_VW={sharpe_vw:.4f}  "
              f"n={len(forecast):,}  ({time.time()-t0:.1f}s)")

        results[model_name] = {
            'r2_oos': r2, 'rank_corr': rank_corr,
            'sharpe_ew': sharpe_ew, 'sharpe_vw': sharpe_vw,
            'n': len(forecast)
        }

        forecast.to_csv(join(market_forecast_dir, f'{model_name}_pred.csv'), index=False)

    # SSD variable importance
    if last_ensembles:
        ssd_data = df[df['DATE'] > str(valid_end + 1)]
        if len(ssd_data) == 0:
            ssd_data = df

        for model_name, ensemble in last_ensembles.items():
            ssd  = compute_ssd(ensemble, ssd_data[feature_cols].values, feature_cols)
            top5 = ssd.sort_values(ascending=False).head(5)
            print(f"    SSD {model_name.upper()}: " +
                  '  '.join(f"{f}={v:.3f}" for f, v in top5.items()))
            ssd.to_csv(join(SUMMARY_DIR, f'ssd_{model_name}_{market}.csv'),
                       header=['importance'])

    return results


# --- Main --------------------------------------------------------------------

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
                join(SUMMARY_DIR, f'local_nn_{model_name}_summary.csv'), index=False
            )

    # combined summary (append-safe)
    combined_rows = [
        {'market': market, 'model': model_name, **r}
        for market in sorted(all_results.keys())
        for model_name, r in all_results[market].items()
    ]
    if combined_rows:
        new_df        = pd.DataFrame(combined_rows)
        combined_path = join(SUMMARY_DIR, 'local_nn_training_summary.csv')

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

# this code does the pooled training for the NN models
# throughout, several outputs get stored in different folder and files
# it predicts returns, so metrics also calculated here. they stored in 'summary' folder. 
  # you can comment out the ones dont want run now, these get appended to 'summary' so it is safe to run one at a time, wont lose previously run ones



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
PARAMS_DIR   = join(your_path, 'results', 'model_parameters', 'Pooled_NN')
FORECAST_DIR = join(your_path, 'results', 'forecasts_pooled_nn')
SUMMARY_DIR  = join(your_path, 'results', 'summary')

DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
META_COLS = ['id', 'DATE', 'TARGET', 'me']

NN_EPOCH_NUM     = 100
NN_LAMBDA1_LIST  = [1e-5, 1e-4, 1e-3]
NN_LEARNING_RATE = 1e-2
NN_NUM_SEEDS     = 10
NN_PATIENCE      = 5
NN_BATCH_SIZE    = 10_000

MODELS_TO_RUN = [
    'nn1',
    'nn2',
    'nn3',
]

MODEL_ORDER = ['nn1', 'nn2', 'nn3']

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


# NN archutecturs

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

# --- Data loading & pooling

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

def get_feature_cols(df):
    exclude = META_COLS + ['market', 'TARGET_original']
    return [c for c in df.columns if c not in exclude]


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

        chunk = month_grp[['id', 'DATE', 'TARGET', 'TARGET_original',
                            'me', 'market']].copy()
        chunk['pred'] = preds
        monthly_preds.append(chunk)

    if not monthly_preds:
        return pd.DataFrame()
    return pd.concat(monthly_preds, ignore_index=True)
#annual training

def train_nn_year(add_year, model_name, df, feature_cols):
    cur_year = VALID_END + add_year

    train_data, valid_data, test_data = split_data(df, TRAIN_END, VALID_END, add_year)

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

    save_dir = join(PARAMS_DIR, model_name)
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

    print(f"    year {cur_year}: train={len(train_data):,}  valid={len(valid_data):,}  "
          f"test={len(test_data):,}  ({test_data['market'].nunique()} mkts)")
    return pred_df, best_model_list

# metrics

def oos_r2(pred, actual):
    ss_res = np.sum((actual - pred) ** 2)
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

#SSD calculation

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

# main ----

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated, e.g. nn1,nn3')
    args = parser.parse_args()

    models_to_run = (
        [m.strip().lower() for m in args.models.split(',')]
        if args.models else MODELS_TO_RUN
    )

    print(f"Device: {DEVICE}  |  Models: {models_to_run}")

    pooled             = load_and_pool_all_markets()
    pooled, dummy_cols = add_country_dummies(pooled)
    pooled             = demean_target_by_month(pooled)
    feature_cols       = get_feature_cols(pooled)

    n_test_years = END_YEAR - VALID_END
    os.makedirs(FORECAST_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR,  exist_ok=True)

    all_results    = {}
    last_ensembles = {}

    for model_name in models_to_run:
        print(f"\n{model_name.upper()}")
        t0            = time.time()
        pred_dfs      = []
        year_ensemble = None

        for add_year in range(1, n_test_years + 1):
            pred_df, ensemble = train_nn_year(add_year, model_name, pooled, feature_cols)
            if pred_df is not None:
                pred_dfs.append(pred_df)
                year_ensemble = ensemble

        if not pred_dfs:
            continue

        forecast = pd.concat(pred_dfs, ignore_index=True)
        last_ensembles[model_name] = year_ensemble

        forecast.to_csv(join(FORECAST_DIR, f'{model_name}_pred.csv'), index=False)
        print(f"  {model_name.upper()} done ({time.time()-t0:.1f}s)")

        global_result = evaluate_forecast(forecast, label="GLOBAL")
        all_results[model_name] = {'GLOBAL': global_result}

        for market in sorted(forecast['market'].unique()):
            mkt_fc = forecast[forecast['market'] == market]
            if len(mkt_fc) > 0:
                all_results[model_name][market] = evaluate_forecast(mkt_fc, label=market)

        # SSD variable importance
        if year_ensemble is not None:
            last_test = pooled[(pooled['DATE'] > str(END_YEAR)) &
                               (pooled['DATE'] <= str(END_YEAR + 1))]
            if len(last_test) == 0:
                last_test = pooled[pooled['DATE'] > str(VALID_END + 1)]

            ssd = compute_ssd(year_ensemble, last_test[feature_cols].values, feature_cols)
            print(f"    SSD top 10:")
            for feat, val in ssd.sort_values(ascending=False).head(10).items():
                print(f"      {feat:25s}  {val:.4f}")
            ssd_path = join(SUMMARY_DIR, f'ssd_{model_name}_pooled.csv')
            ssd.to_csv(ssd_path, header=['importance'])
            print(f"    Saved → {ssd_path}")

    # Global summary (append-safe)
    global_rows = [
        {'model': model_name, **all_results[model_name]['GLOBAL']}
        for model_name in models_to_run
        if model_name in all_results and all_results[model_name].get('GLOBAL')
    ]
    if global_rows:
        new_global_df = pd.DataFrame(global_rows)
        global_path   = join(SUMMARY_DIR, 'pooled_nn_global_summary.csv')

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
        combined_path = join(SUMMARY_DIR, 'pooled_nn_permarket_summary.csv')

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

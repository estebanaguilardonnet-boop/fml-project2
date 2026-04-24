# this code has NN models train on US data only. like US_US, has MODELS_TO_RUN so can comment out some models. 
# code makes predictions (which are stored) and so can calculate metrics. these get summarised in 'summary' folder. re-runing with different model will just append, wont override anything
  # well if re-run nn1 after was already done, then it will just replace old one, but if ran one at a time, you will have results on all three 
# throughout code things get saved 
# other than normal metrics, it also calculates SSD



from pathlib import Path
import pandas as pd
import numpy as np
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import random
from scipy.stats import spearmanr


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'raw_data' / 'cleenerst'
RES_DIR  = BASE_DIR / 'results'
US_CSV   = DATA_DIR / 'USA_clean.csv'

PARAMS_DIR   = RES_DIR / 'model_parameters' / 'USA'
FORECAST_DIR = RES_DIR / 'forecasts' / 'USA'
SUMMARY_DIR  = RES_DIR / 'summary'

for model in ['nn1', 'nn2', 'nn3']:
    (PARAMS_DIR / model).mkdir(parents=True, exist_ok=True)
FORECAST_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

META_COLS   = ['id', 'DATE', 'TARGET', 'me']
START_YEAR  = 1963
TRAIN_SPLIT = 1979
VALID_SPLIT = 1989
END_YEAR    = 2017

NN_EPOCH_NUM     = 100
NN_LAMBDA1_LIST  = [1e-5, 1e-4, 1e-3]
NN_LEARNING_RATE = 1e-2
NN_NUM_SEEDS     = 10
NN_PATIENCE      = 5
NN_BATCH_SIZE    = 10_000

MODELS_TO_RUN = [
    #'nn1',
    'nn2',
    'nn3',
]

MODEL_ORDER = ['nn1','nn2','nn3']

# Network architectures 
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
  
def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]

def split_data(df, add_year):
    t_end = pd.to_datetime(f'{TRAIN_SPLIT + add_year}-12-31')
    v_end = pd.to_datetime(f'{VALID_SPLIT + add_year}-12-31')
    x_end = pd.to_datetime(f'{VALID_SPLIT + add_year + 1}-12-31')

    train = df[df['DATE'] <= t_end].reset_index(drop=True)
    valid = df[(df['DATE'] > t_end) & (df['DATE'] <= v_end)].reset_index(drop=True)
    test  = df[(df['DATE'] > v_end) & (df['DATE'] <= x_end)].reset_index(drop=True)
    return train, valid, test

# monthly prediction w atchnorm
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

# training setup

def train_nn_year(data, add_year, model_name, feature_cols):
    cur_year = VALID_SPLIT + add_year
    train, valid, test = split_data(data, add_year)

    if len(train) == 0 or len(valid) == 0 or len(test) == 0:
        return None, None

    train_x = train[feature_cols].values
    train_y = train['TARGET'].values
    valid_x = valid[feature_cols].values
    valid_y = valid['TARGET'].values

    nn_class   = NN_CLASS_MAP[model_name]
    input_size = train_x.shape[1]
    batch_size = min(NN_BATCH_SIZE, len(train_x))
    h          = int(len(train_x) / batch_size) + 1

    inputs  = torch.from_numpy(train_x).float()
    targets = torch.from_numpy(train_y).float().view(-1, 1)
    val_x_t = torch.from_numpy(valid_x).float().to(DEVICE)
    val_y_t = torch.from_numpy(valid_y).float().view(-1, 1).to(DEVICE)

    save_dir        = PARAMS_DIR / model_name
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
                           str(save_dir / f'year{cur_year}_seed{seed}.pt'))

        best_model_list.append(best_model_seed)

    pred_df = nn_predict_monthly(test, best_model_list, feature_cols)
    if len(pred_df) == 0:
        return None, None

    pred_df['cur_year'] = cur_year
    print(f"  year {cur_year}: train={len(train):,}  valid={len(valid):,}  test={len(test):,}")
    return pred_df, best_model_list

def train_all_years(data, model_name, feature_cols):
    print(f"\nTraining: {model_name.upper()} | windows {VALID_SPLIT+1}–{END_YEAR}")
    results       = []
    total_start   = time.time()
    last_ensemble = None

    for add_year in range(1, END_YEAR - VALID_SPLIT + 1):
        pred, ensemble = train_nn_year(data, add_year, model_name, feature_cols)
        if pred is not None:
            results.append(pred)
            last_ensemble = ensemble

    print(f"  {model_name.upper()} complete — {(time.time()-total_start)/60:.1f} min")
    if results:
        return pd.concat(results).reset_index(drop=True), last_ensemble
    return pd.DataFrame(), None

# calculate metrics

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
    sharpe_ew = long_short_sharpe(forecast_df)
    sharpe_vw = long_short_sharpe_vw(forecast_df) if 'me' in forecast_df.columns else np.nan

    print(f"\n  {model_name.upper()}")
    print(f"  OOS R²    : {r2:.6f}")
    print(f"  Rank Corr : {rc:.4f}")
    print(f"  Sharpe EW : {sharpe_ew:.4f}")
    print(f"  Sharpe VW : {sharpe_vw:.4f}")

    return {
        'model':     model_name,
        'r2_oos':    round(r2, 8),
        'rank_corr': round(rc, 4),
        'sharpe_ew': round(sharpe_ew, 4),
        'sharpe_vw': round(sharpe_vw, 4) if not np.isnan(sharpe_vw) else np.nan,
        'n':         len(forecast_df),
    }

# ssd calc
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

# Summary
def build_nn_summary():
    full_order = ['ols-3', 'linear', 'lasso', 'ridge', 'rf', 'gbrt+h', 'lgbm',
                  'nn1', 'nn2', 'nn3']

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
    ordered_index = [m for m in full_order if m in summary_df.index]
    remaining     = [m for m in summary_df.index if m not in full_order]
    return summary_df.loc[ordered_index + remaining]


# --- Main --------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated models, e.g. nn1,nn3')
    args = parser.parse_args()

    models_to_run = (
        [m.strip().lower() for m in args.models.split(',')]
        if args.models else MODELS_TO_RUN
    )

    print(f"Device: {DEVICE}  |  Models: {models_to_run}")

    us_data      = load_and_prep(US_CSV, start_year=START_YEAR, end_year=END_YEAR)
    feature_cols = get_feature_cols(us_data)
    print(f"  Features: {len(feature_cols)}")

    last_ensembles = {}

    for model_name in models_to_run:
        forecasts, ensemble = train_all_years(us_data, model_name, feature_cols)
        if len(forecasts) > 0:
            save_path = FORECAST_DIR / f'{model_name}_pred.csv'
            forecasts.to_csv(save_path, index=False)
            print(f"  Saved → {save_path}")
            if ensemble is not None:
                last_ensembles[model_name] = ensemble

    # SSD variable importance
    last_test_end   = pd.to_datetime(f'{END_YEAR}-12-31')
    last_test_start = pd.to_datetime(f'{END_YEAR-1}-12-31')
    ssd_data = us_data[(us_data['DATE'] > last_test_start) & (us_data['DATE'] <= last_test_end)]
    if len(ssd_data) == 0:
        ssd_data = us_data[us_data['DATE'] > pd.to_datetime(f'{VALID_SPLIT+1}-12-31')]

    print("\nSSD variable importance:")
    for model_name, ensemble in last_ensembles.items():
        ssd = compute_ssd(ensemble, ssd_data[feature_cols].values, feature_cols)
        print(f"  {model_name.upper()}")
        for feat, val in ssd.sort_values(ascending=False).head(10).items():
            print(f"    {feat:25s}  {val:.4f}")
        ssd_path = SUMMARY_DIR / f'ssd_{model_name}_USA.csv'
        ssd.to_csv(ssd_path, header=['importance'])
        print(f"    Saved → {ssd_path}")

    # Full summary
    print("\nFULL SUMMARY — USA")
    summary_df = build_nn_summary()
    if len(summary_df) > 0:
        print(f"\n{summary_df.to_string()}")
        summary_path = SUMMARY_DIR / 'USA_summary.csv'
        summary_df.to_csv(summary_path)
        print(f"Saved → {summary_path}")

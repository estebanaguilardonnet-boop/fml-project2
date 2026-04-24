!pip install wrds
import gc
import numpy as np
import pandas as pd
import wrds
from scipy.stats.mstats import winsorize
from functools import partial
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MARKET = "market" #change this to specify the country data you want to download

MARKET_PERIODS = {
    "USA": ("1963-01-01", "2017-12-31", "USA"),
    "Japan": ("2008-01-01", "2017-12-31", "JPN"),
    "China": ("1999-01-01", "2017-12-31", "CHN"),
    "India": ("2007-01-01", "2017-12-31", "IND"),
    "Korea": ("1997-01-01", "2017-12-31", "KOR"),
    "Hong_Kong": ("1997-01-01", "2017-12-31", "HKG"),
    "Taiwan": ("2007-01-01", "2017-12-31", "TWN"),
    "France": ("1995-01-01", "2017-12-31", "FRA"),
    "United_Kingdom": ("2005-01-01", "2017-12-31", "GBR"),
    "Thailand": ("1997-01-01", "2017-12-31", "THA"),
    "Australia": ("2008-01-01", "2017-12-31", "AUS"),
    "Singapore": ("2007-01-01", "2017-12-31", "SGP"),
    "Sweden": ("2001-01-01", "2017-12-31", "SWE"),
    "South_Africa": ("1997-01-01", "2017-12-31", "ZAF"),
    "Poland": ("2006-01-01", "2017-12-31", "POL"),
    "Israel": ("2005-01-01", "2017-12-31", "ISR"),
    "Vietnam": ("2010-01-01", "2017-12-31", "VNM"),
    "Italy": ("2001-01-01", "2017-12-31", "ITA"),
    "Turkey": ("2006-01-01", "2017-12-31", "TUR"),
    "Switzerland": ("2002-01-01", "2017-12-31", "CHE"),
    "Indonesia": ("2005-01-01", "2017-12-31", "IDN"),
    "Greece": ("2006-01-01", "2017-12-31", "GRC"),
    "Philippines": ("2006-01-01", "2017-12-31", "PHL"),
    "Norway": ("2007-01-01", "2017-12-31", "NOR"),
    "Sri_Lanka": ("2010-01-01", "2017-12-31", "LKA"),
    "Denmark": ("2007-01-01", "2017-12-31", "DNK"),
    "Finland": ("2007-01-01", "2017-12-31", "FIN"),
    "Saudi_Arabia": ("2010-01-01", "2017-12-31", "SAU"),
    "Jordan": ("2009-01-01", "2017-12-31", "JOR"),
    "Egypt": ("2010-01-01", "2017-12-31", "EGY"),
    "Spain": ("2011-01-01", "2017-12-31", "ESP"),
    "Kuwait": ("2012-01-01", "2017-12-31", "KWT"),
}

START_DATE, END_DATE, EXCNTRY = MARKET_PERIODS[MARKET]

# Output path - adjust if running in Colab vs VSCode
OUTPUT_PATH = Path(f'/content/{MARKET}_clean.csv')
# VSCode:
# OUTPUT_PATH = Path(__file__).resolve().parents[1] / 'raw_data' / 'cleenerst' / f{MARKET}_clean.csv'

# These are the 36 characteristics + identifiers the paper uses (from SetUp.py)
# TARGET = ret_exc_lead1m (next month excess return, the thing we're predicting)
# PERMNO -> renamed to 'id' for consistency with your existing CSVs
FINAL_COLUMNS = [
    'id', 'DATE', 'TARGET', 'me',
    'mom_1', 'mvel1', 'mom_6', 'mom_12', 'chmom_6', 'maxret', 'indmom_a_12',
    'retvol', 'dolvol', 'sp', 'turn', 'bm', 'ep', 'cfp', 'bm_ia', 'cfp_ia',
    'herf', 'mve_ia', 'lev', 'pctacc', 'stddolvol', 'stdturn', 'dy', 'salecash',
    'ill', 'cashpr', 'depr', 'acc', 'absacc', 'roe', 'egr', 'agr',
    'cashdebt', 'lgr', 'sgr', 'chpmia'
]

# Columns never to normalise - TARGET must stay raw
NO_NORM_COLS = ['id', 'DATE', 'TARGET', 'excntry', 'ff49', 'me']

# ══════════════════════════════════════════════════════════════════════════════
# 1. FETCH FROM WRDS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_wrds():
    db = wrds.Connection()

    selected_columns = [
        "id", "eom", "excntry", "me", "ret_exc_lead1m",
        "taccruals_at", "nwc_gr1a", "taccruals_ni", "at_gr1", "be_gr1a",
        "debtlt_gr1a", "sale_gr1", "be_me", "fcf_me", "ni_me", "sale_me",
        "div1m_me", "ocf_debt", "at_be", "ni_be", "ret_12_1", "ret_6_1",
        "ret_1_0", "turnover_126d", "dolvol", "dolvol_var_126d",
        "turnover_var_126d", "ami_126d", "rvol_21d", "rmax1_21d",
        "sales", "prc", "ff49", "cash_conversion", "cash_me",
        "ebit_sale", "dp_gr1a"
    ]

    col_str   = ", ".join(sorted(set(selected_columns)))
    sql_query = f"""
        SELECT {col_str}
        FROM contrib.global_factor
        WHERE common=1 AND exch_main=1 AND primary_sec=1 AND obs_main=1
          AND excntry='{EXCNTRY}'
          AND eom >= '{START_DATE}' AND eom <= '{END_DATE}'
    """

    print(f"Fetching {MARKET} data from WRDS ({START_DATE} to {END_DATE})...")
    df = db.raw_sql(sql_query)
    db.close()
    print(f"  Fetched {len(df):,} rows")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. MEMORY OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

def optimize_dtypes(df):
    """Downcast numerics to save RAM - does not affect values."""
    if 'eom' in df.columns:
        df['eom'] = pd.to_datetime(df['eom'])
    for col in ['excntry', 'ff49']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    float_cols = df.select_dtypes(include=['float64']).columns
    int_cols   = df.select_dtypes(include=['int64']).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
    if len(int_cols):
        df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. WINSORISATION & RETURN CLEANING  (matches paper Section 1.1 exactly)
# ══════════════════════════════════════════════════════════════════════════════

def clean_returns(df):
    """
    Paper rules (Section 1.1):
      - Remove zero monthly returns
      - Remove returns > 300% that reverse within 1 month  (DataStream artefact)
      - Winsorise raw returns at top/bottom 2.5% per exchange per month
    Applied to monthly_return (log price return), NOT to TARGET (ret_exc_lead1m).
    TARGET is the forward-looking variable - we never touch it during cleaning.
    """
    print("Cleaning and winsorising returns...")
    df = df.reset_index(drop=True).sort_values(['id', 'eom'])

    # Compute log monthly return from price
    df['prc_lag'] = df.groupby('id')['prc'].shift(1)
    valid = df['prc'].notna() & df['prc_lag'].notna() & (df['prc_lag'] > 0)
    df['monthly_return'] = np.where(valid, np.log(df['prc'] / df['prc_lag']), np.nan)
    df.drop(columns=['prc_lag'], inplace=True)

    # Remove zeros
    df = df[df['monthly_return'] != 0]

    # Remove >300% that reverse next month
    df['ret_next'] = df.groupby('id')['monthly_return'].shift(-1)
    df = df[~((df['monthly_return'] > 3.0) & (df['ret_next'] < 0))]
    df.drop(columns=['ret_next'], inplace=True)

    # Winsorise at 2.5% per exchange per month
    def winsorise_group(x):
        if x.dropna().empty:
            return x
        return pd.Series(
            winsorize(x.dropna(), limits=(0.025, 0.025)).data,
            index=x.dropna().index
        ).reindex(x.index)

    df['monthly_return'] = (
        df.groupby(['excntry', 'eom'])['monthly_return']
        .transform(winsorise_group)
    )

    print(f"  After return cleaning: {len(df):,} rows")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. DERIVED CHARACTERISTICS  (same calculations as teammate, kept intact)
# ══════════════════════════════════════════════════════════════════════════════

def compute_characteristics(df):
    """
    Compute all derived characteristics from raw WRDS columns.
    These match the 36 variables in the paper (see SetUp.py variable list).
    """
    print("Computing derived characteristics...")
    df = df.sort_values(['id', 'eom'])

    # mvel1: log market equity
    df['mvel1'] = np.where(df['me'].notna() & (df['me'] > 0), np.log(df['me']), np.nan)

    # sp: sales-to-price
    df['sp'] = np.where(
        df['prc'].notna() & df['sales'].notna() & (df['prc'] != 0),
        df['sales'] / df['prc'], np.nan
    )

    # salecash: sales / cash
    df['cash']     = df['cash_me'] * df['me']
    df['salecash'] = np.where(
        df['sales'].notna() & df['cash'].notna() & (df['cash'] != 0),
        df['sales'] / df['cash'], np.nan
    )

    # chmom_6: change in 6-month momentum
    df['ret_6_1_lag']  = df.groupby('id')['ret_6_1'].shift(1)
    df['chmom_6']      = df['ret_6_1'] - df['ret_6_1_lag']
    df.drop(columns=['ret_6_1_lag'], inplace=True)

    # Industry-level variables (using ff49 industry classification)
    df['industry_sale'] = df.groupby(['eom', 'ff49'])['sales'].transform('sum')
    df['firm_mkt_share'] = np.where(
        df['sales'].notna() & df['industry_sale'].notna() & (df['industry_sale'] != 0),
        df['sales'] / df['industry_sale'], np.nan
    )
    df['firm_mkt_share_sq'] = df['firm_mkt_share'] ** 2
    df['herf'] = df.groupby(['eom', 'ff49'])['firm_mkt_share_sq'].transform('sum')

    # bm_ia: industry-adjusted book-to-market
    df['ind_avg_bm']  = df.groupby(['eom', 'ff49'])['be_me'].transform('mean')
    df['bm_ia']       = np.where(
        df['be_me'].notna() & df['ind_avg_bm'].notna(),
        df['be_me'] - df['ind_avg_bm'], np.nan
    )

    # cfp_ia: industry-adjusted cash-flow-to-price
    df['ind_avg_cfp'] = df.groupby(['eom', 'ff49'])['fcf_me'].transform('mean')
    df['cfp_ia']      = np.where(
        df['fcf_me'].notna() & df['ind_avg_cfp'].notna(),
        df['fcf_me'] - df['ind_avg_cfp'], np.nan
    )

    # mve_ia: industry-adjusted market equity
    df['ind_avg_me']  = df.groupby(['eom', 'ff49'])['me'].transform('mean')
    df['mve_ia']      = np.where(
        df['me'].notna() & df['ind_avg_me'].notna(),
        df['me'] - df['ind_avg_me'], np.nan
    )

    # indmom_a_12: industry momentum (avg 12m return within industry)
    df['indmom_a_12'] = df.groupby(['eom', 'ff49'])['ret_12_1'].transform('mean')

    # chpmia: industry-adjusted change in profit margin
    df['ebit_lag']             = df.groupby('id')['ebit_sale'].shift(1)
    df['chg_pm']               = np.where(
        df['ebit_sale'].notna() & df['ebit_lag'].notna(),
        df['ebit_sale'] - df['ebit_lag'], np.nan
    )
    df['ind_avg_chg_pm']       = df.groupby(['eom', 'ff49'])['chg_pm'].transform('mean')
    df['chpmia']               = np.where(
        df['chg_pm'].notna() & df['ind_avg_chg_pm'].notna(),
        df['chg_pm'] - df['ind_avg_chg_pm'], np.nan
    )
    df.drop(columns=['ebit_lag', 'ind_avg_chg_pm'], inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"  Characteristics computed. Shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. RENAME TO PAPER VARIABLE NAMES
# ══════════════════════════════════════════════════════════════════════════════

RENAME_MAP = {
    'eom':                'DATE',
    'ret_1_0':            'mom_1',
    'ret_6_1':            'mom_6',
    'ret_12_1':           'mom_12',
    'rmax1_21d':          'maxret',
    'rvol_21d':           'retvol',
    'fcf_me':             'cfp',
    'turnover_126d':      'turn',
    'be_me':              'bm',
    'ni_me':              'ep',
    'at_be':              'lev',
    'taccruals_ni':       'pctacc',
    'dolvol_var_126d':    'stddolvol',
    'turnover_var_126d':  'stdturn',
    'div1m_me':           'dy',
    'ami_126d':           'ill',
    'cash_conversion':    'cashpr',
    'dp_gr1a':            'depr',
    'nwc_gr1a':           'acc',
    'taccruals_at':       'absacc',
    'ni_be':              'roe',
    'be_gr1a':            'egr',
    'at_gr1':             'agr',
    'ocf_debt':           'cashdebt',
    'debtlt_gr1a':        'lgr',
    'sale_gr1':           'sgr',
    'ret_exc_lead1m':     'TARGET',
}

def rename_columns(df):
    df = df.rename(columns=RENAME_MAP)
    # rename 'id' stays as 'id' (teammate used 'id', paper uses 'PERMNO' - keeping 'id')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. MEDIAN IMPUTATION - by month, characteristics only, TARGET excluded
# ══════════════════════════════════════════════════════════════════════════════

def impute_by_month(df):
    """
    Fill missing characteristic values with the cross-sectional monthly median.
    Matches authors' Handle_NA(df, 'median') exactly.

    KEY FIXES vs teammate's code:
      - Done ONCE only (not twice)
      - Grouped by month so we only use information available at that point
        (no look-ahead bias from global median)
      - TARGET is fully excluded - never imputed, never touched
    """
    print("Imputing missing values with monthly cross-sectional median...")

    char_cols = [c for c in df.columns if c not in NO_NORM_COLS]

    # Group by month and fill with that month's median
    df['_date_bk'] = df['DATE']
    months         = df.groupby(pd.Grouper(key='_date_bk', freq='M'))

    # Apply median fill to characteristics only
    def fill_month(grp):
        grp[char_cols] = grp[char_cols].fillna(grp[char_cols].median())
        return grp

    df = months.apply(fill_month)
    df.drop(columns=['_date_bk'], inplace=True)
    df.sort_index(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Imputation complete. Shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7. RANK NORMALISATION - matches Rank_Norm.py exactly, TARGET excluded
# ══════════════════════════════════════════════════════════════════════════════

def rank_column(sr):
    """
    Rank-normalise a single series within a month (authors' exact formula):
      1. Rank stocks by characteristic value
      2. Subtract mean rank  -> zero-centred
      3. Divide by (N-1)/2   -> bounded in [-1, +1]
    """
    assert not sr.isnull().any(), f"NaNs found before ranking: {sr.name}"
    result  = sr.rank()
    result -= result.mean()
    result /= (len(result) - 1) / 2
    return result


def rank_all_chars(df):
    """Apply rank_column to every characteristic column in a monthly group."""
    if df.shape[0] == 0:
        return df
    # Exclude id/DATE/TARGET - rank characteristics only
    to_rank = [c for c in df.columns if c not in NO_NORM_COLS]
    for col in to_rank:
        df[col] = rank_column(df[col])
    return df


def rank_normalise(df):
    """
    Apply rank normalisation month by month.
    Mirrors authors' Rank_Norm.py: months.apply(partial(rank_all))
    Also checks the paper's minimum data requirement:
      >= 100 stocks with valid observations for >= 3 years (36 months)
    """
    print("Applying rank normalisation by month...")

    # Drop rows where TARGET is NaN before ranking
    # (paper merges TARGET back after imputing characteristics)
    df.dropna(subset=['TARGET'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    months = df.groupby(pd.Grouper(key='DATE', freq='M'))

    # Check minimum coverage requirement (paper: >=100 stocks for >=3 years)
    rows_per_month  = months.apply(len)
    valid_months    = rows_per_month[rows_per_month >= 100]
    if len(valid_months) < 36:
        raise ValueError(
            f"Only {len(valid_months)} months with >=100 stocks. "
            f"Paper requires >=36. Check your data."
        )
    start_month = valid_months.index.min().replace(day=1)
    end_month   = valid_months.index.max()
    end_month   = min(end_month, pd.to_datetime('2017-12-01'))
    print(f"  Valid data range: {start_month.date()} -> {end_month.date()}")
    print(f"  Months with >=100 stocks: {len(valid_months)}")

    # Apply rank normalisation month by month
    df_ranked = months.apply(partial(rank_all_chars))
    df_ranked.sort_index(inplace=True)

    # Filter to valid date range
    df_ranked = df_ranked[df_ranked['DATE'] >= start_month]
    df_ranked = df_ranked[df_ranked['DATE'] <= end_month]

    # Final dropna and reset
    df_ranked.dropna(inplace=True, how='any')
    df_ranked.reset_index(drop=True, inplace=True)

    print(f"  Rank normalisation complete. Final shape: {df_ranked.shape}")
    return df_ranked


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print(f"\n{'═'*60}")
    print(f"  US Data Cleaning Pipeline - corrected version")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"{'═'*60}\n")

    # -- Step 1: Fetch from WRDS ────────────────────────────────────────────
    df = fetch_wrds()
    df = optimize_dtypes(df)
    gc.collect()

    # -- Step 2: Sort early ────────────────────────────────────────────────-
    df = df.sort_values(['id', 'eom']).reset_index(drop=True)

    # -- Step 3: Clean returns (winsorise, remove bad observations) ---------
    df = clean_returns(df)
    gc.collect()

    # -- Step 4: Compute derived characteristics ───────────────────────────
    df = compute_characteristics(df)
    gc.collect()

    # -- Step 5: Rename to paper variable names ────────────────────────────
    df = rename_columns(df)
    print(f"Columns after renaming: {list(df.columns)}")

    # -- Step 6: Drop columns not needed in final output ───────────────────
    drop_cols = [
        'prc', 'sales', 'cash_me', 'ebit_sale', 'monthly_return',
        'cash', 'industry_sale', 'firm_mkt_share', 'firm_mkt_share_sq',
        'ind_avg_bm', 'ind_avg_cfp', 'ind_avg_me', 'chg_pm',
        'sale_me', 'excntry',
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    gc.collect()

    # -- Step 7: Separate TARGET from characteristics ──────────────────────
    # Paper's exact approach: impute/normalise characteristics,
    # then merge TARGET back - TARGET is never touched
    target_df = df[['id', 'DATE', 'TARGET']].copy()
    char_df   = df.drop(columns=['TARGET'])

    # -- Step 8: Impute missing characteristics by month (once only) -------
    char_df = impute_by_month(char_df)
    gc.collect()

    # -- Step 9: Merge TARGET back ────────────────────────────────────────-
    df = pd.merge(target_df, char_df, on=['id', 'DATE'])
    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"After merge + dropna: {len(df):,} rows")
    gc.collect()

    # -- Step 10: Rank normalise characteristics (TARGET excluded) ---------
    df = rank_normalise(df)
    gc.collect()

    # -- Step 11: Keep only final columns in correct order ────────────────-
    available = [c for c in FINAL_COLUMNS if c in df.columns]
    missing   = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing columns (will be absent from output): {missing}")
    df = df[available]

    # -- Step 12: Save ────────────────────────────────────────────────────-
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'═'*60}")
    print(f"  Done. Saved {len(df):,} rows × {len(df.columns)} cols")
    print(f"  -> {OUTPUT_PATH}")
    print(f"  Columns: {list(df.columns)}")
    print(f"{'═'*60}")

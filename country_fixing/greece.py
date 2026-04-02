all_market_raw_data = {}
all_market_calculated_data = {}

market = "Greece" 

csv_file_path = r'/content/Greece.csv' # Using a raw string and a typical Colab path

print(f"Loading data for {market} from {csv_file_path}...")
try:
    df_raw = pd.read_csv(csv_file_path)

    # Convert 'eom' column to datetime if it exists and is not already in datetime format
    if 'eom' in df_raw.columns:
        df_raw['eom'] = pd.to_datetime(df_raw['eom'])

    all_market_raw_data[market] = df_raw
    print("Data loading complete.")
    print(f"Loaded data shape for {market}: {all_market_raw_data[market].shape}")
    display(all_market_raw_data[market].head())
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found. Please ensure the CSV is uploaded or the path is correct.")
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")

market = "Greece"
excntry_code = MARKET_TO_EXCNTRY[market]
start_date, end_date = MARKET_PERIODS[market]

# Combine existing and new variables, removing duplicates
selected_columns = [
    "id", "eom", "excntry", "gvkey", "permno", "size_grp", "me", "ret_exc_lead1m",
    "taccruals_at", "nwc_at", "taccruals_ni", "at_gr1", "be_gr1a", "debtlt_gr1a",
    "sale_gr1", "be_me", "fcf_me", "ni_me", "sale_me", "div1m_me", "ocf_debt",
    "at_be", "ni_be", "ret_12_1", "ret_6_1", "ret_1_0", "turnover_126d",
    "dolvol", "dolvol_var_126d", "turnover_var_126d", "ami_126d", "rvol_21d",
    "rmax1_21d", "sales", "prc","ff49","cash_conversion","cash_me","naics","sic","gics",
    # New variables
    "ppeg_gr1a", "dp_gr1a", "ebit_sale"
]
unique_selected_columns = ", ".join(sorted(list(set(selected_columns))))

sql_query = f"""
    SELECT {unique_selected_columns}
    FROM contrib.global_factor
    WHERE common=1 and exch_main=1 and primary_sec=1 and obs_main=1 and
    excntry='{excntry_code}' AND eom >= '{start_date}' AND eom <= '{end_date}'
"""

print(f"Market: {market}, Ex-country Code: {excntry_code}, Period: {start_date} to {end_date}")
print("SQL Query generated.")

def perform_calculations(df):
    """Performs 'cash', 'sale_to_cash', 'log_me', 'sale_to_price', 'change_mom_6', 'industry_sale', 'firm_market_share', 'hhi', 'industry_avg_be_me', 'industry_adjusted_be_me', 'industry_avg_fcf_me', 'industry_adjusted_fcf_me', 'industry_avg_me', 'industry_adjusted_me', 'industry_ew_mom_12_1', and 'industry_adjusted_change_profit_margin' calculations on a DataFrame."""

    # Calculate cash
    # This will correctly propagate NA/NaN if cash_me or me are NA/NaN
    df['cash'] = df['cash_me'] * df['me']

    # Calculate Sale to cash
    # Explicitly check for non-NA values before applying numerical conditions
    condition_sale_to_cash = (df['sales'].notna()) & \
                             (df['me'].notna()) & \
                             (df['sales'] != 0) & \
                             (df['me'] != 0)
    df['sale_to_cash'] = np.where(
        condition_sale_to_cash,
        df['sales']/df['cash'],
        np.nan
    )

    # Calculate Log(market equity)
    # Explicitly check for non-NA values before applying numerical conditions
    condition_log_me = (df['me'].notna()) & (df['me'] > 0)
    df['log_me'] = np.where(
        condition_log_me,
        np.log(df['me']),
        np.nan
    )

    # Calculate Sale to price
    # Explicitly check for non-NA values before applying numerical conditions
    condition_sale_to_price = (df['prc'].notna()) & \
                              (df['sales'].notna()) & \
                              (df['prc'] != 0)
    df['sale_to_price'] = np.where(
        condition_sale_to_price,
        df['sales'] / df['prc'],
        np.nan
    )

    # Sort by 'id' and 'eom' to ensure correct shifting for time-series data
    df = df.sort_values(by=['id', 'eom'])

    # Calculate Change in 6-month momentum (change_mom_6)
    df['ret_6_1_prev_month'] = df.groupby('id')['ret_6_1'].shift(1)
    df['change_mom_6'] = df['ret_6_1'] - df['ret_6_1_prev_month']
    df = df.drop(columns=['ret_6_1_prev_month']) # Drop temporary column

    # Calculate Industry Sale
    df['industry_sale'] = df.groupby(['eom', 'ff49'])['sales'].transform('sum')

    # Calculate Firm Market Share
    condition_firm_market_share = (df['sales'].notna()) & \
                                  (df['industry_sale'].notna()) & \
                                  (df['industry_sale'] != 0)
    df['firm_market_share'] = np.where(
        condition_firm_market_share,
        df['sales'] / df['industry_sale'],
        np.nan
    )

    # Calculate Herfindahl-Hirschman Index (HHI)
    # Square the firm market share
    df['firm_market_share_sq'] = df['firm_market_share'] ** 2
    # Sum the squared market shares by industry and month
    df['hhi'] = df.groupby(['eom', 'ff49'])['firm_market_share_sq'].transform('sum')
    # Drop temporary column
    df = df.drop(columns=['firm_market_share_sq'])

    # Calculate Industry Average Book-to-Market (be_me)
    df['industry_avg_be_me'] = df.groupby(['eom', 'ff49'])['be_me'].transform('mean')

    # Calculate Industry-adjusted B/M
    condition_ind_adj_bm = (df['be_me'].notna()) & (df['industry_avg_be_me'].notna())
    df['industry_adjusted_be_me'] = np.where(
        condition_ind_adj_bm,
        df['be_me'] - df['industry_avg_be_me'],
        np.nan
    )

    # Calculate Industry Average CF to price ratio (fcf_me)
    df['industry_avg_fcf_me'] = df.groupby(['eom', 'ff49'])['fcf_me'].transform('mean')

    # Calculate Industry-adjusted CF to price ratio (fcf_me)
    condition_ind_adj_fcf_me = (df['fcf_me'].notna()) & (df['industry_avg_fcf_me'].notna())
    df['industry_adjusted_fcf_me'] = np.where(
        condition_ind_adj_fcf_me,
        df['fcf_me'] - df['industry_avg_fcf_me'],
        np.nan
    )

    # Calculate Industry Average Size (me)
    df['industry_avg_me'] = df.groupby(['eom', 'ff49'])['me'].transform('mean')

    # Calculate Industry-adjusted Size (me)
    condition_ind_adj_me = (df['me'].notna()) & (df['industry_avg_me'].notna())
    df['industry_adjusted_me'] = np.where(
        condition_ind_adj_me,
        df['me'] - df['industry_avg_me'],
        np.nan
    )

    # Calculate Industry 12-month equal weighted momentum (ret_12_1)
    df['industry_ew_mom_12_1'] = df.groupby(['eom', 'ff49'])['ret_12_1'].transform('mean')

    # Calculate firm specific change in profit margin (ebit_sale)
    df['ebit_sale_prev_month'] = df.groupby('id')['ebit_sale'].shift(1)
    condition_firm_change_pm = (df['ebit_sale'].notna()) & (df['ebit_sale_prev_month'].notna())
    df['firm_change_profit_margin'] = np.where(
        condition_firm_change_pm,
        df['ebit_sale'] - df['ebit_sale_prev_month'],
        np.nan
    )
    df = df.drop(columns=['ebit_sale_prev_month']) # Drop temporary column

    # Calculate industry average change in profit margin
    df['industry_avg_change_profit_margin'] = df.groupby(['eom', 'ff49'])['firm_change_profit_margin'].transform('mean')

    # Calculate Industry-adjusted change in profit margin
    condition_ind_adj_change_pm = (df['firm_change_profit_margin'].notna()) & (df['industry_avg_change_profit_margin'].notna())
    df['industry_adjusted_change_profit_margin'] = np.where(
        condition_ind_adj_change_pm,
        df['firm_change_profit_margin'] - df['industry_avg_change_profit_margin'],
        np.nan
    )

    return df

print(f"Applying calculations to {market} data...")
all_market_calculated_data[market] = perform_calculations(df_wrds.copy())
print(f"Calculations applied to {market} data.")
display(all_market_calculated_data[market].head())

output_csv_file_path = '/content/Greece_calculated_data.csv'

# Get the calculated data for the market
calculated_df = all_market_calculated_data[market]

# Save the DataFrame to a CSV file
calculated_df.to_csv(output_csv_file_path, index=False)

print(f"Calculated data for {market} saved to {output_csv_file_path}")

# You can then download this file from the Colab files pane (folder icon on the left).
# Alternatively, you can use the following code to trigger a download:
# from google.colab import files
# files.download(output_csv_file_path)

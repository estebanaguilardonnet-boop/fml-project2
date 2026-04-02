!pip install wrds
import pandas as pd
import wrds
import os
from pathlib import Path
import numpy as np

wrds_db = wrds.Connection()

MARKETS = [
    "USA",
    "Japan",
    "China",
    "India",
    "Korea",
    "Hong_Kong",
    "Taiwan",
    "France",
    "United_Kingdom",
    "Thailand",
    "Australia",
    "Singapore",
    "Sweden",
    "South_Africa",
    "Poland",
    "Israel",
    "Vietnam",
    "Italy",
    "Turkey",
    "Switzerland",
    "Indonesia",
    "Greece",
    "Philippines",
    "Norway",
    "Sri_Lanka",
    "Denmark",
    "Finland",
    "Saudi_Arabia",
    "Jordan",
    "Egypt",
    "Spain",
    "Kuwait",
]

MARKET_TO_EXCNTRY = {
    "USA": "USA",
    "Japan": "JPN",
    "China": "CHN",
    "India": "IND",
    "Korea": "KOR",
    "Hong_Kong": "HKG",
    "Taiwan": "TWN",
    "France": "FRA",
    "United_Kingdom": "GBR",
    "Thailand": "THA",
    "Australia": "AUS",
    "Singapore": "SGP",
    "Sweden": "SWE",
    "South_Africa": "ZAF",
    "Poland": "POL",
    "Israel": "ISR",
    "Vietnam": "VNM",
    "Italy": "ITA",
    "Turkey": "TUR",
    "Switzerland": "CHE",
    "Indonesia": "IDN",
    "Greece": "GRC",
    "Philippines": "PHL",
    "Norway": "NOR",
    "Sri_Lanka": "LKA",
    "Denmark": "DNK",
    "Finland": "FIN",
    "Saudi_Arabia": "SAU",
    "Jordan": "JOR",
    "Egypt": "EGY",
    "Spain": "ESP",
    "Kuwait": "KWT",
}

MARKET_PERIODS = {
    "USA": ("1963-01-01", "2017-12-31"),
    "Japan": ("2008-01-01", "2017-12-31"),
    "China": ("1999-01-01", "2017-12-31"),
    "India": ("2007-01-01", "2017-12-31"),
    "Korea": ("1997-01-01", "2017-12-31"),
    "Hong_Kong": ("1997-01-01", "2017-12-31"),
    "Taiwan": ("2007-01-01", "2017-12-31"),
    "France": ("1995-01-01", "2017-12-31"),
    "United_Kingdom": ("2005-01-01", "2017-12-31"),
    "Thailand": ("1997-01-01", "2017-12-31"),
    "Australia": ("2008-01-01", "2017-12-31"),
    "Singapore": ("2007-01-01", "2017-12-31"),
    "Sweden": ("2001-01-01", "2017-12-31"),
    "South_Africa": ("1997-01-01", "2017-12-31"),
    "Poland": ("2006-01-01", "2017-12-31"),
    "Israel": ("2005-01-01", "2017-12-31"),
    "Vietnam": ("2010-01-01", "2017-12-31"),
    "Italy": ("2001-01-01", "2017-12-31"),
    "Turkey": ("2006-01-01", "2017-12-31"),
    "Switzerland": ("2002-01-01", "2017-12-31"),
    "Indonesia": ("2005-01-01", "2017-12-31"),
    "Greece": ("2006-01-01", "2017-12-31"),
    "Philippines": ("2006-01-01", "2017-12-31"),
    "Norway": ("2007-01-01", "2017-12-31"),
    "Sri_Lanka": ("2010-01-01", "2017-12-31"),
    "Denmark": ("2007-01-01", "2017-12-31"),
    "Finland": ("2007-01-01", "2017-12-31"),
    "Saudi_Arabia": ("2010-01-01", "2017-12-31"),
    "Jordan": ("2009-01-01", "2017-12-31"),
    "Egypt": ("2010-01-01", "2017-12-31"),
    "Spain": ("2011-01-01", "2017-12-31"),
    "Kuwait": ("2012-01-01", "2017-12-31"),
}

chars = pd.read_excel('https://raw.githubusercontent.com/bkelly-lab/jkp-data/main/data/factor_details.xlsx', engine='openpyxl')
chars_rel = chars[chars['abr_jkp'].notna()]['abr_jkp'].tolist()

wrds_db.describe_table(library="contrib", table="global_factor")

market = "Denmark"
excntry_code = MARKET_TO_EXCNTRY[market]
start_date, end_date = MARKET_PERIODS[market]

# Combine existing and new variables, removing duplicates
selected_columns = [
    "id", "eom", "excntry", "gvkey", "permno", "size_grp", "me", "ret_exc_lead1m",
    "taccruals_at", "nwc_at", "taccruals_ni", "at_gr1", "be_gr1a", "debtlt_gr1a",
    "sale_gr1", "be_me", "fcf_me", "ni_me", "sale_me", "div1m_me", "ocf_debt",
    "at_be", "ni_be", "ret_12_1", "ret_6_1", "ret_1_0", "turnover_126d",
    "dolvol", "dolvol_var_126d", "turnover_var_126d", "ami_126d", "rvol_21d",
    "rmax1_21d", "sales", "prc","ff49","cash_conversion","cash_me","naics","sic","gics"
]
unique_selected_columns = ", ".join(sorted(list(set(selected_columns))))

sql_query = f"""
    SELECT {unique_selected_columns}
    FROM contrib.global_factor
    WHERE common=1 and exch_main=1 and primary_sec=1 and obs_main=1 and
    excntry='{excntry_code}' AND eom >= '{start_date}' AND eom <= '{end_date}'
"""

print(f"Fetching data for {market} ({excntry_code}) from {start_date} to {end_date}...")
data = wrds_db.raw_sql(sql_query)
print("Data fetching complete.")

data.to_csv('global_factor_denmark.csv', index=False)
print('Data successfully saved to global_factor_denmark.csv')

from google.colab import files
files.download('global_factor_denmark.csv')

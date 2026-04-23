# when the NN models were doing mkt specific training, it was calculating SSD and stored them 
# this code is just extracting those stored and creates the SSD figure for relative importance
# for each country it takes it takes the SSD info from a different NN model depending on which one had teh best VW sharpe (like the paper)
  #these were stroed in 'local_nn_training_summary.csv'
# figure just made my matching best NN model for a country and reporting back the SSD values for each characteristic and then doing the heatmap

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from os.path import join

your_path = '/Users/valentin/Desktop/FML /group'
SSD_DIR   = join(your_path, 'results', 'summary', 'SSD')
NN_CSV    = join(your_path, 'results', 'summary', 'local_nn_training_summary.csv')

# Load NN summary and pick best model per market by VW Sharpe
nn = pd.read_csv(NN_CSV)
best = nn.loc[nn.groupby('market')['sharpe_vw'].idxmax(), ['market', 'model']].reset_index(drop=True)
best.loc[len(best)] = {'market': 'USA', 'model': 'nn1'}  # hardcoded

# Paper market order (top 25 by obs, USA first)
MARKET_ORDER = [
    'USA', 'Japan', 'China', 'India', 'Korea', 'Hong_Kong', 'Taiwan',
    'France', 'United_Kingdom', 'Thailand', 'Australia', 'Singapore',
    'Sweden', 'South_Africa', 'Poland', 'Israel', 'Vietnam', 'Italy',
    'Turkey', 'Switzerland', 'Indonesia', 'Greece', 'Philippines',
    'Norway', 'Sri_Lanka'
]

# Load SSD for each market using its best model
records = {}
for _, row in best.iterrows():
    market, model = row['market'], row['model']
    if market not in MARKET_ORDER:
        continue
    path = join(SSD_DIR, f'ssd_{model}_{market}.csv')
    if not os.path.exists(path):
        print(f'Missing: {path}')
        continue
    df = pd.read_csv(path, index_col=0, header=0)
    df.columns = ['importance']
    records[market] = df['importance']

# Build matrix: markets × features, normalised per market
all_features = sorted({f for s in records.values() for f in s.index})
matrix = pd.DataFrame(index=MARKET_ORDER, columns=all_features, dtype=float)
for market in MARKET_ORDER:
    if market in records:
        s = records[market]
        s = s / s.sum()   # normalise to sum to 1
        matrix.loc[market] = s

# Order features by mean importance descending (matches paper layout)
col_order = matrix.mean(axis=0).sort_values(ascending=False).index.tolist()
matrix = matrix[col_order]

# Custom colormap: red → orange → yellow → green → teal → blue
cmap = mcolors.LinearSegmentedColormap.from_list(
    'ryw_to_blue',
    ['#8B0000', '#CC2200', '#FF4500', '#FF8C00', '#FFD700',
     '#ADFF2F', '#00CED1', '#0080FF', '#00008B'],
    N=512
)

fig, ax = plt.subplots(figsize=(18, 10))
im = ax.imshow(matrix.values.astype(float), aspect='auto', cmap=cmap,
               vmin=0, vmax=matrix.values.max())

ax.set_xticks(range(len(col_order)))
ax.set_xticklabels(col_order, rotation=90, fontsize=7)
ax.set_yticks(range(len(MARKET_ORDER)))
ax.set_yticklabels(MARKET_ORDER, fontsize=8)
ax.tick_params(left=False, bottom=False)
for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
out = join(your_path, 'results', 'summary', 'figure2_ssd_heatmap.pdf')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved → {out}')
plt.show()

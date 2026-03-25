"""
Fix portfolio_value in state files after backfill_dividends.py incorrectly
recalculated PV using entry_price instead of March 20 close prices.

Correct approach: read March 20 close values from paper_log.csv, add the
dividend amounts that were credited per strategy.
"""
import json, csv, os
os.chdir(r'C:\Users\aland\quantbot')

# Dividend amounts credited per strategy by backfill_dividends.py
DIVS_CREDITED = {
    's02_cross_asset_momentum': 41.27,
    's09_dollar_carry':         83.49,
    's10_vol_surface':          260.13,
    's11_congressional':        3.98,
    's14_gamma_wall':           268.16,
}

# Read March 20 close values from paper_log.csv
mar20_pv = {}
with open('state/paper_log.csv') as f:
    for row in csv.reader(f):
        if len(row) >= 3 and row[0] == '2026-03-20':
            mar20_pv[row[1]] = float(row[2])

print('March 20 close values from paper_log.csv:')
for sid, pv in sorted(mar20_pv.items()):
    print(f'  {sid:<35} ${pv:,.2f}')

print()

# Fix each affected state file
import glob
for fpath in sorted(glob.glob('state/*_state.json')):
    fname = os.path.basename(fpath).replace('_state.json', '')

    # Find matching strategy key in paper_log
    log_key = None
    for k in mar20_pv:
        if fname.startswith(k) or k.startswith(fname):
            log_key = k
            break
    # exact match on prefix
    for k in mar20_pv:
        if fname == k or fname.startswith(k + '_') or k.startswith(fname):
            log_key = k
            break

    if log_key not in mar20_pv:
        continue

    correct_pv = mar20_pv[log_key] + DIVS_CREDITED.get(fname, 0)

    with open(fpath) as fh:
        state = json.load(fh)

    old_pv = state.get('portfolio_value', 0)
    state['portfolio_value'] = round(correct_pv, 2)
    state['peak_value']      = max(state.get('peak_value', 0), correct_pv)

    with open(fpath, 'w') as fh:
        json.dump(state, fh, indent=2)

    div = DIVS_CREDITED.get(fname, 0)
    print(f'  {fname:<35} old=${old_pv:,.2f}  ->  correct=${correct_pv:,.2f}  (base=${mar20_pv[log_key]:,.2f} + div=${div:.2f})')

print('\nDone.')

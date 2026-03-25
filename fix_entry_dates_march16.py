"""Fix position entry_dates written as 2026-03-18 during March 16 retroactive sim → 2026-03-16."""
import json, glob, os
os.chdir(r'C:\Users\aland\quantbot')

files = glob.glob('state/*_state.json')
for f in sorted(files):
    with open(f) as fh:
        s = json.load(fh)
    changed = False
    for ticker, pos in s.get('positions', {}).items():
        if pos.get('entry_date') == '2026-03-18':
            print(f"  Fixing {os.path.basename(f)} | {ticker}: entry_date 2026-03-18 -> 2026-03-16")
            pos['entry_date'] = '2026-03-16'
            changed = True
    if changed:
        with open(f, 'w') as fh:
            json.dump(s, fh, indent=2)

print('Done.')

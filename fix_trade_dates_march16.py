"""Fix trade/log dates written as 2026-03-18 (retroactive sim) → 2026-03-16."""
import json, glob, os
os.chdir(r'C:\Users\aland\quantbot')

files = glob.glob('state/*_state.json')
for f in sorted(files):
    with open(f) as fh:
        s = json.load(fh)
    changed = False
    for t in s.get('trades', []):
        if t.get('date') == '2026-03-18':
            t['date'] = '2026-03-16'
            changed = True
    for entry in s.get('daily_log', []):
        if entry.get('date') == '2026-03-18':
            entry['date'] = '2026-03-16'
            changed = True
    if changed:
        with open(f, 'w') as fh:
            json.dump(s, fh, indent=2)
        print(f'Fixed: {f}')

print('Done.')

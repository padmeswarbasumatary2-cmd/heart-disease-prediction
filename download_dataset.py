import urllib.request
import ssl
from pathlib import Path

urls = [
    'https://raw.githubusercontent.com/plotly/datasets/master/heart.csv',
    'https://raw.githubusercontent.com/anishathalye/heart-disease/master/heart.csv',
    'https://raw.githubusercontent.com/ajay1707/Heart-Disease-Prediction/master/heart.csv',
    'https://raw.githubusercontent.com/nikbearbrown/heart-disease-uci/master/heart.csv',
    'https://raw.githubusercontent.com/Datamanim/heart/main/heart.csv',
    'https://raw.githubusercontent.com/selva86/datasets/master/Heart.csv',
    'https://people.sc.fsu.edu/~jburkardt/data/csv/heart.csv'
]

out_path = Path('data/heart.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

for url in urls:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data = resp.read()
        out_path.write_bytes(data)
        print('Downloaded', out_path.as_posix(), 'from', url)
        break
    except Exception as e:
        print('Failed to download from', url, '-', e)
else:
    print('All download attempts failed. Please provide the CSV manually.')
    raise SystemExit(1)

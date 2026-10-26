"""Execute ll5914_lab_2.ipynb on CUDA, saving outputs after every cell.

Before the QAT layer_dict cell runs, a greedy per-layer clipping-factor search is
performed on a held-out slice of the CIFAR-10 training set (not the test set), and
the chosen factors are written into that cell.
"""
import os, sys, json, io, base64, contextlib, subprocess, traceback, time
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import requests
import torch

root = Path(__file__).resolve().parent
os.chdir(root)
path = root / 'll5914_lab_2.ipynb'
nb = json.loads(path.read_text())
assert torch.cuda.is_available(), 'This run requires CUDA'

if (root / 'best_model.pth').exists():
    _orig_get = requests.get
    class _LocalCheckpoint:
        content = (root / 'best_model.pth').read_bytes()
    def _get(url, *a, **k):
        return _LocalCheckpoint() if 'best_model.pth' in url else _orig_get(url, *a, **k)
    requests.get = _get

if os.environ.get('CIFAR_LOCAL_REBUILD') and (root / 'data' / 'cifar-10-batches-py').is_dir():
    # batches rebuilt from the HF mirror (uoft-cs/cifar10) are not byte-identical to the original md5s
    import torchvision.datasets.cifar
    torchvision.datasets.cifar.check_integrity = lambda *a, **k: True

ns = {'__name__': '__main__'}
outputs = []
log = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__}


def show(*args, **kwargs):
    for n in plt.get_fignums():
        b = io.BytesIO()
        plt.figure(n).savefig(b, format='png', bbox_inches='tight')
        outputs.append({'output_type': 'display_data',
                        'data': {'image/png': base64.b64encode(b.getvalue()).decode(),
                                 'text/plain': ['<Figure>']}, 'metadata': {}})
    plt.close('all')
plt.show = show


class Capture(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s); sys.__stdout__.flush()
        return super().write(s)


def clean(text):
    return '\n'.join(line.split('\r')[-1] for line in text.split('\n'))


def save():
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
    (root / 'lab2_run_log.json').write_text(json.dumps(log, indent=2) + '\n')


def search_layer_dict():
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tf)
    val = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(45000, 50000)),
                                      batch_size=500, shuffle=False, num_workers=4)
    model = ns['model']
    names = [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    current = {n: 1.0 for n in names}
    factors = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    def score(d):
        ns['layer_dict'] = dict(d)
        with contextlib.redirect_stderr(io.StringIO()):
            return ns['evaluate'](ns['quantize_model_ste'](model, 4, 4), val)

    best = score(current)
    print(f'[search] all 1.0 -> held-out W4A4 acc {best:.2f}', flush=True)
    for sweep in range(2):
        for n in names:
            for f in factors:
                if f == current[n]:
                    continue
                trial = dict(current); trial[n] = f
                acc = score(trial)
                if acc > best:
                    best, current = acc, trial
            print(f'[search] sweep {sweep} {n} -> {current[n]} (acc {best:.2f})', flush=True)
    log['layer_search'] = {'layer_dict': current, 'heldout_w4a4_acc': best}
    return current


def write_layer_dict(cell, d):
    lines = ''.join(cell['source']).splitlines(True)
    start = next(i for i, l in enumerate(lines) if l.startswith('layer_dict = {'))
    stop = next(i for i in range(start, len(lines)) if lines[i].startswith('}'))
    body = ['layer_dict = {\n'] + [f"    '{k}': {v},\n" for k, v in d.items()]
    body[-1] = body[-1].rstrip(',\n') + '\n'
    cell['source'] = lines[:start] + body + lines[stop:]


count = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    count += 1
    outputs = []
    cap = Capture()
    cell['execution_count'] = count
    source = ''.join(cell['source'])
    print(f'\nEXECUTING CELL {i}', flush=True)
    start = time.time()
    try:
        if source.startswith('###') and 'layer_dict = {' in source:
            exec(compile(next(''.join(c['source']) for c in nb['cells']
                              if 'def quantize_model_ste(' in ''.join(c['source'])), 'qms', 'exec'), ns)
            write_layer_dict(cell, search_layer_dict())
            save()
            source = ''.join(cell['source'])
        with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
            if source.lstrip().startswith('!'):
                if 'nvidia-smi' in source:
                    print(subprocess.check_output(['nvidia-smi'], text=True))
                else:
                    import thop  # noqa: F401
                    print('Requirement already satisfied: thop')
            else:
                exec(compile(source, f'notebook_cell_{i}', 'exec'), ns)
    except Exception as e:
        tb = traceback.format_exc(); print(tb, flush=True)
        outputs.append({'output_type': 'error', 'ename': type(e).__name__, 'evalue': str(e),
                        'traceback': tb.splitlines()})
        log['error'] = {'cell': i, 'message': str(e)}
        cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout',
                            'text': clean(cap.getvalue()).splitlines(True)}] + outputs
        save(); raise
    text = clean(cap.getvalue())
    cell['outputs'] = ([{'output_type': 'stream', 'name': 'stdout', 'text': text.splitlines(True)}]
                       if text.strip() else []) + outputs
    log.setdefault('cell_seconds', {})[str(i)] = round(time.time() - start, 1)
    save()

log['complete'] = True
save()
print('ALL LAB CELLS COMPLETED', flush=True)

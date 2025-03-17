import os 
from urllib.request import urlretrieve
from pathlib import Path

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def get_fn(kind):
    version = "ccnews-small"
    return os.path.join("data", kind, f"{version}.h5"), os.path.join('data', kind, 'gt', f'{version}.h5')

def prepare(kind):
    url = DATASETS['ccnews-small'][kind]['url']
    gt_url = DATASETS['ccnews-small'][kind]['gt_url']
    fn, gt_fn = get_fn(kind)

    download(url, fn)
    download(gt_url, gt_fn)

DATASETS = {
    'ccnews-small': {
        'task1': {
            'url': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-ccnews-fp16.h5?download=true',
            'queries': lambda x: x['itest']['queries'],
            'data': lambda x: x['train'],
            'gt_url': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-ccnews-fp16.h5?download=true',
            'gt_I': lambda x: x['itest']['knns'],
        },
        'task2': {
            'url': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-ccnews-fp16.h5?download=true',
            'queries': lambda x: x['train'],
            'data': lambda x: x['train'],
            'gt_url': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/allknn-benchmark-dev-ccnews.h5?download=true',
            'gt_I': lambda x: x['knns']
        }
    }
}
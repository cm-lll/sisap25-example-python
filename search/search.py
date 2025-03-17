import argparse
import faiss
import h5py
import numpy as np
import os
from pathlib import Path
from urllib.request import urlretrieve
import time

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def get_fn(kind):
    version = "ccnews-small"
    return os.path.join("data", kind, f"{version}.h5")

def prepare(kind):
    if kind == 'task2':
        url = "https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/allknn-benchmark-dev-ccnews.h5?download=true"
    if kind == 'task1':
        url = "https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-ccnews-fp16.h5?download=true"
    fn = get_fn(kind)

    download(url, fn)

def store_results(dst, algo, kind, D, I, buildtime, querytime, params):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['data'] = kind
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()

def run(kind, params):
    print("Running", kind)

    prepare(kind)

    fn = get_fn(kind)
    f = h5py.File(fn)
    data = np.array(f['train'])
    queries = np.array(f['itest']['queries'])
    f.close()

    n, d = data.shape
    k = params['k']

    nlist = 1024 # number of clusters/centroids to build the IVF from
    index_identifier = f"IVF{nlist},SQfp16"
    index = faiss.index_factory(d, index_identifier)

    print(f"Training index on {data.shape}")
    start = time.time()
    index.train(data)
    index.add(data)
    elapsed_build = time.time() - start
    print(f"Done training in {elapsed_build}s.")
    assert index.is_trained

    for nprobe in [1, 2, 5, 10, 20, 50, 100]:
        print(f"Starting search on {queries.shape} with nprobe={nprobe}")
        start = time.time()
        index.nprobe = nprobe
        D, I = index.search(queries, k)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        I = I + 1 # FAISS is 0-indexed, groundtruth is 1-indexed

        identifier = f"index=({index_identifier}),query=(nprobe={nprobe})"

        store_results(os.path.join("result/", kind, f"{identifier}.h5"), "faissIVF", kind, D, I, elapsed_build, elapsed_search, identifier)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=['task1', 'task2'],
        default='task2'
    )


    params = {
        'task1': {
            "k": 30,
        },
        'task2': {
            "k": 15,
        }
    }

    args = parser.parse_args()

    run(args.task, params[args.task])


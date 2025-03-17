import argparse
import faiss
import h5py
import numpy as np
import os
from pathlib import Path
import time
from datasets import DATASETS, prepare, get_fn

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

    fn, _ = get_fn(kind)
    f = h5py.File(fn)
    data = np.array(DATASETS['ccnews-small'][kind]['data'](f))
    queries = np.array(DATASETS['ccnews-small'][kind]['queries'](f))
    f.close()

    n, d = data.shape
    k = params['k']

    nlist = 1024 # number of clusters/centroids to build the IVF from
    if kind == 'task1':
        index_identifier = f"IVF{nlist},SQfp16"
    elif kind == 'task2':
        index_identifier = f"IVF{nlist},PQ{d//2}x4fs"

    index = faiss.index_factory(d, index_identifier)

    print(f"Training index on {data.shape}")
    start = time.time()
    index.train(data)
    index.add(data)
    elapsed_build = time.time() - start
    print(f"Done training in {elapsed_build}s.")
    assert index.is_trained

    if kind == "task2":
        index = faiss.IndexRefineFlat(index, faiss.swig_ptr(data.astype('float32')))
        index.k_factor = 200

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

    parser.add_argument(
        '--dataset',
        choices=[
            'ccnews-small',
        ],
        default='ccnews-small'
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


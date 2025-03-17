import argparse
import h5py
import numpy as np
import os
import csv
import glob
from pathlib import Path
from datasets import DATASETS

def get_groundtruth(size="100K", private=False):
    # test
    gt_f = h5py.File(out_fn, "r")
    true_I = np.array(gt_f['knns'])
    gt_f.close()
    return true_I

def get_all_results(dirname):
    mask = [dirname + "/**/*.h5"]
    print("search for results matching:")
    print("\n".join(mask))
    for m in mask:
        for fn in glob.iglob(m):
            print(fn)
            f = h5py.File(fn, "r")
            if "knns" not in f or not ("data" in f or "data" in f.attrs):
                print("Ignoring " + fn)
                f.close()
                continue
            yield f
            f.close()

def get_recall(I, gt, k):
    assert k <= I.shape[1]
    assert len(I) == len(gt)

    n = len(I)
    recall = 0
    for i in range(n):
        recall += len(set(I[i, :k]) & set(gt[i, :k]))
    return recall / (n * k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        help='directory in which results are stored',
        default="result"
    )
    parser.add_argument(
        '--private',
        help="private queries held out for evaluation",
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--dataset',
        choices = ['ccnews-small'],
        default='ccnews-small',
    )

    parser.add_argument("csvfile")
    args = parser.parse_args()
    true_I_cache = {}


    columns = ["data", "kind", "algo", "buildtime", "querytime", "params", "recall"]

    with open(args.csvfile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for res in get_all_results(args.results):
            data = res.attrs["data"]
            d = dict(res.attrs)
            print(d)
            gt_I = np.array(DATASETS['ccnews-small'][data]['gt_I'](res))
            recall = get_recall(np.array(res["knns"]), gt_I, 10)
            d['recall'] = recall
            print(d["data"], d["algo"], d["params"], "=>", recall)
            writer.writerow(d)
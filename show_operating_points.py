import argparse
import pandas as pd

from datasets import get_query_count

# show best performing parameters exceeding threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algorithm',)
    parser.add_argument(
        '--threshold',
        default=0.9,
        help='minimum recall',
        type=float)
    parser.add_argument(
        'csv',
        metavar='CSV',
        help='input csv')
    parser.add_argument(
        '--task',
        choices=['task1', 'task2'],
    )

    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    df = df[df.task == args.task]

    if args.algorithm:
        algorithms = [args.algorithm]
    else:
        algorithms = set(df.algo.values)
    for algo in algorithms:
        print(f'show {algo}')
        if (len(df[(df.recall > args.threshold) & (df.algo == algo)].groupby(['algo', 'dataset']).min()[['querytime']])) == 0:
            print("didn't exceed recall, print highest recall:")
            print(df[(df.algo == algo)].groupby(['algo', 'dataset']).max()[['recall', 'querytime']])
    
        else:
            print(df[(df.recall > args.threshold) & (df.algo == algo)].groupby(['algo', 'dataset']).min()[['querytime']])

    print("Overview passing threshold")

    print(df[(df.recall >= args.threshold - 1e-6)][['algo', 'dataset', 'querytime', 'params']].sort_values(by=['dataset', 'algo', 'querytime']))

    print("Overview NOT passing threshold")

    print(df[(df.recall < args.threshold - 1e-6)][['algo', 'dataset', 'querytime', 'params']].sort_values(by=['dataset', 'algo', 'querytime']))
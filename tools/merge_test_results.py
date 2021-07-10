import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser('merge results')
    parser.add_argument('results', help='files to merge', nargs=argparse.REMAINDER)

    return parser.parse_args()

def main(args):
    files =args.results
    with open(files[0], 'r') as f:
        meta = [','.join(line.split(',')[:-1]) for line in f]

    scores = []
    for file in files:
        with open(file, 'r') as f:
            score = [float(line.split(',')[-1]) for line in f]
            scores.append(score)

    score_mx = np.array(scores)
    mean_score = np.mean(score_mx, axis=0)
    max_score = np.max(score_mx, axis=0)

    print(mean_score[0])
    print(max_score[0])
    print(meta[0])

    with open('detection_mean.csv', 'w') as f:
        for m, score in zip(meta, mean_score):
            f.write(f"{m},{score:.4f}\n")

    with open('detection_max.csv', 'w') as f:
        for m, score in zip(meta, max_score):
            f.write(f"{m},{score:.4f}\n")

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)
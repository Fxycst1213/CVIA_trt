"""Measure refinement on a cached replay, compensating common target motion.

The motion estimate is fitted once to raw points and shared by both methods.
These residuals measure temporal consistency, not absolute localization error.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np


def summarize(raw, refined, times, frames):
    result = {'frames': frames, 'point_pairs': len(raw)}
    if not len(raw):
        return result
    for name, values in [('raw', raw), ('refined', refined)]:
        values = np.asarray(values)
        result[name] = {
            'median_residual_px': float(np.median(values)),
            'p95_residual_px': float(np.percentile(values, 95)),
            'p99_residual_px': float(np.percentile(values, 99)),
            'jumps_above_5px': int((values > 5).sum()),
            'jumps_above_10px': int((values > 10).sum()),
        }
    result['refinement_ms'] = {'mean': float(np.mean(times)), 'p95': float(np.percentile(times, 95))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('data/refinement_eval'))
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv.open()))
    raw = np.array([[[float(r[f'raw_x{i}']), float(r[f'raw_y{i}'])] for i in range(1, 18)] for r in rows])
    refined = np.array([[[float(r[f'ref_x{i}']), float(r[f'ref_y{i}'])] for i in range(1, 18)] for r in rows])
    raw_conf = np.array([[float(r[f'raw_conf{i}']) for i in range(1, 18)] for r in rows])
    ref_conf = np.array([[float(r[f'ref_conf{i}']) for i in range(1, 18)] for r in rows])
    groups = {}
    frame_metrics = []
    image_sizes = {}
    for index, row in enumerate(rows):
        sequence = row['sequence']
        group = groups.setdefault(sequence, {'raw': [], 'refined': [], 'times': [], 'frames': 0})
        group['frames'] += 1
        group['times'].append(float(row['refine_ms']))
        if sequence not in image_sizes:
            height, width = cv2.imread(row['image']).shape[:2]
            image_sizes[sequence] = np.array([width, height])
        if index == 0 or rows[index-1]['sequence'] != sequence:
            continue
        good = (raw_conf[index] >= 0.5) & (raw_conf[index-1] >= 0.5)
        good &= np.isfinite(raw[index]).all(axis=1) & np.isfinite(raw[index-1]).all(axis=1)
        good &= (raw[index] >= 0).all(axis=1) & (raw[index] < image_sizes[sequence]).all(axis=1)
        good &= (raw[index-1] >= 0).all(axis=1) & (raw[index-1] < image_sizes[sequence]).all(axis=1)
        if good.sum() < 5:
            continue
        motion, inliers = cv2.estimateAffinePartial2D(raw[index-1, good], raw[index, good],
                                                    method=cv2.RANSAC, ransacReprojThreshold=3,
                                                    maxIters=500, confidence=0.99)
        if motion is None or inliers.sum() < max(4, int(good.sum() * 0.5)):
            continue
        # Evaluate both outputs on exactly the same valid point pairs.
        good &= (ref_conf[index] >= 0.5) & (ref_conf[index-1] >= 0.5)
        if not good.any():
            continue
        before = raw[index-1, good] @ motion[:, :2].T + motion[:, 2]
        before_ref = refined[index-1, good] @ motion[:, :2].T + motion[:, 2]
        error = np.linalg.norm(raw[index, good] - before, axis=1)
        error_ref = np.linalg.norm(refined[index, good] - before_ref, axis=1)
        group['raw'].extend(error.tolist())
        group['refined'].extend(error_ref.tolist())
        frame_metrics.append({'index': index, 'sequence': sequence,
                              'raw_p95': float(np.percentile(error, 95)),
                              'refined_p95': float(np.percentile(error_ref, 95))})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {'method': __doc__.strip(), 'total_frames': len(rows),
              'raw_valid_observations': int((raw_conf >= 0.5).sum()),
              'refined_valid_observations': int((ref_conf >= 0.5).sum()),
              'color_matches': sum(int(r['color_matches']) for r in rows),
              'flow_matches': sum(int(r['flow_matches']) for r in rows),
              'rejected_jumps': sum(int(r['rejected_jumps']) for r in rows), 'clips': {}}
    for sequence, group in groups.items():
        report['clips'][sequence] = summarize(group['raw'], group['refined'], group['times'], group['frames'])
    for split in ['tune', 'validation', 'all']:
        selected = [g for name, g in groups.items() if split == 'all' or name.endswith('_' + split)]
        report[split] = summarize([x for g in selected for x in g['raw']],
                                  [x for g in selected for x in g['refined']],
                                  [x for g in selected for x in g['times']],
                                  sum(g['frames'] for g in selected))
    (args.out_dir/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({name: report[name] for name in ['total_frames', 'raw_valid_observations',
          'refined_valid_observations', 'color_matches', 'flow_matches', 'rejected_jumps',
          'tune', 'validation', 'all']}, indent=2))
    if not args.plots:
        return
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/cvia-refinement-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(report['clips'])
    fig, ax = plt.subplots(figsize=(12, 10))
    y = np.arange(len(names))
    ax.barh(y-0.18, [report['clips'][n]['raw']['p95_residual_px'] for n in names],
            height=0.35, label='Raw network output', color='#db8e46')
    ax.barh(y+0.18, [report['clips'][n]['refined']['p95_residual_px'] for n in names],
            height=0.35, label='Refined', color='#27966e')
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace('WIN_20261016_', '').replace('_Pro', '') for n in names], fontsize=8)
    ax.set_xlabel('95th percentile residual after common motion compensation (pixels)')
    ax.set_title('Same model outputs and point pairs; smaller is smoother')
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir/'jitter_comparison.png', dpi=160)
    plt.close(fig)

    # A short visual check from the clip with the largest raw motion residual.
    sequence = max(names, key=lambda n: report['clips'][n]['raw']['p95_residual_px'])
    indices = [i for i, r in enumerate(rows) if r['sequence'] == sequence]
    from PIL import Image
    images = []
    for index in indices[::3]:
        image = cv2.imread(rows[index]['image'])
        panels = []
        for coordinates, confidence, label, color in [(raw[index], raw_conf[index], 'Raw', (255, 100, 0)),
                                                       (refined[index], ref_conf[index], 'Refined', (0, 200, 0))]:
            panel = cv2.resize(image, (640, 360))
            for key, point in enumerate(coordinates):
                if confidence[key] < 0.5 or not np.isfinite(point).all():
                    continue
                center = tuple(np.rint(point / [image.shape[1]/640, image.shape[0]/360]).astype(int))
                cv2.circle(panel, center, 3, color, 1, cv2.LINE_AA)
            cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            panels.append(panel)
        images.append(Image.fromarray(cv2.cvtColor(np.hstack(panels), cv2.COLOR_BGR2RGB)))
    if images:
        images[0].save(args.out_dir/'comparison.gif', save_all=True, append_images=images[1:],
                       duration=100, loop=0, optimize=True)
    (args.out_dir/'visualization_sequence.txt').write_text(sequence+'\n')


if __name__ == '__main__':
    main()

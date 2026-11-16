"""Scale the extracted 17 GLB markers into the aircraft body frame, in mm.

The origin stays at the GLB origin. Body +X points toward the nose, +Y toward
the aircraft's right side, and +Z downward. No center-of-gravity offset is known.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def calibrate(source, wingspan_mm, out_dir):
    if not math.isfinite(wingspan_mm) or wingspan_mm <= 0:
        raise ValueError('Wingspan must be positive and finite')
    metadata = json.loads(source.read_text())
    records = metadata['keypoints']
    if [record['id'] for record in records] != list(range(1, 18)):
        raise ValueError('Expected markers 1 through 17 in photo order')
    scale = wingspan_mm / metadata['model_wingspan']
    points = []
    for record in records:
        x_glb, y_glb, z_glb = record['xyz_glb']
        points.append([z_glb * scale, -x_glb * scale, -y_glb * scale])

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'points_body_mm.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'x_mm', 'y_mm', 'z_mm'])
        for record, point in zip(records, points):
            writer.writerow([record['id']] + [f'{value:.6f}' for value in point])
    result = {
        'source_coordinates': str(source),
        'source_glb_sha256': metadata['source_sha256'],
        'point_order_reference': metadata['point_order_reference'],
        'precision_note': metadata['precision_note'],
        'unit': 'mm',
        'measured_wingspan_mm': wingspan_mm,
        'model_wingspan': metadata['model_wingspan'],
        'scale_mm_per_model_unit': scale,
        'frame': {'origin': 'unchanged GLB origin; not a measured center of gravity',
                  'x': 'toward the nose', 'y': "toward the aircraft's right side",
                  'z': 'downward'},
        'rotation_body_from_glb': [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        'keypoints': [{'id': record['id'], 'xyz_body_mm': point}
                      for record, point in zip(records, points)],
    }
    (out_dir / 'points_body_mm.json').write_text(json.dumps(result, indent=2) + '\n')
    initializer = (
        '// Aircraft body frame: +X nose, +Y aircraft right, +Z down; units: mm.\n'
        '// Origin: unchanged GLB origin. Marker IDs follow modelPlane.png, 1..17.\n'
        f'// Wingspan: {wingspan_mm:g} mm; scale: {scale:.12f} mm per GLB unit.\n'
        '_p3d = (cv::Mat_<double>(NUM_KEYPOINTS, 3) <<\n'
    )
    for i, point in enumerate(points, 1):
        initializer += '    ' + ', '.join(f'{value:.6f}' for value in point)
        initializer += (',' if i < len(points) else ');') + f' // {i}\n'
    (out_dir / 'opencv_points_body_mm.txt').write_text(initializer)
    print(f'Scale: {scale:.12f} mm per GLB unit. Saved 17 body-frame points to {out_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/plane17/points_glb.json'))
    parser.add_argument('--wingspan-mm', type=float, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path('data/plane17'))
    args = parser.parse_args()
    calibrate(args.source, args.wingspan_mm, args.out_dir)


if __name__ == '__main__':
    main()

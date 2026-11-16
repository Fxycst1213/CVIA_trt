from pathlib import Path
import csv
import json
import numpy as np

root = Path(__file__).resolve().parents[2]
with (root / 'build/plane9_smoke/raw.csv').open() as stream:
    reader = csv.DictReader(stream)
    rows = list(reader)
    columns = reader.fieldnames
assert len(rows) == 1, f'Expected one decoded frame, got {len(rows)}'
assert len(columns) == 8 + 3 * 9, f'Unexpected output width: {len(columns)}'
row = rows[0]
confidence = float(row['confidence'])
assert confidence > .25, f'No aircraft detection: {confidence}'
points = np.array([[float(row[f'raw_{axis}{i}']) for axis in ('x', 'y', 'conf')]
                   for i in range(1, 10)])
assert np.isfinite(points).all()
assert ((points[:, :2] >= 0) & (points[:, :2] < 640)).all()
reference = json.loads((root / 'data/plane9/inference_validation.json').read_text())
view = next(r for r in reference['render_views'] if
            r['texture'] == 'texture_diffuse' and r['tilt'] == 65 and r['rotation'] == 10)
expected = np.array(view['keypoints'])
errors = np.linalg.norm(points[:, :2] - expected[:, :2], axis=1)
assert errors.max() < 5, f'TensorRT decoding differs from ONNX reference: {errors}'
engine = root / 'models/engine/x_plane1-fp16.engine'
assert engine.is_file() and engine.stat().st_size > 1024
result = {'model': 'models/onnx/x_plane1.onnx', 'precision': 'FP16',
          'engine': str(engine.relative_to(root)), 'engine_bytes': engine.stat().st_size,
          'decoded_keypoints': 9, 'detection_confidence': confidence,
          'keypoints': points.tolist(), 'onnx_reference_pixel_errors': errors.tolist(),
          'max_difference_from_onnx_reference_px': float(errors.max()),
          'note': 'Offline synthetic image test of actual C++ TensorRT decode, not real-camera pose accuracy.'}
(root / 'build/plane9_smoke/validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))

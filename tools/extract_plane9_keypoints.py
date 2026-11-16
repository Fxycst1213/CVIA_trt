"""Extract this white aircraft's nine marker centers in x_plane1.onnx order.

The fixed seed order was checked by running that ONNX on textured OBJ renders.
The central blue marker is excluded from PnP and used only as the frame origin.
The body frame is +X nose, +Y aircraft right, +Z down, derived from the OBJ axes.
Run from the repository root, e.g.:
    python3 tools/extract_plane9_keypoints.py --wingspan-mm 3445

Requires NumPy, SciPy, Pillow and OpenCV, plus the adjacent plane17 mesh helpers.
The resulting precision is limited by the source mesh and texture, not the
number of decimal places written. This is not a physical metrology result.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from extract_plane17_keypoints import render_patch, surface_point


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = {
    'models/CAD_model/base.obj': '8704651dd5047ea764df1166d3d9a6e9cd7f26124d547cf43c03d81ca9434bce',
    'models/CAD_model/texture_diffuse.png': 'bdf87e9e41854ef35257479272934e9229d7c637e50f4c854dd68f39236f8eaa',
    'models/onnx/x_plane1.onnx': '5607b1d53ed5ed18053ac649bd8a6a34c85af30e324365a5bf61e2913dd2647e',
}
BODY_FROM_OBJ = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
# Last entry is the excluded central marker. The first nine are ONNX indices 0..8.
SEEDS_XZ = np.array([
    [-0.2552277150826576, -0.5611509717427767],
    [0.06143895158400903, -0.5611509717427767],
    [0.1973363874814449, -0.5611509717427767],
    [-0.8731764330313756, -0.10474071533252031],
    [-0.6103559202108628, -0.004740715332520314],
    [-0.24497130482624738, 0.1221823615905566],
    [0.2345158746609321, 0.10295159235978737],
    [0.6114389515840091, -0.03550994610175108],
    [-0.007791817646760207, 0.5183362077444028],
    [-0.07830463815958072, 0.06833620774440276],
])
LOCATIONS = [
    '右侧尾翼点', '尾翼中间点', '左侧尾翼点',
    '右主翼外侧点', '右主翼中间点', '右主翼内侧点',
    '左主翼内侧点', '左主翼外侧点', '机头点',
]


def sha256(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def protobuf_fields(data):
    """Read protobuf wire fields to obtain ONNX metadata without an ONNX install."""
    pos = 0

    def varint():
        nonlocal pos
        value, shift = 0, 0
        while True:
            byte = data[pos]
            pos += 1
            value |= (byte & 127) << shift
            if not byte & 128:
                return value
            shift += 7

    while pos < len(data):
        key = varint()
        wire = key & 7
        if wire == 0:
            value = varint()
        elif wire in (1, 2, 5):
            size = varint() if wire == 2 else {1: 8, 5: 4}[wire]
            value = data[pos:pos + size]
            pos += size
        else:
            raise ValueError(f'Unsupported protobuf wire type: {wire}')
        yield key >> 3, value


def onnx_metadata(path):
    result = {}
    for field, value in protobuf_fields(path.read_bytes()):
        if field == 14:
            entry = dict(protobuf_fields(value))
            result[entry[1].decode()] = entry[2].decode()
    if json.loads(result.get('kpt_shape', 'null')) != [9, 3]:
        raise ValueError('Expected x_plane1.onnx with kpt_shape=[9, 3]')
    return result


def load_obj(path):
    vertices, uv, faces, face_uv = [], [], [], []
    with path.open() as stream:
        for line in stream:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('vt '):
                uv.append([float(x) for x in line.split()[1:3]])
            elif line.startswith('f '):
                corners = [item.split('/') for item in line.split()[1:]]
                if any(len(item) < 2 or not item[1] for item in corners):
                    raise ValueError('The OBJ must include UV coordinates')
                for i in range(1, len(corners) - 1):
                    faces.append([int(corners[j][0]) - 1 for j in (0, i, i + 1)])
                    face_uv.append([int(corners[j][1]) - 1 for j in (0, i, i + 1)])
    vertices, uv = np.asarray(vertices), np.asarray(uv)
    faces, face_uv = np.asarray(faces, dtype=np.int32), np.asarray(face_uv, dtype=np.int32)
    if faces.min() < 0 or face_uv.min() < 0:
        raise ValueError('This supplied OBJ is expected to use positive indices')
    return vertices, uv, faces, face_uv


def extract(vertices, uv, faces, face_uv, texture, out_dir):
    triangles = vertices[faces]
    image_uv = uv.copy()
    image_uv[:, 1] = 1 - image_uv[:, 1]  # OBJ UV origin is at image bottom left.
    triangle_uv = image_uv[face_uv]
    low = triangles[:, :, [0, 2]].min(axis=1)
    high = triangles[:, :, [0, 2]].max(axis=1)
    sheet = Image.new('RGB', (5 * 256, 2 * 280), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    records = []
    for index, seed in enumerate(SEEDS_XZ):
        half_width, size = 0.045, 240
        selected = np.flatnonzero(((high >= seed - half_width) &
                                   (low <= seed + half_width)).all(axis=1))
        local_triangles = triangles[selected]
        patch, depth, left, top, step = render_patch(
            local_triangles, triangle_uv[selected], texture, seed, half_width, size)
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = ndimage.binary_fill_holes((mask > 0) & np.isfinite(depth))
        labels, count = ndimage.label(mask)
        if count == 0:
            raise ValueError(f'No marker at seed {index}')
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        component = labels == areas.argmax()
        rows, cols = np.nonzero(component)
        x, z = left + (cols.mean() + 0.5) * step, top + (rows.mean() + 0.5) * step
        if np.linalg.norm([x - seed[0], z - seed[1]]) > 0.01:
            raise ValueError(f'Marker {index} too far from its seed')
        if not 3000 < len(rows) < 25000 or component[[0, -1], :].any() or component[:, [0, -1]].any():
            raise ValueError(f'Invalid or truncated marker region {index}')
        point, local_face, weights = surface_point(local_triangles, x, z)
        mesh_error = np.linalg.norm(point - weights @ local_triangles[local_face])
        records.append({
            'onnx_index': index if index < 9 else None,
            'display_id': index + 1 if index < 9 else None,
            'location_zh': LOCATIONS[index] if index < 9 else '中央蓝色标记，仅作原点',
            'xyz_obj': point.tolist(), 'face_index': int(selected[local_face]),
            'barycentric': weights.tolist(), 'mesh_intersection_error_obj_units': float(mesh_error),
            'marker_pixels': len(rows), 'otsu_threshold': float(threshold),
            'seed_xz': seed.tolist(), 'patch_half_width': half_width, 'patch_size': size,
        })
        tile = Image.fromarray(patch)
        marker = ImageDraw.Draw(tile)
        cx, cy = cols.mean(), rows.mean()
        marker.line((cx - 9, cy, cx + 9, cy), fill=(0, 255, 0))
        marker.line((cx, cy - 9, cx, cy + 9), fill=(0, 255, 0))
        sx, sy = index % 5 * 256, index // 5 * 280
        sheet.paste(tile, (sx + 8, sy + 32))
        label = f'index {index} / ID {index + 1}' if index < 9 else 'origin only (excluded)'
        draw.text((sx + 8, sy + 5), label, font=font, fill='black')
    sheet.save(out_dir / 'marker_center_checks.png')
    return records


def write_csv(path, columns, records, key):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['onnx_index', 'display_id', 'location_zh'] + columns)
        for record in records:
            writer.writerow([record['onnx_index'], record['display_id'], record['location_zh']] +
                            [f'{value:.6f}' for value in record[key]])


def check_geometry(points, records):
    camera = np.array([[950., 0., 640.], [0., 940., 360.], [0., 0., 1.]])
    max_pixel_error = max_rotation_error = max_translation_error = 0.
    for rotation in ([0.1, 0.2, 1.5], [0.6, -0.3, 1.2], [-0.4, 0.1, 0.8]):
        rvec = np.array(rotation, dtype=float)
        tvec = np.array([120., -70., 6500.])
        pixels = cv2.projectPoints(points, rvec, tvec, camera, None)[0]
        ok, estimated_r, estimated_t = cv2.solvePnP(points, pixels, camera, None,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError('Synthetic PnP check failed')
        projected = cv2.projectPoints(points, estimated_r, estimated_t, camera, None)[0]
        max_pixel_error = max(max_pixel_error, float(np.linalg.norm(projected - pixels, axis=2).max()))
        max_rotation_error = max(max_rotation_error, float(np.max(np.abs(
            cv2.Rodrigues(rvec)[0] - cv2.Rodrigues(estimated_r)[0]))))
        max_translation_error = max(max_translation_error, float(np.linalg.norm(estimated_t.ravel() - tvec)))
    assert max_pixel_error < 1e-5 and max_translation_error < 1e-3
    mesh_error = max(record['mesh_intersection_error_obj_units'] for record in records)
    assert mesh_error < 1e-8 and np.linalg.det(BODY_FROM_OBJ) == 1
    return {
        'mesh_intersection_max_error_obj_units': mesh_error,
        'rotation_determinant': float(np.linalg.det(BODY_FROM_OBJ)),
        'object_point_rank': int(np.linalg.matrix_rank(points - points.mean(axis=0))),
        'synthetic_pnp_poses_checked': 3,
        'synthetic_pnp_max_reprojection_error_px': max_pixel_error,
        'synthetic_pnp_max_rotation_matrix_error': max_rotation_error,
        'synthetic_pnp_max_translation_error_mm': max_translation_error,
        'note': 'Synthetic checks verify coordinates and API use; not real-camera pose accuracy.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wingspan-mm', type=float, required=True)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'data/plane9')
    args = parser.parse_args()
    if not np.isfinite(args.wingspan_mm) or args.wingspan_mm <= 0:
        parser.error('--wingspan-mm must be positive and finite')
    obj_path = ROOT / 'models/CAD_model/base.obj'
    texture_path = obj_path.with_name('texture_diffuse.png')
    onnx_path = ROOT / 'models/onnx/x_plane1.onnx'
    for relative, expected in SOURCE_SHA256.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'{relative} changed; marker seeds/order must be checked again')
    meta = onnx_metadata(onnx_path)
    vertices, uv, faces, face_uv = load_obj(obj_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = extract(vertices, uv, faces, face_uv, np.asarray(Image.open(texture_path).convert('RGB')), args.out_dir)
    origin = np.asarray(records[9]['xyz_obj'])
    span = float(np.ptp(vertices[:, 0]))
    scale = args.wingspan_mm / span
    points = (np.array([r['xyz_obj'] for r in records[:9]]) - origin) @ BODY_FROM_OBJ.T * scale
    for record, point in zip(records, points):
        record['xyz_body_mm'] = point.tolist()
    result = {
        'source_obj': str(obj_path.relative_to(ROOT)), 'source_obj_sha256': sha256(obj_path),
        'source_texture': str(texture_path.relative_to(ROOT)), 'source_texture_sha256': sha256(texture_path),
        'source_onnx': str(onnx_path.relative_to(ROOT)), 'source_onnx_sha256': sha256(onnx_path),
        'onnx_metadata': meta, 'onnx_output_shape': [1, 8400, 32],
        'onnx_indexing': 'zero-based, output[0, detection, 5+3*i:8+3*i] = (x,y,confidence)',
        'point_order_evidence': 'inference_validation.json; inference on textured renders, not embedded semantic names',
        'method': 'Otsu marker-region centroid in an orthographic top view; ray intersection with highest OBJ surface',
        'precision_note': 'Mesh/texture-derived coordinates, scaled with user-supplied wingspan; not measured 3D marker locations.',
        'unit': 'mm', 'measured_wingspan_mm': args.wingspan_mm,
        'model_wingspan_obj_units': span, 'scale_mm_per_obj_unit': scale,
        'obj_bounds_min': vertices.min(axis=0).tolist(), 'obj_bounds_max': vertices.max(axis=0).tolist(),
        'frame': {'origin': 'excluded central blue marker; not the center of gravity',
                  'x': 'toward nose (+OBJ Z)', 'y': 'aircraft right (-OBJ X)',
                  'z': 'down (-OBJ Y)', 'axes_note': 'Axis permutation of the OBJ, not independently measured mechanical axes.'},
        'origin_obj': origin.tolist(), 'rotation_body_from_obj': BODY_FROM_OBJ.tolist(),
        'transform': 'p_body_mm = scale_mm_per_obj_unit * rotation_body_from_obj @ (p_obj - origin_obj)',
        'excluded_marker': records[9], 'keypoints': records[:9],
        'verification': check_geometry(points, records),
    }
    (args.out_dir / 'points_body_mm.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    write_csv(args.out_dir / 'points_body_mm.csv', ['x_mm', 'y_mm', 'z_mm'], records[:9], 'xyz_body_mm')
    write_csv(args.out_dir / 'points_obj.csv', ['x_obj', 'y_obj', 'z_obj'], records[:9], 'xyz_obj')
    np.save(args.out_dir / 'object_points_mm.npy', points)
    initializer = ('// x_plane1.onnx indices 0..8; each row matches the same 2D keypoint index.\n'
                   '// Units: mm. Origin: excluded central blue marker (not center of gravity).\n'
                   '// +X nose, +Y aircraft right, +Z down. Wingspan: ' + f'{args.wingspan_mm:g} mm.\n'
                   '// Coordinates derived from the OBJ mesh and texture; not physical metrology.\n'
                   'const cv::Mat objectPoints = (cv::Mat_<double>(9, 3) <<\n')
    for i, point in enumerate(points):
        initializer += '    ' + ', '.join(f'{value:.6f}' for value in point)
        initializer += (',' if i < 8 else ');') + f' // index {i}, display ID {i + 1}: {LOCATIONS[i]}\n'
    (args.out_dir / 'opencv_object_points_mm.txt').write_text(initializer)
    (args.out_dir / 'geometry_validation.json').write_text(json.dumps(result['verification'], indent=2) + '\n')
    print(f'Wingspan: {span:.9f} OBJ units -> {args.wingspan_mm:g} mm; scale = {scale:.12f}')
    for i, point in enumerate(points):
        print(f'{i}: {point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}  {LOCATIONS[i]}')
    print(json.dumps(result['verification'], indent=2))


if __name__ == '__main__':
    main()

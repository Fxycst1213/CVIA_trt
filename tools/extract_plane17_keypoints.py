"""Extract the 17 textured red marker centers from base_basic_shaded.glb.

The seed order was matched to modelPlane.png. Coordinates stay in the GLB
frame: +X toward the image's right wing, +Y up, +Z toward the nose.
The reconstructed mesh and texture give approximate marker locations, not
measured physical coordinates. Supply --wingspan-mm after measuring the model.
"""

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


# Coarse X/Z surface locations, ordered by the supplied photograph's labels.
SEEDS_XZ = np.array([
    (-0.7893328, -0.0596402),
    (-0.5909023, 0.0101507),
    (-0.4340241, 0.0639680),
    (-0.2488827, 0.0216848),
    (-0.1927547, 0.1429706),
    (-0.1254792, 0.2461280),
    (-0.0469184, 0.0586038),
    (0.0067702, 0.3292460),
    (0.0034431, 0.4759568),
    (0.1293404, 0.2462071),
    (0.1800925, 0.0950632),
    (0.3847067, 0.0964192),
    (0.5730618, -0.0114535),
    (0.7967639, -0.0599419),
    (-0.2032949, -0.5018332),
    (-0.0411672, -0.5018539),
    (0.2217704, -0.5020714),
], dtype=np.float64)


def load_glb(path):
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from('<4sII', raw)
    if magic != b'glTF' or version != 2 or length != len(raw):
        raise ValueError('Expected a GLB 2.0 file')
    json_length = struct.unpack_from('<I', raw, 12)[0]
    info = json.loads(raw[20:20 + json_length])
    node = info['nodes'][0]
    if len(info['nodes']) != 1 or any(key in node for key in ('matrix', 'translation', 'rotation', 'scale')):
        raise ValueError('This extractor expects the supplied single untransformed mesh')
    binary_start = 20 + json_length + 8
    binary = memoryview(raw)[binary_start:]

    def accessor(index):
        entry = info['accessors'][index]
        view = info['bufferViews'][entry['bufferView']]
        if 'byteStride' in view or 'sparse' in entry:
            raise ValueError('Interleaved/sparse accessors are not supported here')
        dtype = {5126: '<f4', 5125: '<u4'}[entry['componentType']]
        width = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[entry['type']]
        offset = view.get('byteOffset', 0) + entry.get('byteOffset', 0)
        return np.frombuffer(binary, dtype=dtype, count=entry['count'] * width,
                             offset=offset).reshape(-1, width).copy()

    primitive = info['meshes'][node['mesh']]['primitives'][0]
    vertices = accessor(primitive['attributes']['POSITION'])
    uv = accessor(primitive['attributes']['TEXCOORD_0'])
    faces = accessor(primitive['indices']).reshape(-1, 3)
    image = info['images'][0]
    view = info['bufferViews'][image['bufferView']]
    start = view.get('byteOffset', 0)
    import io
    texture = np.asarray(Image.open(io.BytesIO(binary[start:start + view['byteLength']])).convert('RGB'))
    return vertices, uv, faces, texture


def sample_texture(texture, uv):
    height, width = texture.shape[:2]
    x = np.clip(uv[..., 0] * width - 0.5, 0, width - 1)
    y = np.clip(uv[..., 1] * height - 0.5, 0, height - 1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return ((1-fx)*(1-fy)*texture[y0, x0] + fx*(1-fy)*texture[y0, x1] +
            (1-fx)*fy*texture[y1, x0] + fx*fy*texture[y1, x1])


def render_patch(triangles, triangle_uv, texture, center, half_width=0.04, size=192):
    left, top = center - half_width
    step = 2 * half_width / size
    depth = np.full((size, size), -np.inf)
    image = np.full((size, size, 3), 230, dtype=np.uint8)
    for triangle, tex_uv in zip(triangles, triangle_uv):
        screen = (triangle[:, [0, 2]] - [left, top]) / step - 0.5
        low = np.maximum(np.ceil(screen.min(axis=0)).astype(int), 0)
        high = np.minimum(np.floor(screen.max(axis=0)).astype(int), size - 1)
        if np.any(low > high):
            continue
        a, b, c = screen
        determinant = np.cross(b-a, c-a)
        if abs(determinant) < 1e-10:
            continue
        yy, xx = np.mgrid[low[1]:high[1]+1, low[0]:high[0]+1]
        dx, dy = xx-a[0], yy-a[1]
        w1 = (dx*(c[1]-a[1]) - dy*(c[0]-a[0])) / determinant
        w2 = ((b[0]-a[0])*dy - (b[1]-a[1])*dx) / determinant
        w0 = 1-w1-w2
        y_values = w0*triangle[0, 1] + w1*triangle[1, 1] + w2*triangle[2, 1]
        visible = ((w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8) &
                   (y_values > depth[yy, xx]))
        if not visible.any():
            continue
        rows, cols = yy[visible], xx[visible]
        weights = np.stack((w0[visible], w1[visible], w2[visible]), axis=1)
        image[rows, cols] = np.rint(sample_texture(texture, weights @ tex_uv)).astype(np.uint8)
        depth[rows, cols] = y_values[visible]
    return image, depth, left, top, step


def surface_point(triangles, x, z):
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    dx1, dz1, dx2, dz2 = b[:, 0]-a[:, 0], b[:, 2]-a[:, 2], c[:, 0]-a[:, 0], c[:, 2]-a[:, 2]
    determinant = dx1*dz2-dz1*dx2
    safe = np.where(abs(determinant) > 1e-12, determinant, np.nan)
    w1 = ((x-a[:, 0])*dz2 - (z-a[:, 2])*dx2) / safe
    w2 = (dx1*(z-a[:, 2]) - dz1*(x-a[:, 0])) / safe
    weights = np.stack((1-w1-w2, w1, w2), axis=1)
    inside = np.isfinite(weights).all(axis=1) & (weights >= -1e-7).all(axis=1)
    y = (weights * triangles[:, :, 1]).sum(axis=1)
    candidates = np.flatnonzero(inside)
    if not len(candidates):
        raise ValueError('Marker center does not intersect the mesh')
    index = candidates[np.argmax(y[candidates])]
    return np.array([x, y[index], z]), int(index), weights[index]


def extract(vertices, uv, faces, texture, out_dir):
    triangles = vertices[faces]
    triangle_uv = uv[faces]
    low, high = triangles[:, :, [0, 2]].min(axis=1), triangles[:, :, [0, 2]].max(axis=1)
    records = []
    sheet = Image.new('RGB', (6*224, 3*232), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    for i, seed in enumerate(SEEDS_XZ, 1):
        selected = np.flatnonzero(((high >= seed-0.04) & (low <= seed+0.04)).all(axis=1))
        local_triangles = triangles[selected]
        patch, depth, left, top, step = render_patch(local_triangles, triangle_uv[selected], texture, seed)
        colors = patch.astype(float)
        red = ((colors[:, :, 0] > 50) & (colors[:, :, 0] > 1.4*colors[:, :, 1]) &
               (colors[:, :, 0] > 1.4*colors[:, :, 2]) & np.isfinite(depth))
        red = ndimage.binary_fill_holes(ndimage.binary_closing(red, iterations=1))
        labels, count = ndimage.label(red)
        if not count:
            raise ValueError(f'No red marker at seed {i}')
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        component = labels == areas.argmax()
        rows, cols = np.nonzero(component)
        x = left + (cols.mean()+0.5)*step
        z = top + (rows.mean()+0.5)*step
        point, local_face, weights = surface_point(local_triangles, x, z)
        if np.linalg.norm(point[[0, 2]]-seed) > 0.01:
            raise ValueError(f'Marker {i} moved too far from its visual seed')
        records.append({'id': i, 'xyz_glb': point.tolist(), 'face_index': int(selected[local_face]),
                        'barycentric': weights.tolist(), 'marker_pixels': int(component.sum())})
        tile = Image.fromarray(patch)
        marker = ImageDraw.Draw(tile)
        center = ((x-left)/step-0.5, (z-top)/step-0.5)
        marker.line((center[0]-9, center[1], center[0]+9, center[1]), fill=(0, 255, 0), width=1)
        marker.line((center[0], center[1]-9, center[0], center[1]+9), fill=(0, 255, 0), width=1)
        column, row = (i-1) % 6, (i-1) // 6
        sheet.paste(tile, (column*224+16, row*232+32))
        draw.text((column*224+16, row*232+5), f'{i}', fill='black', font=font)
        print(f'{i:2d}: {point[0]: .6f}, {point[1]: .6f}, {point[2]: .6f}', flush=True)
    sheet.save(out_dir/'marker_center_checks.png')
    return records


def save_csv(path, points):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'x', 'y', 'z'])
        for i, point in enumerate(points, 1):
            writer.writerow([i] + [f'{value:.7f}' for value in point])


def save_overview(vertices, uv, faces, texture, records, path):
    centers = vertices[faces].mean(axis=1)
    colors = np.rint(sample_texture(texture, uv[faces].mean(axis=1))).astype(np.uint8)
    width, height, margin = 1800, 1200, 60
    scale = min((width-2*margin)/np.ptp(vertices[:, 0]), (height-2*margin)/np.ptp(vertices[:, 2]))
    x_min, z_min = vertices[:, 0].min(), vertices[:, 2].min()
    px = np.rint(margin+(centers[:, 0]-x_min)*scale).astype(int)
    pz = np.rint(margin+(centers[:, 2]-z_min)*scale).astype(int)
    pixel = pz*width+px
    order = np.lexsort((-centers[:, 1], pixel))
    _, first = np.unique(pixel[order], return_index=True)
    selected = order[first]
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    canvas[pz[selected], px[selected]] = colors[selected]
    known = np.zeros((height, width), bool)
    known[pz[selected], px[selected]] = True
    distance, nearest = ndimage.distance_transform_edt(~known, return_indices=True)
    fill = (distance <= 2.5) & (~known)
    canvas[fill] = canvas[nearest[0, fill], nearest[1, fill]]
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    draw.text((20, 14), 'GLB coordinates: +X right, +Z toward nose/down, +Y up. Physical scale uncalibrated.',
              fill='black', font=font)
    for record in records:
        x, _, z = record['xyz_glb']
        sx, sy = margin+(x-x_min)*scale, margin+(z-z_min)*scale
        draw.line((sx-8, sy, sx+8, sy), fill=(0, 255, 255), width=2)
        draw.line((sx, sy-8, sx, sy+8), fill=(0, 255, 255), width=2)
        label = str(record['id'])
        box = draw.textbbox((sx+14, sy-34), label, font=font)
        draw.rectangle((box[0]-3, box[1]-2, box[2]+3, box[3]+2), fill='white')
        draw.text((sx+14, sy-34), label, fill='black', font=font)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('base_basic_shaded.glb'))
    parser.add_argument('--out-dir', type=Path, default=Path('data/plane17'))
    parser.add_argument('--wingspan-mm', type=float)
    args = parser.parse_args()
    if args.wingspan_mm is not None and (not np.isfinite(args.wingspan_mm) or args.wingspan_mm <= 0):
        parser.error('--wingspan-mm must be positive')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vertices, uv, faces, texture = load_glb(args.model)
    records = extract(vertices, uv, faces, texture, args.out_dir)
    save_overview(vertices, uv, faces, texture, records, args.out_dir/'keypoint_order_glb.png')
    points = np.array([record['xyz_glb'] for record in records])
    extent = np.ptp(vertices, axis=0)
    metadata = {'source_glb': str(args.model), 'source_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
                'method': 'top-view red texture region centroid, vertical intersection with highest mesh surface',
                'precision_note': 'Approximate reconstruction/texture coordinates; not measured marker centers.',
                'frame': {'origin': 'unchanged GLB origin', 'x': 'right in the supplied photo',
                          'y': 'up from the airframe', 'z': 'toward the nose'},
                'unit': 'unscaled GLB model units; physical size is not calibrated',
                'point_order_reference': 'modelPlane.png, labels 1 through 17',
                'bounds_min': vertices.min(axis=0).tolist(), 'bounds_max': vertices.max(axis=0).tolist(),
                'model_wingspan': float(extent[0]),
                'point_1_to_14_distance_model': float(np.linalg.norm(points[13]-points[0])),
                'scale_to_mm': None, 'keypoints': records}
    save_csv(args.out_dir/'points_glb.csv', points)
    save_csv(args.out_dir/'points_origin7_glb.csv', points-points[6])
    initializer = '// Approximate GLB model units, NOT calibrated millimeters. IDs 1 through 17.\n'
    initializer += '_p3d = (cv::Mat_<double>(17, 3) <<\n'
    initializer += ',\n'.join('    '+', '.join(f'{value:.7f}' for value in point) for point in points)+');\n'
    (args.out_dir/'opencv_points_glb.txt').write_text(initializer)
    if args.wingspan_mm is not None:
        scale = args.wingspan_mm/extent[0]
        metadata['scale_to_mm'] = float(scale)
        save_csv(args.out_dir/'points_mm.csv', points*scale)
    (args.out_dir/'points_glb.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Model wingspan: {extent[0]:.9f} model units. Physical millimeters require a measured reference.')


if __name__ == '__main__':
    main()

"""Builds the mesh of a round brilliant diamond.

The GIA brilliant is one of the three cuts in the study, and this file generates it from its proportions, so that the
exact trace and the operator share one geometry. The mesh has flat facets, because each facet must reflect a different
part of the surroundings.
"""

import numpy as np
import math


# Returns the vertices and faces of a round brilliant with flat facets.
# Angles are in degrees from the horizontal. `table_frac` is the table radius as a fraction of the girdle radius.
# `num_main_facets` is the number of main facets around the diamond, which is 8 for a standard round brilliant.
# `culet_radius` sets the small flat facet at the bottom. A value of 0 gives a perfectly sharp point instead.
def make_round_brilliant(
    girdle_radius=1.0,
    crown_angle_deg=34.5,
    pavilion_angle_deg=40.75,
    table_frac=0.56,
    num_main_facets=8,
    culet_radius=0.02,
):
    crown_angle = math.radians(crown_angle_deg)
    pavilion_angle = math.radians(pavilion_angle_deg)

    table_radius = girdle_radius * table_frac
    crown_height = (girdle_radius - table_radius) * math.tan(crown_angle)
    pavilion_height = girdle_radius * math.tan(pavilion_angle)

    n = num_main_facets
    n2 = 2 * n                # Points on the girdle ring

    verts = []

    def add(p):
        verts.append(p)
        return len(verts) - 1

    # The table is a flat polygon, fanned from its own centre.
    table_center = add([0.0, 0.0, crown_height])
    table_ring = []
    for i in range(n):
        ang = 2 * math.pi * i / n
        table_ring.append(add([
            table_radius * math.cos(ang),
            table_radius * math.sin(ang),
            crown_height,
        ]))

    # The girdle ring interleaves two sets of points. An even index 2*i sits under table vertex i. An odd index 2*i+1
    # sits under the midpoint of table vertices i and i+1.
    girdle_ring = []
    for k in range(n2):
        ang = 2 * math.pi * k / n2
        girdle_ring.append(add([
            girdle_radius * math.cos(ang),
            girdle_radius * math.sin(ang),
            0.0,
        ]))

    # The culet is either a single point or a small flat polygon.
    if culet_radius <= 1e-9:
        culet_center = add([0.0, 0.0, -pavilion_height])
        culet_ring = None
    else:
        culet_center = add([0.0, 0.0, -pavilion_height])
        culet_ring = []
        # The flat polygon sits just above the deepest point, in line with the even girdle points.
        culet_z = -pavilion_height + culet_radius * math.tan(pavilion_angle)
        for i in range(n):
            ang = 2 * math.pi * i / n
            culet_ring.append(add([
                culet_radius * math.cos(ang),
                culet_radius * math.sin(ang),
                culet_z,
            ]))

    faces = []

    def tri(a, b, c):
        faces.append([a, b, c])

    # The star facets join a table edge to the odd girdle point directly below it.
    for i in range(n):
        t_a = table_ring[i]
        t_b = table_ring[(i + 1) % n]
        g_mid = girdle_ring[(2 * i + 1) % n2]
        tri(t_b, t_a, g_mid)

    # The bezel facets are kites. Each joins table vertex i to the even girdle point beneath it and the two beside it.
    for i in range(n):
        t_v = table_ring[i]
        g_apex = girdle_ring[2 * i]
        g_left = girdle_ring[(2 * i - 1) % n2]
        g_right = girdle_ring[(2 * i + 1) % n2]
        tri(t_v, g_left, g_apex)
        tri(t_v, g_apex, g_right)

    # The table cap closes the flat top of the diamond.
    for i in range(n):
        tri(table_center, table_ring[i], table_ring[(i + 1) % n])

    # Every girdle edge must belong to exactly one pavilion triangle. Two copies of a triangle would coincide, so a ray
    # that refracts through one would hit the other at once and refract straight back out. The diamond would then show
    # the background through those wedges, as flat grey patches seen through the crown.
    if culet_ring is None:
        # With a sharp culet, the pavilion is a closed fan from the culet point to the girdle ring. This gives two
        # triangles per main facet, which is exactly one per girdle edge. The lower-girdle facets merge into this fan,
        # so they need no triangles of their own.
        for i in range(n):
            g_even = girdle_ring[2 * i]
            g_odd_next = girdle_ring[(2 * i + 1) % n2]
            g_odd_prev = girdle_ring[(2 * i - 1) % n2]

            # Each main pavilion facet is a kite from an even girdle point down to the culet, beside two odd points.
            tri(g_even, g_odd_prev, culet_center)
            tri(g_even, culet_center, g_odd_next)
    else:
        # With a flat culet, the pavilion is a band between the girdle ring and the culet ring. Three triangles per main
        # direction cover the two girdle edges after `g_even` and the culet edge between `c_this` and `c_next`.
        for i in range(n):
            g_even = girdle_ring[2 * i]
            g_odd_next = girdle_ring[(2 * i + 1) % n2]
            g_even_next = girdle_ring[(2 * i + 2) % n2]
            c_this = culet_ring[i]
            c_next = culet_ring[(i + 1) % n]

            # These are the halves of the main pavilion facets. Each girdle edge drops to its nearest culet point.
            tri(g_even, c_this, g_odd_next)
            tri(g_even_next, g_odd_next, c_next)

            # This is the lower-girdle facet, where the odd girdle point spans the culet edge.
            tri(g_odd_next, c_this, c_next)

        # The culet cap is wound the opposite way from the table cap, so that its normal points down and out of the
        # diamond.
        for i in range(n):
            tri(culet_center, culet_ring[(i + 1) % n], culet_ring[i])

    vertices = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.uint32)
    return vertices, faces


# Returns a flat-shaded copy of a mesh, with vertices, normals, UVs, and faces. Each face gets its own vertices with the
# face normal, which keeps facet edges sharp. Smoothed normals would blur the facets and lose the sparkle.
def make_flat_shaded(vertices, faces):
    flat_vertices = []
    flat_normals = []
    flat_uvs = []

    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
        else:
            normal = np.array([0.0, 0.0, 1.0])

        for v, uv in zip([v0, v1, v2], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]):
            flat_vertices.append(v)
            flat_normals.append(normal)
            flat_uvs.append(uv)

    flat_vertices = np.array(flat_vertices, dtype=np.float32)
    flat_normals = np.array(flat_normals, dtype=np.float32)
    flat_uvs = np.array(flat_uvs, dtype=np.float32)

    num_faces = len(faces)
    flat_faces = np.array(
        [[i * 3, i * 3 + 1, i * 3 + 2] for i in range(num_faces)],
        dtype=np.uint32,
    )

    return flat_vertices, flat_normals, flat_uvs, flat_faces


# Prints the number of faces, the number of degenerate faces, and the largest triangle area, to check a mesh by eye.
def _check_mesh(vertices, faces, label=""):
    degenerate = 0
    max_area = 0.0
    for tri_idx in faces:
        a, b, c = vertices[tri_idx[0]], vertices[tri_idx[1]], vertices[tri_idx[2]]
        area = np.linalg.norm(np.cross(b - a, c - a))
        max_area = max(max_area, area)
        if area < 1e-8:
            degenerate += 1
    print(f"[check{(' ' + label) if label else ''}] "
          f"faces={len(faces)} degenerate={degenerate} max_tri_area={max_area:.4f}")


if __name__ == "__main__":
    print("Testing brilliant geometry generation...")

    verts, faces = make_round_brilliant(culet_radius=0.0)
    print(f"Sharp-culet: {len(verts)} vertices, {len(faces)} faces")
    _check_mesh(verts, faces, "sharp-culet")
    print("bbox min:", verts.min(axis=0), " bbox max:", verts.max(axis=0))

    verts2, faces2 = make_round_brilliant(culet_radius=0.02)
    print(f"Flat-culet:  {len(verts2)} vertices, {len(faces2)} faces")
    _check_mesh(verts2, faces2, "flat-culet")

    flat_verts, flat_norms, flat_uvs, flat_faces = make_flat_shaded(verts, faces)
    print(f"Flat-shaded: {len(flat_verts)} vertices, {len(flat_faces)} faces")
"""Builds the mesh of a pear brilliant diamond, with the same conventions as the round brilliant.

The pear is one of the three cuts in the study, so that results can be compared across different outlines. The girdle
uses a teardrop curve with a point at one end and a round bowl at the other, and raising `taper` sharpens the point.
The girdle is flat at z=0, the table is flat, and the pavilion meets at a single apex.
"""
import numpy as np


# Returns the points of the girdle outline, normalised so that the largest radius is 1.
def pear_outline(count, taper=1.0):
    # The samples skip t = 0, because a vertex on the point of the curve would give two coincident edges.
    t = np.linspace(0., 2.*np.pi, count, endpoint=False) + np.pi/count
    x = np.cos(t)
    y = np.sin(t)*np.abs(np.sin(t/2.))**taper
    points = np.stack([x, y], axis=1)
    return points/np.linalg.norm(points, axis=1).max()


# Returns the vertices and faces of a watertight pear brilliant. `num_girdle_points` sets how finely the outline is
# sampled, and therefore the number of triangles, as `num_main_facets` does for the round brilliant.
def make_pear_brilliant(girdle_radius=1.0, crown_angle_deg=34.5,
                        pavilion_angle_deg=40.75, table_frac=0.56,
                        num_girdle_points=16, taper=1.0):
    if num_girdle_points < 6:
        raise ValueError('num_girdle_points must be at least six')
    if not 0. < table_frac < 1.:
        raise ValueError('table_frac must lie strictly between zero and one')
    if taper <= 0.:
        raise ValueError('taper must be positive')

    outline = pear_outline(num_girdle_points, taper)*girdle_radius
    centroid = outline.mean(axis=0)
    mean_radius = float(np.linalg.norm(outline-centroid, axis=1).mean())

    # The girdle and table are flat, as in a real cut. Heights use the mean radius, so that the crown and pavilion
    # angles hold on average around the diamond.
    crown_height = mean_radius*(1.-table_frac)*np.tan(np.radians(crown_angle_deg))
    pavilion_depth = mean_radius*np.tan(np.radians(pavilion_angle_deg))

    girdle = np.column_stack([outline, np.zeros(num_girdle_points)])
    table_xy = centroid + table_frac*(outline-centroid)
    table = np.column_stack([table_xy, np.full(num_girdle_points, crown_height)])
    apex = np.array([centroid[0], centroid[1], -pavilion_depth])

    vertices = np.vstack([girdle, table, apex]).astype(np.float32)
    girdle_base, table_base, apex_index = 0, num_girdle_points, 2*num_girdle_points

    faces = []
    for i in range(num_girdle_points):
        j = (i+1) % num_girdle_points
        g0, g1 = girdle_base+i, girdle_base+j
        t0, t1 = table_base+i, table_base+j
        faces.append([g0, g1, t1])          # Crown band
        faces.append([g0, t1, t0])
        faces.append([g1, g0, apex_index])  # Pavilion
    for i in range(1, num_girdle_points-1):  # Table cap, as a fan
        faces.append([table_base, table_base+i, table_base+i+1])
    faces = np.array(faces, dtype=np.uint32)

    # Every triangle is turned to face outward from the centre, so that refraction enters and leaves correctly.
    centre = vertices.mean(axis=0)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    inward = (normals*(triangles.mean(axis=1)-centre)).sum(axis=1) < 0
    faces[inward] = faces[inward][:, ::-1]
    return vertices, faces

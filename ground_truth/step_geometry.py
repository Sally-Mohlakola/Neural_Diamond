"""Builds the mesh of a step cut diamond, with the same conventions as the round brilliant.

The step cut is one of the three cuts in the study, so that results can be compared across different layouts of
facets. It has an octagonal outline, concentric steps on the crown and pavilion, and a small flat culet.
"""
import numpy as np


# Returns the octagonal girdle outline, normalised so that the largest radius is 1. It is a rectangle with half-extents
# of `length_ratio` and 1, with each corner cut by `corner_frac` of the half-extent. The points run counter-clockwise
# from the +x side, so that their order matches the other generators.
def step_outline(length_ratio=1.35, corner_frac=0.20):
    if length_ratio < 1.:
        raise ValueError('length_ratio must be at least one')
    if not 0. < corner_frac < 0.5:
        raise ValueError('corner_frac must lie strictly between zero and one half')
    a, b = float(length_ratio), 1.
    ta, tb = corner_frac*a, corner_frac*b
    points = np.array([[a, b-tb], [a-ta, b], [-(a-ta), b], [-a, b-tb],
                       [-a, -(b-tb)], [-(a-ta), -b], [a-ta, -b], [a, -(b-tb)]])
    return points/np.linalg.norm(points, axis=1).max()


# Returns the facet angle of each step, spread evenly around the nominal angle.
def _band_angles(nominal_deg, count, spread, steepest_first):
    if count == 1:
        return [np.radians(nominal_deg)]
    fraction = (np.arange(count)+.5)/count
    factor = 1.+spread*(1.-2.*fraction) if steepest_first else 1.+spread*(2.*fraction-1.)
    return [np.radians(nominal_deg*f) for f in factor]


# Returns the height of each step, rescaled so that the last one lands exactly on `target`.
def _cumulative(scales, angles, mean_radius, target):
    increments = [mean_radius*(scales[k]-scales[k+1])*np.tan(angles[k])
                  for k in range(len(angles))]
    total = float(sum(increments))
    if total <= 0.:
        raise ValueError('Degenerate step run')
    factor = target/total
    heights, running = [], 0.
    for increment in increments:
        running += increment*factor
        heights.append(running)
    return heights


# Returns the vertices and faces of a watertight step cut.
# `crown_steps` and `pavilion_steps` set the number of triangles. The mesh has 8*(crown_steps+pavilion_steps+1)+2
# vertices and 16*(crown_steps+pavilion_steps)+16 faces.
# `culet_frac` is the width of the culet as a fraction of the outline. It cannot be 0, because a bottom ring that
# collapses to a point would change the number of faces above.
def make_step_cut(girdle_radius=1.0, crown_angle_deg=34.5, pavilion_angle_deg=40.75,
                  table_frac=0.56, length_ratio=1.35, corner_frac=0.20,
                  crown_steps=3, pavilion_steps=3, culet_frac=0.15,
                  step_angle_spread=0.06):
    if crown_steps < 1 or pavilion_steps < 1:
        raise ValueError('crown_steps and pavilion_steps must be at least one')
    if not 0. < table_frac < 1.:
        raise ValueError('table_frac must lie strictly between zero and one')
    if not 0. < culet_frac < 1.:
        raise ValueError('culet_frac must lie strictly between zero and one')
    if step_angle_spread < 0. or step_angle_spread >= 1.:
        raise ValueError('step_angle_spread must lie in [0, 1)')

    outline = step_outline(length_ratio, corner_frac)*girdle_radius
    sides = len(outline)
    centroid = outline.mean(axis=0)
    mean_radius = float(np.linalg.norm(outline-centroid, axis=1).mean())
    crown_height = mean_radius*(1.-table_frac)*np.tan(np.radians(crown_angle_deg))
    pavilion_depth = mean_radius*np.tan(np.radians(pavilion_angle_deg))

    # The crown runs from the girdle up to the table, with its steepest step at the girdle.
    crown_scales = np.linspace(1., table_frac, crown_steps+1)
    crown_heights = _cumulative(crown_scales,
                                _band_angles(crown_angle_deg, crown_steps,
                                             step_angle_spread, True),
                                mean_radius, crown_height)
    # The pavilion runs from the girdle down to the culet ring, with its steepest step at the culet.
    pavilion_scales = np.linspace(1., culet_frac, pavilion_steps+1)
    pavilion_heights = _cumulative(pavilion_scales,
                                   _band_angles(pavilion_angle_deg, pavilion_steps,
                                                step_angle_spread, False),
                                   mean_radius, pavilion_depth)

    def ring(scale, height):
        return np.column_stack([centroid+scale*(outline-centroid),
                                np.full(sides, height)])

    # The rings run from top to bottom. They are the table, the crown steps, the girdle, and the pavilion steps. The
    # last pavilion ring is the culet.
    rings = [ring(table_frac, crown_height)]
    for k in range(crown_steps-1, 0, -1):
        rings.append(ring(crown_scales[k], crown_heights[k-1]))
    rings.append(ring(1., 0.))
    for i in range(pavilion_steps):
        rings.append(ring(pavilion_scales[i+1], -pavilion_heights[i]))

    # The table and culet each get a centre vertex, so that they are fans, as in the round brilliant.
    table_centre = np.array([[centroid[0], centroid[1], crown_height]])
    culet_centre = np.array([[centroid[0], centroid[1], -pavilion_depth]])
    vertices = np.vstack(rings+[table_centre, culet_centre]).astype(np.float32)
    culet_base = (len(rings)-1)*sides
    table_apex, culet_apex = len(rings)*sides, len(rings)*sides+1

    faces = []
    for i in range(sides):                          # Table cap
        faces.append([table_apex, i, (i+1) % sides])
    for r in range(len(rings)-1):                   # Step bands
        upper, lower = r*sides, (r+1)*sides
        for i in range(sides):
            j = (i+1) % sides
            faces.append([upper+i, upper+j, lower+j])
            faces.append([upper+i, lower+j, lower+i])
    for i in range(sides):                          # Culet cap
        faces.append([culet_apex, culet_base+i, culet_base+(i+1) % sides])
    faces = np.array(faces, dtype=np.uint32)

    # Every triangle is turned to face outward from the centre, so that refraction enters and leaves correctly.
    centre = vertices.mean(axis=0)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    inward = (normals*(triangles.mean(axis=1)-centre)).sum(axis=1) < 0
    faces[inward] = faces[inward][:, ::-1]
    return vertices, faces

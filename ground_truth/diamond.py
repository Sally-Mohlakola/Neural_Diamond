"""Renders a rotating round brilliant with the exact path tracer, as a standalone animation.

This script shows what the exact transport looks like, without any network. Nothing imports it. However, the renderer's
side view follows its line of sight, and its roll sets the motion of the animations. The mesh is flat-shaded, so that
each facet keeps its own normal. The material is dispersive, and the studio lighting gives the facets something
structured to refract and reflect.
"""

import mitsuba as mi
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from brilliant_geometry import make_round_brilliant, make_flat_shaded

# The variant is spectral because dispersion produces a diamond's fire, and an RGB renderer has no wavelengths to split.
mi.set_variant("scalar_spectral")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bsdf.dispersive_dielectric import DispersiveDielectric
from utils.studio_env import studio_lighting, display_exposure

# The mesh of the diamond.
raw_verts, raw_faces = make_round_brilliant(
    girdle_radius=1.0,
    crown_angle_deg=34.5,
    pavilion_angle_deg=40.75,
    table_frac=0.56,
)
verts, normals, uvs, faces = make_flat_shaded(raw_verts, raw_faces)

print(f"[mesh] flat-shaded vertices : {len(verts)}")
print(f"[mesh] faces                : {len(faces)}")

mesh = mi.Mesh(
    "diamond_mesh",
    vertex_count=len(verts),
    face_count=len(faces),
    has_vertex_normals=True,
    has_vertex_texcoords=True,
)
mesh_params = mi.traverse(mesh)
mesh_params["vertex_positions"] = mi.ArrayXf(verts.flatten())
mesh_params["vertex_normals"]   = mi.ArrayXf(normals.flatten())
mesh_params["vertex_texcoords"] = mi.ArrayXf(uvs.flatten())
mesh_params["faces"]            = mi.ArrayXu(faces.flatten())
mesh_params.update()

# The material of the diamond.
diamond_bsdf = {
    # The index varies with wavelength, following the Sellmeier fit in `bsdf/dispersion.py`. Blue therefore refracts
    # more strongly than red, and the diamond shows coloured flashes. Mitsuba's own `dielectric` has one index and
    # stays colourless, so it cannot show fire.
    "type": "dispersive_dielectric",
    "ext_ior": 1.000277,  # Air
    "dispersion": True,
}
mesh.set_bsdf(mi.load_dict(diamond_bsdf))

# The scene holds the camera, the ground plane, and the lights.
ground_bsdf = {
    "type": "diffuse",
    "reflectance": {"type": "rgb", "value": [0.12, 0.12, 0.14]},
}

# The scene without the diamond, which is added for each frame.
base_scene_dict = {
    "type": "scene",

    "integrator": {
        "type": "path",
        # A low depth cuts off paths that a diamond needs, and these show up as black facets.
        "max_depth": 64,
    },

    "sensor": {
        "type": "perspective",
        "fov": 32,
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[0.0, -4.6, 2.4],
            target=[0.0,  0.0, -0.05],
            up    =[0.0,  0.0, 1.0],
        ),
        "film": {
            "type": "hdrfilm",
            "width":  768,
            "height": 768,
            "rfilter": {"type": "gaussian"},
            "pixel_format": "rgb",
        },
        "sampler": {
            "type": "independent",
            "sample_count": 256,
        },
    },

    # The lighting comes from `utils/studio_env.py`, which the renderer also uses, so that all renders are lit
    # identically. A constant environment would not work. Every path leaving the diamond would return the same
    # radiance whatever its route, and the facets would render as flat grey.
    **studio_lighting(),

    "ground": {
        "type": "rectangle",
        "to_world": mi.ScalarTransform4f.translate([0, 0, -0.86]).scale([6, 6, 1]),
        "bsdf": ground_bsdf,
    },

}

# Renders one frame, with the diamond turned about `rotation_axis` by its share of a full turn.
def render_frame(frame_num, total_frames, rotation_axis='y'):
    # The frames span one full turn.
    angle = (frame_num / total_frames) * 2 * np.pi

    if rotation_axis.lower() == 'x':
        rotation = mi.ScalarTransform4f.rotate([1, 0, 0], np.degrees(angle))
    elif rotation_axis.lower() == 'y':
        rotation = mi.ScalarTransform4f.rotate([0, 1, 0], np.degrees(angle))
    elif rotation_axis.lower() == 'z':
        rotation = mi.ScalarTransform4f.rotate([0, 0, 1], np.degrees(angle))
    else:
        rotation = mi.ScalarTransform4f.rotate([0, 1, 0], np.degrees(angle))

    # The rotated copy of the mesh.
    rotated_mesh = mi.Mesh(
        "diamond_mesh",
        vertex_count=len(verts),
        face_count=len(faces),
        has_vertex_normals=True,
        has_vertex_texcoords=True,
    )

    # Rotates the vertices.
    rotated_verts = []
    for v in verts:
        x, y, z = v
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        if rotation_axis.lower() == 'y':
            rx = x * cos_a + z * sin_a
            ry = y
            rz = -x * sin_a + z * cos_a
        elif rotation_axis.lower() == 'x':
            rx = x
            ry = y * cos_a - z * sin_a
            rz = y * sin_a + z * cos_a
        elif rotation_axis.lower() == 'z':
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            rz = z
        else:
            rx, ry, rz = x, y, z
        rotated_verts.append([rx, ry, rz])

    rotated_verts = np.array(rotated_verts, dtype=np.float32)

    # Rotates the normals in the same way.
    rotated_normals = []
    for n in normals:
        nx, ny, nz = n
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        if rotation_axis.lower() == 'y':
            rnx = nx * cos_a + nz * sin_a
            rny = ny
            rnz = -nx * sin_a + nz * cos_a
        elif rotation_axis.lower() == 'x':
            rnx = nx
            rny = ny * cos_a - nz * sin_a
            rnz = ny * sin_a + nz * cos_a
        elif rotation_axis.lower() == 'z':
            rnx = nx * cos_a - ny * sin_a
            rny = nx * sin_a + ny * cos_a
            rnz = nz
        else:
            rnx, rny, rnz = nx, ny, nz
        rotated_normals.append([rnx, rny, rnz])

    rotated_normals = np.array(rotated_normals, dtype=np.float32)

    # Fills the rotated mesh.
    mesh_params_rot = mi.traverse(rotated_mesh)
    mesh_params_rot["vertex_positions"] = mi.ArrayXf(rotated_verts.flatten())
    mesh_params_rot["vertex_normals"]   = mi.ArrayXf(rotated_normals.flatten())
    mesh_params_rot["vertex_texcoords"] = mi.ArrayXf(uvs.flatten())
    mesh_params_rot["faces"]            = mi.ArrayXu(faces.flatten())
    mesh_params_rot.update()

    rotated_mesh.set_bsdf(mi.load_dict(diamond_bsdf))

    scene_dict = dict(base_scene_dict)
    scene_dict["mesh"] = rotated_mesh

    scene = mi.load_dict(scene_dict)

    image = mi.render(scene, spp=256)

    return image

# Renders every frame of the animation to `output_dir`, and then makes a video and a GIF if `make_video` is set.
# With 36 frames, the diamond turns 10 degrees per frame.
def create_rotation_animation(
    total_frames=36,
    rotation_axis='y',
    output_dir="diamond_animation",
    make_video=True
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[animation] Generating {total_frames} frames...")
    print(f"[animation] Rotation axis: {rotation_axis}")
    print(f"[animation] Output directory: {output_dir}")

    frames = []
    start_time = time.time()

    for i in range(total_frames):
        frame_start = time.time()
        print(f"  Rendering frame {i+1}/{total_frames}...", end="", flush=True)

        image = render_frame(i, total_frames, rotation_axis)

        # The studio lights emit normalised radiance, with 1.0 as the brightest source. Therefore, the PNG is scaled by
        # `display_exposure()` before the sRGB curve, or it comes out near black. The EXR below stays raw.
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        bmp = mi.Bitmap(np.array(image) * display_exposure())
        bmp = bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
        bmp.write(frame_path)
        frames.append(frame_path)

        # Every tenth frame is also saved as an EXR.
        if i % 10 == 0:
            exr_path = os.path.join(output_dir, f"frame_{i:04d}.exr")
            mi.util.write_bitmap(exr_path, image)

        frame_time = time.time() - frame_start
        print(f" done ({frame_time:.1f}s)")

    total_time = time.time() - start_time
    print(f"\n[animation] Saved {len(frames)} frames to {output_dir}")
    print(f"[animation] Total time: {total_time:.1f}s ({total_time/len(frames):.1f}s per frame)")

    # The video and GIF are optional, so a missing library does not stop the render.
    if make_video:
        try:
            import cv2
            import imageio

            print("[video] Creating video from frames...")

            # The first frame gives the dimensions of the video.
            first_frame = cv2.imread(frames[0])
            height, width = first_frame.shape[:2]

            video_path = os.path.join(output_dir, "diamond_rotation.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

            for frame_path in frames:
                frame = cv2.imread(frame_path)
                video_writer.write(frame)

            video_writer.release()
            print(f"[video] Video saved to {video_path}")

            # The GIF is easier to share than the video.
            print("[video] Creating GIF...")
            gif_path = os.path.join(output_dir, "diamond_rotation.gif")

            # The GIF keeps about 36 frames, to limit its size.
            gif_frames = []
            step = max(1, len(frames) // 36)
            for i in range(0, len(frames), step):
                img = cv2.imread(frames[i])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gif_frames.append(img_rgb)

            imageio.mimsave(gif_path, gif_frames, duration=0.05, loop=0)
            print(f"[video] GIF saved to {gif_path}")

        except ImportError as e:
            print(f"\n[video] Required library not installed: {e}")
            print("[video] To create video, install: pip install imageio opencv-python")
        except Exception as e:
            print(f"\n[video] Error creating video: {e}")

if __name__ == "__main__":
    create_rotation_animation(
        total_frames=36,
        rotation_axis='y',
        output_dir="diamond_animation",
        make_video=True
    )
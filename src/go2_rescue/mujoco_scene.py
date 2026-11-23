"""Load the official binary, or assemble a labelled local task from its geometry."""
from __future__ import annotations

from pathlib import Path
import hashlib

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def quat(matrix):
    return np.roll(Rotation.from_matrix(matrix).as_quat(), 1)


def task_scene(root: Path, output: Path, spawn: tuple[float, float], layout="finish-area", save_model=True,
               pipe_ramp="official"):
    """Compile a local scene with optional user-authorized ramp relocation.

    This is a development scene, never presented as an unmodified official MJB.
    pipe_ramp='entry' relocates only the original east ramp to the west mouth.
    Original files are not written. Imported mesh vertices are in their compiled
    local frames; recompilation carries the mesh-centering transform into geoms.
    """
    source = mujoco.MjModel.from_binary_path(str(root / "scene/mujoco/competition-course.mjb"))
    if pipe_ramp not in {"official", "entry"}:
        raise ValueError("Unknown pipe ramp layout")
    moved_name = "tunnel2_to3_sloped_continuous_threshold"
    moved_id = source.geom(moved_name).id if pipe_ramp == "entry" else None
    turn = np.diag([-1., -1., 1.])
    # Mirror the existing ramp to the western mouth by rotating about the
    # measured pipe centre. Compile a fresh model so collision BVHs agree.
    if layout == "finish-area":
        # These named partitions identify the far finish enclosure in this
        # resource revision. Placement is scene setup only, never a nav goal.
        west = source.geom("finish_west_partition_north_continuous_mortar_core").pos[0]
        south = source.geom("finish_south_partition_continuous_mortar_core").pos[1]
        marker_position = np.array([west + 1.9, south + 1.8, .43])
    elif layout == "near-start":
        # Retain the old short integration fixture only as an explicit option.
        marker_position = np.array([spawn[0] + 1.25, spawn[1], .43])
    elif layout == "perception":
        marker_position = np.array([spawn[0] + 1.55, spawn[1] + .20, .43])
    else:
        raise ValueError(f"Unknown task layout: {layout}")
    ball_position = marker_position + [.35, .25, -.32]
    state = mujoco.MjData(source)
    mujoco.mj_forward(source, state)
    center = state.geom_xpos[source.geom("tunnel2_internal_circular_segment_flat_insert").id].copy()
    spec = mujoco.MjSpec.from_file(str(root / "robot/go2/physics/go2.xml"))
    spec.option.timestep = .002
    meshes = {}
    imported = []
    for i in range(source.ngeom):
        body = int(source.geom_bodyid[i])
        if source.body_rootid[body] == source.body("base_link").id:
            continue
        kind = int(source.geom_type[i])
        extras = {}
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(source.geom_dataid[i])
            if mid not in meshes:
                va, vn = int(source.mesh_vertadr[mid]), int(source.mesh_vertnum[mid])
                fa, fn = int(source.mesh_faceadr[mid]), int(source.mesh_facenum[mid])
                name = f"course_mesh_{mid}"
                spec.add_mesh(name=name, uservert=source.mesh_vert[va:va + vn].ravel().tolist(),
                              userface=source.mesh_face[fa:fa + fn].ravel().tolist())
                meshes[mid] = name
            extras["meshname"] = meshes[mid]
        name = f"course_geom_{i}"
        position, rotation = state.geom_xpos[i].copy(), state.geom_xmat[i].reshape(3, 3).copy()
        if i == moved_id:
            position = center+turn @ (position-center)
            rotation = turn @ rotation
        spec.worldbody.add_geom(name=name, type=kind, size=source.geom_size[i], pos=position,
                                quat=quat(rotation),
                                contype=int(source.geom_contype[i]), conaffinity=int(source.geom_conaffinity[i]),
                                condim=int(source.geom_condim[i]), friction=source.geom_friction[i],
                                solref=source.geom_solref[i], solimp=source.geom_solimp[i],
                                margin=float(source.geom_margin[i]), gap=float(source.geom_gap[i]),
                                priority=int(source.geom_priority[i]), group=3, rgba=source.geom_rgba[i], **extras)
        imported.append(i)
    spec.worldbody.add_light(name="task_light", pos=[3, 3, 7], dir=[0, 0, -1], diffuse=[.8, .8, .8])
    spec.add_texture(name="football_checker", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[.02, .02, .02],
                     rgb2=[.92, .92, .9], width=512, height=512)
    spec.add_material(name="football_pattern", textures=["", "football_checker", "", "", "", "", "", "", "", ""],
                      texrepeat=[5, 3], specular=.25, roughness=.62)
    ball = spec.worldbody.add_body(name="local_football", pos=ball_position)
    ball.add_freejoint(name="local_football_free")
    ball.add_geom(name="local_football_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  size=[.11, 0, 0], mass=.43, material="football_pattern", friction=[.8, .02, .001], condim=3)
    if layout != "perception":
        marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 42, 256)
        marker = cv2.copyMakeBorder(marker, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255)
        # Raw pixels are embedded in the runtime model; no remote texture required.
        rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
        spec.add_texture(name="local_finish_texture", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         width=320, height=320, nchannel=3, data=rgb.ravel().tolist())
        spec.add_material(name="local_finish_material", textures=["", "local_finish_texture", "", "", "", "", "", "", "", ""],
                          texuniform=True, texrepeat=[1, 1], emission=.5, reflectance=0.)
        rotation = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
        # Explicit UVs avoid MuJoCo's size-dependent box texture repetition.
        vertices = [[x, y, z] for z in [-.003, .003] for y in [-.125, .125] for x in [-.125, .125]]
        faces = [[4, 5, 7], [4, 7, 6], [0, 2, 3], [0, 3, 1], [0, 1, 5], [0, 5, 4],
                 [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]]
        uv = [[0, 0], [1, 0], [0, 1], [1, 1]] * 2
        spec.add_mesh(name="local_marker_mesh", uservert=np.asarray(vertices).ravel().tolist(),
                      userface=np.asarray(faces).ravel().tolist(), usertexcoord=np.asarray(uv).ravel().tolist(),
                      userfacetexcoord=np.asarray(faces).ravel().tolist())
        spec.add_material(name="local_board_white", rgba=[1, 1, 1, 1], emission=.5)
        spec.worldbody.add_geom(name="local_finish_marker", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="local_marker_mesh",
                                pos=marker_position, quat=quat(rotation),
                                size=[.125, .125, .003], material="local_board_white", rgba=[1, 1, 1, 1], contype=0, conaffinity=0)
        # Render the actual marker cells as tiny static geometries. This keeps its
        # optical appearance independent of mesh UV conventions across MuJoCo builds.
        pattern = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 42, 6)
        cell = .2 / 6
        for row, col in np.argwhere(pattern == 0):
            spec.worldbody.add_geom(name=f"local_finish_cell_{row}_{col}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                    pos=marker_position + [-.004, -(col - 2.5) * cell, (2.5 - row) * cell],
                                    quat=quat(rotation), size=[cell / 2, cell / 2, .0008],
                                    rgba=[0, 0, 0, 1], contype=0, conaffinity=0)
    model = spec.compile()
    data = mujoco.MjData(model)
    # Check imported collision placement after mesh re-centering by comparing
    # actual world vertices. This prevents a visually similar but moved course.
    mujoco.mj_forward(model, data)
    max_error = 0.
    from scipy.spatial import cKDTree
    for i in imported:
        j = model.geom(f"course_geom_{i}").id
        if source.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            def vertices(m, d, geom):
                mesh = int(m.geom_dataid[geom])
                start, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
                return m.mesh_vert[start:start + count] @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]
            a, b = vertices(source, state, i), vertices(model, data, j)
            if i == moved_id:
                a = (a-center) @ turn.T+center
            max_error = max(max_error, float(cKDTree(a).query(b)[0].max()))
        else:
            max_error = max(max_error, float(np.linalg.norm(state.geom_xpos[i] - data.geom_xpos[j])))
    if max_error > 1e-4:
        raise RuntimeError(f"Imported official collision geometry moved by {max_error}m")
    import json
    (output / "scene_provenance.json").write_text(json.dumps({
        "source": "scene/mujoco/competition-course.mjb",
        "sha256": hashlib.sha256((root / "scene/mujoco/competition-course.mjb").read_bytes()).hexdigest(),
        "scene_kind": "local_entry_ramp_plus_task_objects" if moved_id is not None else "official_geometry_plus_local_task_objects",
        "pipe_ramp": pipe_ramp,
        "geometry_changes": [] if moved_id is None else [{"source_geom": moved_name,
            "operation": "rotate 180 degrees about tunnel2 centre; move east ramp to west entrance",
            "center_world_xyz": center.tolist(), "new_world_position": data.geom_xpos[model.geom(f"course_geom_{moved_id}").id].tolist()}],
        "imported_geoms": len(imported),
        "max_geometry_vertex_error_m": max_error,
        "task_layout": layout,
        "diagnostics_only_placements": {"football_world_xyz": ball_position.tolist(),
                                         "finish_marker_world_xyz": marker_position.tolist() if layout != "perception" else None},
        "additions": ["dynamic football", "light"] if layout == "perception" else ["dynamic football", "local ArUco marker", "light"]}, indent=2) + "\n")
    if save_model:
        mujoco.mj_saveModel(model, str(output / "runtime-scene.mjb"))
    return model

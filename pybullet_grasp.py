"""Predict and execute closed-loop wrist-camera RGB-D grasps on a simulated Franka Panda."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "robotic-grasping"
RESULTS = REPO / "results"
CHECKPOINT = REPO / "trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93"
CHECKPOINT_SHA256 = "adfb2cbbb8df2708a732e12ddc4db114f3ec399ffb5d403ca75c5b5b9e769171"
SIZE = 224
FOV = 24
NEAR, FAR = 0.05, 1.5
EYE = (0.55, 0.0, 0.65)
FORWARD = (0, 0, -1)
UP = (0, 1, 0)
WRIST_CAMERA_OFFSET = (0.070, 0, -0.026)  # In the Panda grasp-target link frame.
ARM = tuple(range(7))
FINGERS = (9, 10)
TCP = 11
HOME = (0, -0.5, 0, -2.4, 0, 2.0, 0.8)
HOLD_STEPS = 240
LIFT_THRESHOLD = 0.10
SERVO_CONTROL_STEPS = 12  # 20 Hz visual servo loop at 240 Hz physics.
MAX_SERVO_STEPS = 90
MAX_TRANS_STEP = 0.028
MAX_ROT_STEP = math.radians(14.0)
VISUAL_STANDOFF = 0.32
PREGRASP_STANDOFF = 0.17
MIN_GRASP_OPENING = 0.003
DEFAULT_MAX_ATTEMPTS = 2
OBJECTS = {
    # Geometry, dimensions (diameter/height for the cylinder), mass and color.
    "cylinder": ("cylinder", (0.052, 0.045), 0.055, (0.95, 0.56, 0.09, 1)),
    "cube": ("box", (0.042, 0.042, 0.042), 0.045, (0.22, 0.66, 0.89, 1)),
    "long_box": ("box", (0.091, 0.039, 0.036), 0.060, (0.43, 0.77, 0.32, 1)),
    "flat_box": ("box", (0.075, 0.052, 0.023), 0.035, (0.77, 0.35, 0.67, 1)),
    "sphere": ("sphere", (0.046,), 0.040, (0.87, 0.32, 0.23, 1)),
}
YCB_ROOT = ROOT / "assets" / "ycb"
YCB_OBJECTS = (
    "tomato_soup_can", "boya_tower", "gelatin_box", "banana", "large_marker", "mug", "mustard_bottle",
)
BOYA_MESH = ROOT / "assets" / "boya_tower.obj"
BOYA_TEX = ROOT / "assets" / "boya_tower.png"


class NoGraspCandidate(RuntimeError):
    """The RGB-D prediction has no center allowed by the depth mask."""


def desktop_work_area():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        # Set this before Bullet creates its GUI thread and OpenGL window.
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        work_area = wintypes.RECT()
        if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work_area), 0):  # SPI_GETWORKAREA
            return work_area.left, work_area.top, work_area.right - work_area.left, work_area.bottom - work_area.top
    return 0, 0, 1366, 864


def desktop_window_options(work_area):
    _, _, width, height = work_area
    width = min(1920, width - preview_width(work_area) - 112)
    height = min(1080, height - 88)
    return f"--width={width} --height={height}"


def preview_width(work_area):
    return min(424, work_area[2] // 3)


def move_bullet_window(work_area, preview):
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32")
    user32.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, wintypes.UINT)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def position_window(window, _):
        if not user32.IsWindowVisible(window):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
        if pid.value != os.getpid():
            return True
        title = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(window, title, len(title))
        if "bullet" in title.value.lower() or "example" in title.value.lower():
            _, top, _, _ = work_area
            preview_right = preview.root.winfo_rootx() + preview.root.winfo_width()
            user32.SetWindowPos(window, None, preview_right + 28, top + 8,
                                0, 0, 0x0001 | 0x0004 | 0x0010)
        return True

    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.EnumWindows.argtypes = (callback_type, wintypes.LPARAM)
    user32.EnumWindows(position_window, 0)


class CameraPreview:
    def __init__(self, work_area):
        import tkinter as tk

        left, top, _, _ = work_area
        width = preview_width(work_area)
        self.size = width - 24
        self.root = tk.Tk()
        self.root.title("Wrist camera")
        self.root.geometry(f"{width}x{width + 22}+{left + 8}+{top + 8}")
        self.root.protocol("WM_DELETE_WINDOW", self.root.withdraw)
        self.label = tk.Label(self.root, text="Press Space in PyBullet to capture")
        self.label.pack(expand=True)
        self.update()

    def show(self, image):
        from PIL import Image, ImageTk

        self.image = ImageTk.PhotoImage(image.resize((self.size, self.size), Image.Resampling.BILINEAR))
        self.label.configure(image=self.image, text="")
        self.root.deiconify()
        self.update()

    def update(self):
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        self.root.destroy()


def spawn_object(p, kind, x, y, yaw):
    orientation = p.getQuaternionFromEuler((0, 0, yaw))
    if kind == "boya_tower":
        collision = p.createCollisionShape(p.GEOM_MESH, fileName=str(BOYA_MESH))
        visual = p.createVisualShape(p.GEOM_MESH, fileName=str(BOYA_MESH), rgbaColor=(1, 1, 1, 1))
        obj = p.createMultiBody(0.060, collision, visual, (x, y, 0.05), orientation)
        if BOYA_TEX.is_file():
            tex = p.loadTexture(str(BOYA_TEX))
            p.changeVisualShape(obj, -1, textureUniqueId=tex, rgbaColor=(1, 1, 1, 1),
                                specularColor=(0.24, 0.24, 0.24))
        lower, _ = p.getAABB(obj)
        p.resetBasePositionAndOrientation(obj, (x, y, 0.005 - lower[2]), orientation)
        return obj
    if kind in YCB_OBJECTS:
        urdf = YCB_ROOT / f"{kind}.urdf"
        obj = p.loadURDF(str(urdf), (x, y, 0), orientation)
        lower, _ = p.getAABB(obj)
        p.resetBasePositionAndOrientation(obj, (x, y, 0.005 - lower[2]), orientation)
        return obj
    shape, dimensions, mass, color = OBJECTS[kind]
    if shape == "box":
        half = [d / 2 for d in dimensions]
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        bottom_offset = half[2]
    elif shape == "cylinder":
        diameter, height = dimensions
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=diameter / 2, height=height)
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=diameter / 2, length=height, rgbaColor=color)
        bottom_offset = height / 2
    else:
        bottom_offset = dimensions[0] / 2
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=bottom_offset)
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=bottom_offset, rgbaColor=color)
    return p.createMultiBody(mass, collision, visual, (x, y, bottom_offset + 0.005),
                             orientation)


def pixel_to_world(row, col, depth, eye=EYE, forward=FORWARD, up=UP):
    right = np.cross(forward, up)
    scale = 2 * math.tan(math.radians(FOV / 2)) / SIZE
    offset = (SIZE - 1) / 2
    ray = (np.asarray(forward) + np.asarray(col - offset)[..., None] * scale * right
           + np.asarray(offset - row)[..., None] * scale * np.asarray(up))
    return np.asarray(eye) + np.asarray(depth)[..., None] * ray


def grasp_direction(angle, forward=FORWARD, up=UP):
    right = np.cross(forward, up)
    return (math.cos(angle) * right + math.sin(angle) * np.asarray(up))[:2]


def wrist_camera(p, robot):
    position, orientation = p.getLinkState(robot, TCP, computeForwardKinematics=True)[4:6]
    rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    eye = np.asarray(position) + rotation @ WRIST_CAMERA_OFFSET
    return eye, rotation[:, 2], -rotation[:, 1]


def lift_succeeded(lift, fingers, on_support):
    return lift >= LIFT_THRESHOLD and set(FINGERS).issubset(fingers) and not on_support


def render(p, view, projection, size, robot=None):
    _, _, rgba, z_buffer, segmentation = p.getCameraImage(
        size, size, viewMatrix=view, projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER, flags=p.ER_NO_SEGMENTATION_MASK if robot is None else 0,
    )
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(size, size, 4)[:, :, :3]
    z_buffer = np.asarray(z_buffer).reshape(size, size)
    depth = FAR * NEAR / (FAR - (FAR - NEAR) * z_buffer)
    if robot is None:
        return rgb, depth
    return rgb, depth, np.asarray(segmentation).reshape(size, size) == robot


def predict(rgb, depth, model, device, points, robot_mask):
    import torch
    from scipy.ndimage import distance_transform_edt

    from inference.post_process import post_process_output
    from utils.dataset_processing.image import DepthImage, Image

    # Remove the tabletop's depth gradient in an oblique view before normalising.
    foreground = (points[..., 2] > 0.012) & (depth < FAR - 0.01) & ~robot_mask
    relative_depth = np.where(foreground, -points[..., 2], 0).astype(np.float32)
    depth_image = DepthImage(relative_depth)
    depth_image.normalise()
    rgb_image = Image(rgb.copy())
    rgb_image.normalise()
    image = np.concatenate((depth_image.img[None], rgb_image.img.transpose(2, 0, 1)))
    tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred = model.predict(tensor)
    quality, angle, width = post_process_output(pred["pos"], pred["cos"], pred["sin"], pred["width"])

    clearance = distance_transform_edt(foreground)
    feasible = (clearance >= 7) & np.isfinite(quality) & np.isfinite(angle) & np.isfinite(width)
    if not feasible.any():
        raise NoGraspCandidate("No grasp center has enough depth-derived tabletop clearance")
    row, col = np.unravel_index(np.argmax(np.where(feasible, quality, -np.inf)), quality.shape)
    camera_depth = float(depth[row, col])
    return int(row), int(col), float(quality[row, col]), float(angle[row, col]), float(width[row, col]), camera_depth


def arm_ik(p, robot, target, orientation):
    limits = [p.getJointInfo(robot, j)[8:10] for j in (*ARM, *FINGERS)]
    lower, upper = zip(*limits)
    return p.calculateInverseKinematics(
        robot, TCP, target, orientation, lowerLimits=lower, upperLimits=upper,
        jointRanges=[hi - lo for lo, hi in limits], restPoses=(*HOME, 0.04, 0.04),
        maxNumIterations=160, residualThreshold=1e-5,
    )[:7]


def set_arm(p, robot, joints):
    p.setJointMotorControlArray(
        robot, ARM, p.POSITION_CONTROL, targetPositions=joints, forces=[120] * 7,
    )


def move(p, robot, target, orientation, steps, advance):
    set_arm(p, robot, arm_ik(p, robot, target, orientation))
    advance(steps)


def step(p, steps, realtime=False):
    for _ in range(steps):
        p.stepSimulation()
        if realtime:
            time.sleep(1 / 240)


def set_fingers(p, robot, opening):
    p.setJointMotorControlArray(
        robot, FINGERS, p.POSITION_CONTROL, targetPositions=[opening / 2] * 2, forces=[30, 30],
    )


def contacts(p, robot, obj, support):
    fingers = {c[3] for c in p.getContactPoints(robot, obj) if c[3] in FINGERS and c[9] > 0}
    return fingers, any(c[9] > 0 for c in p.getContactPoints(obj, support))


def record_pose(p, robot, obj, support, initial_z, run, phase, index):
    obj_position = p.getBasePositionAndOrientation(obj)[0]
    tcp = p.getLinkState(robot, TCP, computeForwardKinematics=True)[4]
    fingers, on_support = contacts(p, robot, obj, support)
    lift = obj_position[2] - initial_z
    state = {
        "phase": phase, "object_xyz_m": list(obj_position), "tcp_xyz_m": list(tcp),
        "lift_m": lift, "finger_contact_links": sorted(fingers), "on_support": on_support,
    }
    run.log({"phase_index": index, "object_lift_m": lift, "finger_contacts": len(fingers),
             "object_on_support": int(on_support)})
    print(json.dumps(state), flush=True)
    return state


def side_image(p, path):
    from PIL import Image

    view = p.computeViewMatrix((1.0, -0.86, 0.53), (0.42, 0, 0.18), (0, 0, 1))
    projection = p.computeProjectionMatrixFOV(47, 1, NEAR, FAR)
    rgb, _ = render(p, view, projection, 720)
    Image.fromarray(rgb).save(path)


def randomize_arm(p, robot, rng):
    from scipy.spatial.transform import Rotation

    limits = [p.getJointInfo(robot, joint)[8:10] for joint in ARM]
    for _ in range(40):
        azimuth = rng.uniform(-math.pi, math.pi)
        tilt = math.radians(rng.uniform(9, 38))
        view_center = rng.uniform((0.53, -0.02, 0.03), (0.57, 0.02, 0.06))
        forward = np.array((math.sin(tilt) * math.cos(azimuth), math.sin(tilt) * math.sin(azimuth), -math.cos(tilt)))
        eye = view_center - rng.uniform(0.56, 0.72) * forward
        roll = rng.uniform(-math.pi * 4 / 9, math.pi * 4 / 9)
        up = np.array((math.sin(roll), math.cos(roll), 0))
        up -= np.dot(up, forward) * forward
        up /= np.linalg.norm(up)
        rotation = np.column_stack((np.cross(forward, up), -up, forward))
        orientation = Rotation.from_matrix(rotation).as_quat()
        target = eye - rotation @ WRIST_CAMERA_OFFSET
        for joint, position in zip(ARM, HOME):
            p.resetJointState(robot, joint, position)
        joints = arm_ik(p, robot, target, orientation)
        if any(not lo <= value <= hi for value, (lo, hi) in zip(joints, limits)):
            continue
        for joint, position in zip(ARM, joints):
            p.resetJointState(robot, joint, position)
        actual_eye, forward, _ = wrist_camera(p, robot)
        p.performCollisionDetection()
        if (np.linalg.norm(actual_eye - eye) < 0.015 and np.dot(forward, rotation[:, 2]) > 0.999
                and not p.getContactPoints(robot)):
            set_arm(p, robot, joints)
            return
    raise RuntimeError("Could not sample a reachable Panda camera pose facing the workspace")


def setup_scene(p, gui, kind, x, y, yaw, rng):
    import pybullet_data

    p.resetSimulation()
    if gui:
        p.resetDebugVisualizerCamera(0.62, 42, -26, (0.52, 0.0, 0.10))
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition=(1.6, -1.2, 2.2))
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, 1,
            shadowMapResolution=8192,
            shadowMapWorldSize=2,
            shadowMapIntensity=0.25,
        )
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)
    p.setPhysicsEngineParameter(numSolverIterations=150, deterministicOverlappingPairs=1)
    support = p.loadURDF("plane.urdf")
    p.changeVisualShape(support, -1, specularColor=(0.0, 0.0, 0.0))
    obj = spawn_object(p, kind, x, y, yaw)
    p.changeDynamics(obj, -1, lateralFriction=1.2)
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    for joint in FINGERS:
        p.resetJointState(robot, joint, 0.04)
        p.changeDynamics(robot, joint, lateralFriction=1.5)
    set_fingers(p, robot, 0.08)
    randomize_arm(p, robot, rng)
    step(p, 240)
    return support, obj, robot


def grasp_axes(direction, angle_deg):
    closing = np.array((*direction, 0.0))
    closing /= np.linalg.norm(closing)
    across = np.cross((0, 0, 1), closing)
    tilt = math.radians(angle_deg)
    forward = math.sin(tilt) * across + np.array((0, 0, -math.cos(tilt)))
    return np.column_stack((np.cross(closing, forward), closing, forward))


def approach_waypoints(target, forward):
    return np.linspace(np.asarray(target) - 0.17 * forward, target, 24)


def plan_grasp(p, robot, support, xyz, direction, points, camera_eye, requested_angle):
    from scipy.spatial.transform import Rotation

    angles = (25, -25, 40, -40, 55, -55) if requested_angle is None else (requested_angle,)
    candidates = []
    for angle in angles:
        axes = grasp_axes(direction, angle)
        forward = axes[:, 2]
        target = xyz + 0.026 * forward
        target[2] = max(0.010, target[2])
        path = approach_waypoints(target, forward)
        # Check the visible approach corridor above the final contact zone.
        offsets = points - target
        along = np.einsum("ij,j->i", offsets, -forward)
        radial_squared = np.sum(offsets**2, axis=1) - along**2
        obstacles = np.count_nonzero((along > 0.045) & (along < 0.18) & (radial_squared < 0.038**2))
        candidates.append((obstacles, np.linalg.norm(path[0] - camera_eye), angle, axes, path))

    current = p.getJointStates(robot, ARM)
    limits = [p.getJointInfo(robot, j)[8:10] for j in ARM]
    try:
        for _, _, angle, axes, path in sorted(candidates, key=lambda item: item[:2]):
            orientation = Rotation.from_matrix(axes).as_quat()
            for joint, state in zip(ARM, current):
                p.resetJointState(robot, joint, state[0], state[1])
            joint_path = []
            for waypoint in path:
                joints = arm_ik(p, robot, waypoint, orientation)
                if any(not lo <= value <= hi for value, (lo, hi) in zip(joints, limits)):
                    break
                for joint, value in zip(ARM, joints):
                    p.resetJointState(robot, joint, value)
                position, rotation = p.getLinkState(robot, TCP, computeForwardKinematics=True)[4:6]
                if (np.linalg.norm(np.asarray(position) - waypoint) > 0.012
                        or abs(np.dot(rotation, orientation)) < 0.998
                        or any(contact[3] >= 0 for contact in p.getClosestPoints(robot, support, 0.0))):
                    break
                joint_path.append(joints)
            else:
                return angle, orientation, path, joint_path
    finally:
        for joint, state in zip(ARM, current):
            p.resetJointState(robot, joint, state[0], state[1])
    raise NoGraspCandidate("No reachable grasp approach clears the tabletop")


def align_grasp_direction(direction, reference):
    vec = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.array((1.0, 0.0)) if reference is None else np.asarray(reference, dtype=float)
    vec = vec / norm
    if reference is not None and float(np.dot(vec, reference)) < 0.0:
        vec = -vec
    return vec


def target_has_foreground(visible_points, target_xyz, radius=0.025):
    if len(visible_points) == 0:
        return False
    diffs = visible_points - np.asarray(target_xyz, dtype=float)
    return bool(np.any(np.einsum("ij,ij->i", diffs, diffs) <= radius**2))


def blend_grasp_target(previous, observation, alpha_smooth=0.35, alpha_jump=0.80, jump_threshold=0.03):
    obs_xyz = np.asarray(observation[0], dtype=float)
    obs_dir = align_grasp_direction(observation[1], None if previous is None else previous[1])
    obs_opening = float(observation[2])
    if previous is None:
        return obs_xyz.copy(), obs_dir, obs_opening, 0.0
    prev_xyz, prev_dir, prev_opening = previous
    shift = float(np.linalg.norm(obs_xyz - prev_xyz))
    alpha = alpha_jump if shift >= jump_threshold else alpha_smooth
    alpha_dir = alpha_jump if shift >= jump_threshold else 0.20
    xyz = (1.0 - alpha) * prev_xyz + alpha * obs_xyz
    direction = align_grasp_direction((1.0 - alpha_dir) * prev_dir + alpha_dir * obs_dir, prev_dir)
    opening = (1.0 - alpha) * prev_opening + alpha * obs_opening
    return xyz, direction, opening, shift


def nearest_grasp_frame(direction, angle_deg, cur_quat):
    from scipy.spatial.transform import Rotation

    axes_a = grasp_axes(direction, angle_deg)
    quat_a = Rotation.from_matrix(axes_a).as_quat()
    axes_b = grasp_axes(-np.asarray(direction, dtype=float), -angle_deg)
    quat_b = Rotation.from_matrix(axes_b).as_quat()
    if quaternion_angle(cur_quat, quat_b) < quaternion_angle(cur_quat, quat_a):
        return axes_b, quat_b
    return axes_a, quat_a


def camera_centered_tcp(target_xyz, grasp_target, axes, standoff, camera_blend):
    forward = axes[:, 2]
    tcp_aligned = grasp_target - standoff * forward
    tcp_camera = target_xyz - standoff * forward - axes @ WRIST_CAMERA_OFFSET
    blend = float(np.clip(camera_blend, 0.0, 1.0))
    return (1.0 - blend) * tcp_aligned + blend * tcp_camera


def quaternion_angle(q1, q2):
    from scipy.spatial.transform import Rotation

    return float((Rotation.from_quat(q1).inv() * Rotation.from_quat(q2)).magnitude())


def step_pose_toward(cur_pos, cur_quat, des_pos, des_quat, max_trans=MAX_TRANS_STEP, max_rot=MAX_ROT_STEP):
    from scipy.spatial.transform import Rotation, Slerp

    cur_pos = np.asarray(cur_pos, dtype=float)
    des_pos = np.asarray(des_pos, dtype=float)
    diff = des_pos - cur_pos
    dist = float(np.linalg.norm(diff))
    next_pos = des_pos.copy() if dist <= max_trans or dist < 1e-9 else cur_pos + diff * (max_trans / dist)

    angle = quaternion_angle(cur_quat, des_quat)
    if angle <= max_rot or angle < 1e-9:
        next_quat = np.asarray(des_quat, dtype=float)
    else:
        t = max_rot / angle
        slerp = Slerp([0.0, 1.0], Rotation.from_quat([cur_quat, des_quat]))
        next_quat = slerp(t).as_quat()
    return next_pos, next_quat


def finger_opening(p, robot):
    return float(sum(p.getJointState(robot, j)[0] for j in FINGERS))


def make_grasp_overlay(rgb, row, col, angle, width_px):
    from PIL import Image, ImageDraw

    overlay = Image.fromarray(rgb)
    draw = ImageDraw.Draw(overlay)
    dx, dy = math.cos(angle) * width_px / 2, -math.sin(angle) * width_px / 2
    draw.line((col - dx, row - dy, col + dx, row + dy), fill="cyan", width=2)
    draw.ellipse((col - 3, row - 3, col + 3, row + 3), outline="white", width=2)
    return overlay


def capture_and_predict(p, robot, model, device, preview=None):
    from PIL import Image

    camera = wrist_camera(p, robot)
    eye, forward, up = camera
    view = p.computeViewMatrix(eye, eye + forward, up)
    projection = p.computeProjectionMatrixFOV(FOV, 1, NEAR, FAR)
    rgb, depth, robot_mask = render(p, view, projection, SIZE, robot)
    rows, cols = np.ogrid[:SIZE, :SIZE]
    points = pixel_to_world(rows, cols, depth, *camera)
    try:
        row, col, quality, angle, width_px, camera_depth = predict(rgb, depth, model, device, points, robot_mask)
    except NoGraspCandidate:
        if preview is not None:
            preview.show(Image.fromarray(rgb))
        raise
    xyz = points[row, col]
    width_m = width_px * 2 * camera_depth * math.tan(math.radians(FOV / 2)) / SIZE
    direction = grasp_direction(angle, forward, up)
    opening = float(np.clip(width_m + 0.034, 0.068, 0.080))
    overlay = make_grasp_overlay(rgb, row, col, angle, width_px)
    if preview is not None:
        preview.show(overlay)
    visible = points[(depth < FAR - 0.01) & ~robot_mask & (points[..., 2] > 0.006)]
    return {
        "camera": camera, "rgb": rgb, "depth": depth, "robot_mask": robot_mask,
        "points": points, "visible": visible, "overlay": overlay,
        "row": row, "col": col, "quality": quality, "angle": angle,
        "width_px": width_px, "width_m": width_m, "opening": opening,
        "xyz": xyz, "direction": direction,
    }


def safe_ik_step(p, robot, support, target_pos, target_quat):
    current = p.getJointStates(robot, ARM)
    limits = [p.getJointInfo(robot, j)[8:10] for j in ARM]
    try:
        joints = arm_ik(p, robot, target_pos, target_quat)
        if any(not lo <= value <= hi for value, (lo, hi) in zip(joints, limits)):
            return None
        for joint, value in zip(ARM, joints):
            p.resetJointState(robot, joint, value)
        if any(contact[3] >= 0 for contact in p.getClosestPoints(robot, support, 0.0)):
            return None
        return joints
    finally:
        for joint, state in zip(ARM, current):
            p.resetJointState(robot, joint, state[0], state[1])


def apply_object_disturbance(p, obj, rng, distance_m):
    if distance_m <= 0:
        return None
    pos, quat = p.getBasePositionAndOrientation(obj)
    heading = float(rng.uniform(-math.pi, math.pi))
    nx = float(np.clip(pos[0] + distance_m * math.cos(heading), 0.47, 0.63))
    ny = float(np.clip(pos[1] + distance_m * math.sin(heading), -0.09, 0.09))
    new_pos = (nx, ny, pos[2])
    p.resetBasePositionAndOrientation(obj, new_pos, quat)
    p.resetBaseVelocity(obj, (0, 0, 0), (0, 0, 0))
    return {"before_xyz_m": list(pos), "after_xyz_m": list(new_pos),
            "displacement_m": float(np.linalg.norm(np.asarray(new_pos) - np.asarray(pos)))}


def finish_lift_and_hold(p, robot, obj, support, initial_z, run, output, placement, prediction,
                         grasp_target, orientation, states, advance, preview, extra_summary=None):
    for lift in np.linspace(0, 0.16, 25)[1:]:
        move(p, robot, grasp_target + (0, 0, lift), orientation, 10, advance)
    states.append(record_pose(p, robot, obj, support, initial_z, run, "lift", len(states)))
    hold_passed = True
    minimum_hold_lift = float("inf")
    for index in range(HOLD_STEPS):
        advance(1)
        if preview is not None and index % 8 == 0:
            preview.update()
        current_lift = p.getBasePositionAndOrientation(obj)[0][2] - initial_z
        minimum_hold_lift = min(minimum_hold_lift, current_lift)
        fingers, on_support = contacts(p, robot, obj, support)
        hold_passed &= lift_succeeded(current_lift, fingers, on_support)
    final = record_pose(p, robot, obj, support, initial_z, run, "hold", len(states))
    states.append(final)
    side_image(p, output / "lifted-object.png")
    outcome = "success"
    if not hold_passed:
        closed = next((s for s in reversed(states) if s["phase"] == "close"), states[-3])
        lifted = next((s for s in reversed(states) if s["phase"] == "lift"), states[-2])
        if not set(FINGERS).issubset(closed["finger_contact_links"]):
            outcome = "no_bilateral_contact_at_close"
        elif not lift_succeeded(lifted["lift_m"], lifted["finger_contact_links"], lifted["on_support"]):
            outcome = "failed_lift"
        else:
            outcome = "failed_hold"
    summary = {
        **placement, "success": hold_passed, "outcome": outcome, "minimum_hold_lift_m": minimum_hold_lift,
        "initial_object_z_m": initial_z, "hold_seconds": HOLD_STEPS / 240,
        "prediction": prediction, "phases": states, **(extra_summary or {}),
    }
    save_trial(output, summary)
    run.log({"grasp_success": int(hold_passed), "final_lift_m": final["lift_m"],
             "minimum_hold_lift_m": minimum_hold_lift, "hold_seconds": HOLD_STEPS / 240})
    return hold_passed


def simulate_open_loop(p, model, device, run, output, rng, gui=False, kind="cylinder", x=0.56, y=0.034,
                       yaw=0.0, approach_deg=None, scene=None, preview=None, disturb_m=0.0):
    from PIL import Image

    def advance(steps):
        for start in range(0, steps, 8):
            step(p, min(8, steps - start), realtime=gui)
            if preview is not None and steps >= 8:
                preview.update()

    support, obj, robot = scene if scene is not None else setup_scene(p, gui, kind, x, y, yaw, rng)
    start_position, start_orientation = p.getBasePositionAndOrientation(obj)
    initial_z = start_position[2]
    placement = {"object": kind, "spawn_xy_m": [x, y], "spawn_yaw_rad": yaw,
                 "settled_xyz_m": list(start_position), "settled_quaternion": list(start_orientation),
                 "requested_approach_deg": approach_deg, "control_mode": "open-loop"}

    camera = wrist_camera(p, robot)
    eye, forward, up = camera
    placement["camera"] = {"mount_link": "panda_grasptarget", "eye_xyz_m": eye.tolist(),
                           "forward_world": forward.tolist(), "up_world": up.tolist(),
                           "arm_joint_rad": [p.getJointState(robot, joint)[0] for joint in ARM]}
    view_tilt = math.degrees(math.acos(np.clip(-forward[2], -1, 1)))
    run.log({"camera_tilt_deg": view_tilt})
    try:
        obs = capture_and_predict(p, robot, model, device, preview=preview)
    except NoGraspCandidate:
        save_trial(output, {**placement, "success": False, "outcome": "no_candidate"})
        run.log({"grasp_success": 0})
        return False

    Image.fromarray(obs["rgb"]).save(output / "camera-rgb.png")
    np.save(output / "camera-depth-m.npy", obs["depth"])
    obs["overlay"].save(output / "predicted-grasp.png")
    prediction = {
        "row": obs["row"], "column": obs["col"], "quality": obs["quality"], "angle_rad": obs["angle"],
        "width_px": obs["width_px"], "width_m": obs["width_m"], "opening_m": obs["opening"],
        "surface_xyz_m": obs["xyz"].tolist(),
    }
    print(json.dumps(prediction), flush=True)
    run.log({"prediction_quality": obs["quality"], "prediction_width_m": obs["width_m"],
             "prediction_angle_rad": obs["angle"], "predicted_center_row": obs["row"],
             "predicted_center_col": obs["col"]})

    try:
        angle_deg, orientation, waypoints, joint_path = plan_grasp(
            p, robot, support, obs["xyz"], obs["direction"], obs["visible"], eye, approach_deg
        )
    except NoGraspCandidate:
        save_trial(output, {**placement, "prediction": prediction, "success": False,
                            "outcome": "no_reachable_approach"})
        run.log({"grasp_success": 0})
        return False
    placement["approach_deg"] = angle_deg
    prediction["tcp_target_xyz_m"] = waypoints[-1].tolist()
    prediction["tcp_target_quaternion"] = orientation.tolist()
    run.log({"executed_approach_deg": angle_deg})
    set_fingers(p, robot, obs["opening"])
    set_arm(p, robot, joint_path[0])
    advance(48)
    disturbance = apply_object_disturbance(p, obj, rng, disturb_m)
    advance(192)
    states = [record_pose(p, robot, obj, support, initial_z, run, "pregrasp", 0)]
    for joints in joint_path[1:]:
        set_arm(p, robot, joints)
        advance(10)
    advance(80)
    states.append(record_pose(p, robot, obj, support, initial_z, run, "approach", 1))
    set_fingers(p, robot, 0)
    advance(160)
    states.append(record_pose(p, robot, obj, support, initial_z, run, "close", 2))
    extra = {"disturbance": disturbance} if disturbance is not None else None
    return finish_lift_and_hold(
        p, robot, obj, support, initial_z, run, output, placement, prediction,
        waypoints[-1], orientation, states, advance, preview, extra_summary=extra,
    )


def simulate_closed_loop(p, model, device, run, output, rng, gui=False, kind="cylinder", x=0.56, y=0.034,
                         yaw=0.0, approach_deg=None, scene=None, preview=None,
                         max_attempts=DEFAULT_MAX_ATTEMPTS, disturb_m=0.0):
    from PIL import Image
    from scipy.spatial.transform import Rotation

    def advance(steps):
        for start in range(0, steps, 8):
            step(p, min(8, steps - start), realtime=gui)
            if preview is not None and steps >= 8:
                preview.update()

    support, obj, robot = scene if scene is not None else setup_scene(p, gui, kind, x, y, yaw, rng)
    start_position, start_orientation = p.getBasePositionAndOrientation(obj)
    initial_z = start_position[2]
    placement = {"object": kind, "spawn_xy_m": [x, y], "spawn_yaw_rad": yaw,
                 "settled_xyz_m": list(start_position), "settled_quaternion": list(start_orientation),
                 "requested_approach_deg": approach_deg, "control_mode": "closed-loop"}

    camera = wrist_camera(p, robot)
    eye, forward, up = camera
    placement["camera"] = {"mount_link": "panda_grasptarget", "eye_xyz_m": eye.tolist(),
                           "forward_world": forward.tolist(), "up_world": up.tolist(),
                           "arm_joint_rad": [p.getJointState(robot, joint)[0] for joint in ARM]}
    view_tilt = math.degrees(math.acos(np.clip(-forward[2], -1, 1)))
    run.log({"camera_tilt_deg": view_tilt})

    states = []
    servo_trace = []
    visual_updates = 0
    total_servo_steps = 0
    disturbance = None
    first_prediction = None
    last_overlay = None
    last_grasp_target = None
    last_orientation = None
    angle_deg = None

    for attempt in range(1, max_attempts + 1):
        try:
            obs = capture_and_predict(p, robot, model, device, preview=preview)
        except NoGraspCandidate:
            if attempt == 1:
                save_trial(output, {**placement, "success": False, "outcome": "no_candidate"})
                run.log({"grasp_success": 0})
                return False
            break

        visual_updates += 1
        last_overlay = obs["overlay"]
        if first_prediction is None:
            Image.fromarray(obs["rgb"]).save(output / "camera-rgb.png")
            np.save(output / "camera-depth-m.npy", obs["depth"])
            obs["overlay"].save(output / "predicted-grasp.png")
            first_prediction = {
                "row": obs["row"], "column": obs["col"], "quality": obs["quality"], "angle_rad": obs["angle"],
                "width_px": obs["width_px"], "width_m": obs["width_m"], "opening_m": obs["opening"],
                "surface_xyz_m": obs["xyz"].tolist(),
            }
            print(json.dumps(first_prediction), flush=True)
            run.log({"prediction_quality": obs["quality"], "prediction_width_m": obs["width_m"],
                     "prediction_angle_rad": obs["angle"], "predicted_center_row": obs["row"],
                     "predicted_center_col": obs["col"]})

        target_xyz, target_dir, target_opening, last_shift = blend_grasp_target(
            None, (obs["xyz"], obs["direction"], obs["opening"])
        )
        try:
            angle_deg, orientation, _, _ = plan_grasp(
                p, robot, support, target_xyz, target_dir, obs["visible"], obs["camera"][0], approach_deg
            )
        except NoGraspCandidate:
            if attempt == 1:
                save_trial(output, {**placement, "prediction": first_prediction, "success": False,
                                    "outcome": "no_reachable_approach"})
                run.log({"grasp_success": 0})
                return False
            break

        phase = "pregrasp"
        standoff = PREGRASP_STANDOFF
        pregrasp_recorded = False
        pregrasp_steps = 0
        approach_stall = 0
        prev_approach_err = float("inf")
        best_quality = obs["quality"]

        for step_idx in range(MAX_SERVO_STEPS):
            total_servo_steps += 1
            if attempt == 1 and step_idx == 4 and disturb_m > 0 and disturbance is None:
                disturbance = apply_object_disturbance(p, obj, rng, disturb_m)

            if step_idx > 0:
                try:
                    new_obs = capture_and_predict(p, robot, model, device, preview=preview)
                    cam_dist = float(np.linalg.norm(new_obs["xyz"] - new_obs["camera"][0]))
                    in_frame = (16 <= new_obs["row"] <= SIZE - 16) and (16 <= new_obs["col"] <= SIZE - 16)
                    obs_shift = float(np.linalg.norm(new_obs["xyz"] - target_xyz))
                    old_occupied = target_has_foreground(new_obs["visible"], target_xyz)
                    quality_ok = new_obs["quality"] >= max(0.35, 0.65 * best_quality)
                    # Accept visual updates outside the near-field crop zone:
                    # - at mid/far range (>= 0.28 m) or when the old target became empty, track any confident shift;
                    # - at closer range (0.18–0.28 m), only refine smoothly with consistent predictions.
                    accept_update = cam_dist >= 0.18 and in_frame and (
                        (not old_occupied and new_obs["quality"] >= 0.30)
                        or (quality_ok and (cam_dist >= 0.28 or obs_shift <= 0.04))
                    )
                    if accept_update:
                        target_xyz, target_dir, target_opening, last_shift = blend_grasp_target(
                            (target_xyz, target_dir, target_opening),
                            (new_obs["xyz"], new_obs["direction"], new_obs["opening"]),
                        )
                        best_quality = max(best_quality, new_obs["quality"]) if old_occupied else new_obs["quality"]
                        visual_updates += 1
                        last_overlay = new_obs["overlay"]
                        if last_shift > 0.035 or not old_occupied:
                            try:
                                angle_deg, _, _, _ = plan_grasp(
                                    p, robot, support, target_xyz, target_dir,
                                    new_obs["visible"], new_obs["camera"][0], approach_deg,
                                )
                            except NoGraspCandidate:
                                pass
                            if phase == "approach" and standoff > 0.06:
                                phase = "pregrasp"
                                pregrasp_steps = 0
                                standoff = PREGRASP_STANDOFF
                    else:
                        last_shift = 0.0
                except NoGraspCandidate:
                    last_shift = 0.0
                    # If the object moved out of the close-range FOV, step back slightly to re-acquire it.
                    if phase == "pregrasp" and standoff < 0.32:
                        standoff = min(0.32, standoff + 0.04)

            cur_pos, cur_quat = p.getLinkState(robot, TCP, computeForwardKinematics=True)[4:6]
            cur_pos = np.asarray(cur_pos, dtype=float)
            cur_quat = np.asarray(cur_quat, dtype=float)
            axes = grasp_axes(target_dir, angle_deg)
            grasp_fwd = axes[:, 2]
            grasp_target = target_xyz + 0.026 * grasp_fwd
            grasp_target[2] = max(0.010, grasp_target[2])
            des_quat = Rotation.from_matrix(axes).as_quat()

            if phase == "pregrasp":
                pregrasp_steps += 1
                des_pos = grasp_target - PREGRASP_STANDOFF * grasp_fwd
                pos_err = float(np.linalg.norm(cur_pos - des_pos))
                rot_err = quaternion_angle(cur_quat, des_quat)
                if ((pos_err < 0.025 and rot_err < math.radians(10.0) and last_shift < 0.015)
                        or (pregrasp_steps >= 28 and pos_err < 0.045)):
                    phase = "approach"
                    standoff = PREGRASP_STANDOFF
                    approach_stall = 0
                    prev_approach_err = float("inf")
                    if not pregrasp_recorded:
                        states.append(record_pose(p, robot, obj, support, initial_z, run, "pregrasp", len(states)))
                        pregrasp_recorded = True
            if phase == "approach":
                cur_wp = grasp_target - standoff * grasp_fwd
                if float(np.linalg.norm(cur_pos - cur_wp)) < 0.028:
                    standoff = max(0.0, standoff - 0.016)
                des_pos = grasp_target - standoff * grasp_fwd
                pos_err = float(np.linalg.norm(cur_pos - grasp_target))
                rot_err = quaternion_angle(cur_quat, des_quat)
                if standoff <= 1e-6:
                    if pos_err < 0.016 and rot_err < math.radians(10.0):
                        last_grasp_target = grasp_target
                        last_orientation = des_quat
                        break
                    approach_stall = approach_stall + 1 if pos_err >= prev_approach_err - 0.002 else 0
                    prev_approach_err = min(prev_approach_err, pos_err)
                    if approach_stall >= 8:
                        last_grasp_target = grasp_target
                        last_orientation = des_quat
                        break

            next_pos, next_quat = step_pose_toward(cur_pos, cur_quat, des_pos, des_quat)
            if phase == "pregrasp" and float(np.linalg.norm(cur_pos[:2] - des_pos[:2])) > 0.05:
                next_pos[2] = max(next_pos[2], float(des_pos[2]), float(target_xyz[2]) + 0.14)

            joints = safe_ik_step(p, robot, support, next_pos, next_quat)
            if joints is None:
                joints = arm_ik(p, robot, next_pos, next_quat)
            set_fingers(p, robot, target_opening)
            set_arm(p, robot, joints)
            advance(SERVO_CONTROL_STEPS)
            last_grasp_target = grasp_target
            last_orientation = des_quat
            servo_trace.append({
                "attempt": attempt, "step": step_idx, "phase": phase,
                "standoff_m": round(standoff, 4), "target_xyz_m": [round(float(v), 4) for v in target_xyz],
                "target_shift_m": round(last_shift, 4), "visual_updates": visual_updates,
            })

        if not pregrasp_recorded:
            states.append(record_pose(p, robot, obj, support, initial_z, run, "pregrasp", len(states)))
        advance(40)
        states.append(record_pose(p, robot, obj, support, initial_z, run, "approach", len(states)))
        set_fingers(p, robot, 0)
        advance(160)
        states.append(record_pose(p, robot, obj, support, initial_z, run, "close", len(states)))

        # Proprioceptive closed-loop check: retry if fingers closed on empty air or TCP stalled short of target.
        tcp_pos_now = np.asarray(p.getLinkState(robot, TCP, computeForwardKinematics=True)[4], dtype=float)
        stalled_short = float(np.linalg.norm(tcp_pos_now - last_grasp_target)) > 0.035
        if (finger_opening(p, robot) >= MIN_GRASP_OPENING and not stalled_short) or attempt >= max_attempts:
            break
        set_fingers(p, robot, 0.08)
        advance(40)
        retract_axes = grasp_axes((1.0, 0.0), 0.0)
        retract_quat = Rotation.from_matrix(retract_axes).as_quat()
        retract_goal = np.array((0.55, 0.0, 0.60))
        for _ in range(20):
            cur_pos, cur_quat = p.getLinkState(robot, TCP, computeForwardKinematics=True)[4:6]
            next_pos, next_quat = step_pose_toward(cur_pos, cur_quat, retract_goal, retract_quat, max_trans=0.04)
            set_arm(p, robot, arm_ik(p, robot, next_pos, next_quat))
            advance(SERVO_CONTROL_STEPS)
            if float(np.linalg.norm(np.asarray(cur_pos) - retract_goal)) < 0.03:
                break
        advance(24)

    if last_overlay is not None:
        last_overlay.save(output / "closed-loop-final-grasp.png")
    placement["approach_deg"] = angle_deg
    first_prediction["tcp_target_xyz_m"] = last_grasp_target.tolist()
    first_prediction["tcp_target_quaternion"] = last_orientation.tolist()
    run.log({"executed_approach_deg": angle_deg, "visual_updates": visual_updates,
             "servo_steps": total_servo_steps, "grasp_attempts": attempt})
    extra = {
        "closed_loop": {
            "control_hz": 240 / SERVO_CONTROL_STEPS,
            "visual_updates": visual_updates,
            "servo_steps": total_servo_steps,
            "attempts": attempt,
            "final_target_xyz_m": last_grasp_target.tolist(),
            "trace_tail": servo_trace[-10:],
        }
    }
    if disturbance is not None:
        extra["disturbance"] = disturbance
    return finish_lift_and_hold(
        p, robot, obj, support, initial_z, run, output, placement, first_prediction,
        last_grasp_target, last_orientation, states, advance, preview, extra_summary=extra,
    )


def simulate(p, model, device, run, output, rng, gui=False, kind="cylinder", x=0.56, y=0.034,
             yaw=0.0, approach_deg=None, scene=None, preview=None, control_mode="closed-loop",
             max_attempts=DEFAULT_MAX_ATTEMPTS, disturb_m=0.0):
    if control_mode == "open-loop":
        return simulate_open_loop(
            p, model, device, run, output, rng, gui=gui, kind=kind, x=x, y=y, yaw=yaw,
            approach_deg=approach_deg, scene=scene, preview=preview, disturb_m=disturb_m,
        )
    return simulate_closed_loop(
        p, model, device, run, output, rng, gui=gui, kind=kind, x=x, y=y, yaw=yaw,
        approach_deg=approach_deg, scene=scene, preview=preview,
        max_attempts=max_attempts, disturb_m=disturb_m,
    )


def save_trial(output, summary):
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def random_placement(rng, kind):
    bounds = ((0.52, -0.03), (0.58, 0.03)) if kind in YCB_OBJECTS else (
        (0.495, -0.069), (0.605, 0.069))
    x, y = map(float, rng.uniform(*bounds))
    return x, y, float(rng.uniform(-math.pi, math.pi))


def interactive_grasp(p, model, device, run, output, kinds, seed, approach_deg, preview,
                      control_mode="closed-loop", max_attempts=DEFAULT_MAX_ATTEMPTS, disturb_m=0.0):
    rng = np.random.default_rng(seed)
    index = 0
    scene = None
    ready = False
    trial_index = 0
    hud = -1
    modes = (None, 25.0, -25.0, 40.0, -40.0, 0.0)
    print("In the 3D window: drag object (even during closed-loop servo!), Space = grasp, R = random reset, "
          "N = next object, T = change approach (Auto / +25 / -25 / +40 / -40 / 0), C = toggle closed/open-loop",
          flush=True)
    while p.isConnected():
        preview.update()
        keys = p.getKeyboardEvents()
        pressed = {key for key, state in keys.items() if state & p.KEY_WAS_TRIGGERED}
        next_object = ord("n") in pressed
        if next_object:
            index = (index + 1) % len(kinds)
        kind = kinds[index]
        if scene is None or next_object or ord("r") in pressed:
            x, y, yaw = random_placement(rng, kind)
            scene = setup_scene(p, True, kind, x, y, yaw, rng)
            ready = True
            hud = -1
        changed_approach = ord("t") in pressed
        if changed_approach:
            mode = modes.index(approach_deg) if approach_deg in modes else -1
            approach_deg = modes[(mode + 1) % len(modes)]
        changed_control = ord("c") in pressed
        if changed_control:
            control_mode = "open-loop" if control_mode == "closed-loop" else "closed-loop"
        if hud < 0 or changed_approach or changed_control:
            label = "Auto" if approach_deg is None else f"{approach_deg:+.0f} deg"
            hud = p.addUserDebugText(
                f"{kind} | {control_mode} | approach {label}\nSpace: grasp  R: shuffle  N: next  T: angle  C: mode",
                (0.58, -0.26, 0.01), textColorRGB=(0, 0, 0), textSize=1.1, replaceItemUniqueId=hud,
            )
        if ready and ord(" ") in pressed:
            trial_index += 1
            folder = output / f"{kind}-{trial_index:03}"
            folder.mkdir()
            # Settle the object after any mouse drag before taking the RGB-D frame.
            step(p, 60)
            simulate(p, model, device, run, folder, rng, gui=True, kind=kind, x=x, y=y, yaw=yaw,
                     approach_deg=approach_deg, scene=scene, preview=preview,
                     control_mode=control_mode, max_attempts=max_attempts, disturb_m=disturb_m)
            ready = False
            p.getKeyboardEvents()
        if ready:
            step(p, 4)
        time.sleep(1 / 60)
    return True


def evaluate_random_placements(p, model, device, run, output, gui, trials_per_object, seed, kinds, approach_deg,
                               preview, control_mode="closed-loop", max_attempts=DEFAULT_MAX_ATTEMPTS,
                               disturb_m=0.0):
    rng = np.random.default_rng(seed)
    trials = []
    counts = {}
    for kind in kinds:
        successes = 0
        for index in range(1, trials_per_object + 1):
            x, y, yaw = random_placement(rng, kind)
            folder = output / f"{kind}-{index:02}"
            folder.mkdir()
            print(f"Trial {kind}-{index:02}: x={x:.3f}, y={y:.3f}, yaw={math.degrees(yaw):.1f} deg", flush=True)
            success = simulate(p, model, device, run, folder, rng, gui=gui, kind=kind, x=x, y=y, yaw=yaw,
                               approach_deg=approach_deg, preview=preview, control_mode=control_mode,
                               max_attempts=max_attempts, disturb_m=disturb_m)
            trial = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            trials.append({"trial": folder.name, **trial})
            successes += int(success)
            print(f"Result {folder.name}: {trial['outcome']}", flush=True)
        counts[kind] = {"successes": successes, "trials": trials_per_object,
                        "success_rate": successes / trials_per_object}
    total = sum(item["successes"] for item in counts.values())
    object_properties = {
        kind: {"mesh": str(BOYA_MESH.relative_to(ROOT))}
        if kind == "boya_tower" else
        {"urdf": str((YCB_ROOT / f"{kind}.urdf").relative_to(ROOT))}
        if kind in YCB_OBJECTS else OBJECTS[kind] for kind in kinds
    }
    summary = {"seed": seed, "control_mode": control_mode, "disturb_m": disturb_m,
               "object_properties": object_properties, "per_object": counts,
               "successes": total, "trial_count": len(trials), "success_rate": total / len(trials),
               "trials": trials}
    save_trial(output, summary)
    run.log({"eval/success_rate": summary["success_rate"], "eval/trials": len(trials),
             "eval/successes": total,
             **{f"eval/{kind}_success_rate": value["success_rate"] for kind, value in counts.items()}})
    print(json.dumps({"per_object": counts, "successes": total, "trial_count": len(trials)}), flush=True)
    return total == len(trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Use CPU for GR-ConvNet inference")
    parser.add_argument("--gui", action="store_true", help="Show the live PyBullet desktop window")
    parser.add_argument("--gui-controls", action="store_true", help="Show Bullet's built-in diagnostic panels")
    parser.add_argument("--interactive", action="store_true", help="Adjust scene and trigger grasps in the GUI")
    parser.add_argument("--control-mode", choices=("closed-loop", "open-loop"), default="closed-loop",
                        help="Visual control mode: continuous 20 Hz closed-loop servoing or single-shot open-loop")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help="Maximum closed-loop grasp attempts with proprioceptive retry")
    parser.add_argument("--disturb-cm", type=float, default=0.0,
                        help="Optional mid-approach horizontal object shift (cm) to test closed-loop tracking")
    parser.add_argument("--approach-deg", type=float, help="Fix the gripper tilt from vertical; defaults to automatic")
    parser.add_argument("--checkpoint", type=Path, help="Local 224px RGB-D training checkpoint")
    parser.add_argument("--trials-per-object", type=int, help="Evaluate this many random placements per object")
    parser.add_argument("--seed", type=int, default=20261114, help="Random placement seed")
    parser.add_argument("--random-seed", action="store_true", help="Choose a different placement seed each run")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--object-set", choices=("primitive", "ycb"), default="primitive")
    selection.add_argument("--object", choices=(*OBJECTS, *YCB_OBJECTS), help="Only use this object")
    args = parser.parse_args()
    if args.interactive and (not args.gui or args.trials_per_object is not None):
        parser.error("--interactive requires --gui without --trials-per-object")
    if args.approach_deg is not None and not -55 <= args.approach_deg <= 55:
        parser.error("--approach-deg must be between -55 and 55")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if args.disturb_cm < 0:
        parser.error("--disturb-cm must be non-negative")
    disturb_m = args.disturb_cm / 100.0
    if args.random_seed:
        args.seed = int(np.random.SeedSequence().generate_state(1)[0])
    work_area = desktop_work_area() if args.gui else None
    if args.trials_per_object is not None and args.trials_per_object < 1:
        parser.error("--trials-per-object must be positive")
    if args.object:
        kinds = (args.object,)
    elif args.object_set == "ycb":
        kinds = YCB_OBJECTS
    else:
        kinds = tuple(OBJECTS) if args.trials_per_object or args.interactive else ("cylinder",)
    trials_per_object = args.trials_per_object or (
        1 if (args.object_set == "ycb" or args.random_seed) and not args.interactive else None)
    for kind in kinds:
        if kind == "boya_tower":
            if not BOYA_MESH.is_file():
                parser.error(f"Mesh asset missing for {kind}: {BOYA_MESH}")
        elif kind in YCB_OBJECTS and not (YCB_ROOT / f"{kind}.urdf").is_file():
            parser.error(f"YCB asset missing for {kind}: {YCB_ROOT}")

    import pybullet as p
    import torch

    os.environ.setdefault("TRACKIO_DIR", str(RESULTS / "trackio"))
    import trackio

    sys.path.insert(0, str(REPO))
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Pass --cpu to run inference on CPU.")
    device = torch.device("cpu" if args.cpu else "cuda")
    checkpoint = args.checkpoint or CHECKPOINT
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if args.checkpoint:
        from inference.models.grconvnet3 import GenerativeResnet

        state = torch.load(checkpoint, map_location=device, weights_only=True)
        config = state["config"]
        if config["input_size"] != SIZE:
            parser.error(f"The Panda camera pipeline expects a {SIZE}px model")
        model = GenerativeResnet(input_channels=4, dropout=False,
                                  channel_size=config.get("channel_size", 32)).to(device)
        model.load_state_dict(state["model"])
        model.eval()
    else:
        if checkpoint_hash != CHECKPOINT_SHA256:
            raise RuntimeError("Unexpected checkpoint SHA-256")
        model = torch.load(checkpoint, map_location=device, weights_only=False).eval()
    run_name = f"pybullet-panda-{datetime.now():%Y%m%d-%H%M%S-%f}"
    output = RESULTS / "pybullet" / run_name
    output.mkdir(parents=True)
    run = trackio.init(
        project="grconvnet-jacquard", group="pybullet-ycb" if any(k in YCB_OBJECTS for k in kinds)
        else "pybullet-random" if trials_per_object else "pybullet",
        name=run_name, embed=False,
        config={"device": str(device), "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash, "simulator": "PyBullet",
                "robot": "Franka Panda", "camera_mount": "wrist", "camera_size": SIZE, "camera_fov_deg": FOV,
                "control_mode": args.control_mode, "max_attempts": args.max_attempts,
                "disturb_cm": args.disturb_cm,
                "objects": list(kinds), "trials_per_object": trials_per_object, "seed": args.seed,
                "approach_deg": args.approach_deg, "interactive": args.interactive,
                "lift_threshold_m": LIFT_THRESHOLD},
    )
    preview = None
    if args.gui:
        connection = p.connect(p.GUI, options=desktop_window_options(work_area))
        if not args.gui_controls:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    else:
        connection = p.connect(p.DIRECT)
    try:
        try:
            if args.gui:
                preview = CameraPreview(work_area)
                move_bullet_window(work_area, preview)
            print(f"Trackio run: {run_name}\nOutput: {output}", flush=True)
            if args.interactive:
                success = interactive_grasp(p, model, device, run, output, kinds, args.seed, args.approach_deg,
                                            preview, control_mode=args.control_mode,
                                            max_attempts=args.max_attempts, disturb_m=disturb_m)
            elif trials_per_object:
                success = evaluate_random_placements(p, model, device, run, output, args.gui,
                                                     trials_per_object, args.seed, kinds, args.approach_deg, preview,
                                                     control_mode=args.control_mode,
                                                     max_attempts=args.max_attempts, disturb_m=disturb_m)
            else:
                success = simulate(p, model, device, run, output, np.random.default_rng(args.seed),
                                   gui=args.gui, kind=kinds[0], approach_deg=args.approach_deg, preview=preview,
                                   control_mode=args.control_mode, max_attempts=args.max_attempts,
                                   disturb_m=disturb_m)
        finally:
            run.finish()
        if args.gui and not args.interactive:
            print("Close the PyBullet window or press Ctrl+C here to exit.", flush=True)
            try:
                while p.isConnected(connection):
                    preview.update()
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        return 0 if success else 1
    finally:
        if preview is not None:
            preview.close()
        if p.isConnected(connection):
            p.disconnect(connection)


if __name__ == "__main__":
    raise SystemExit(main())

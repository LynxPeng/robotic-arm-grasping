"""Interactive & batch 6-DoF GA-DDPG grasping demonstration in PyBullet."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RL_DIR = Path(__file__).resolve().parent
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))

DEFAULT_CHECKPOINT = (
    ROOT / "robotic-grasping/results/qdg500-omg-ga-v5-retract-b32-lr5e5-bc20k-20261118/best.pt"
)
DEFAULT_DATASET = ROOT / "robotic-grasping/results/qdg500-demo-w16-20261116-210216"
DEFAULT_EVAL_LOG = (
    ROOT / "robotic-grasping/results/qdg500-omg-ga-v5-retract-b32-lr5e5-bc20k-20261118/eval-252000.json"
)
DEFAULT_CAM = (0.62, 42.0, -26.0, (0.52, 0.0, 0.10))
OBJECT_COLORS = {
    "train": (0.88, 0.58, 0.18, 1.0),    # Warm Amber
    "holdout": (0.18, 0.49, 0.72, 1.0),  # Steel Blue
    "custom": (0.16, 0.62, 0.56, 1.0),   # Teal Jade
}
CUSTOM_KEY = "pku/boya_tower"
CUSTOM_XY = (0.56, 0.02)  # Center of the tabletop workspace.
CUSTOM_SEED = 7           # Fixed object yaw and fixed, known-good arm start pose.
HOLDOUT_KEYS = set()


def desktop_work_area():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        work_area = wintypes.RECT()
        if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work_area), 0):
            return (
                work_area.left,
                work_area.top,
                work_area.right - work_area.left,
                work_area.bottom - work_area.top,
            )
    return 0, 0, 1366, 864


def preview_width(work_area):
    return min(440, max(340, work_area[2] // 3))


def desktop_window_options(work_area):
    _, _, width, height = work_area
    win_w = min(1920, width - preview_width(work_area) - 96)
    win_h = min(1080, height - 80)
    return f"--width={win_w} --height={win_h}"


def disable_ime_for_window(hwnd):
    if sys.platform != "win32" or not hwnd:
        return
    import ctypes
    from ctypes import wintypes

    imm32 = ctypes.WinDLL("imm32", use_last_error=True)
    imm32.ImmAssociateContext.argtypes = (wintypes.HWND, wintypes.HANDLE)
    imm32.ImmAssociateContext.restype = wintypes.HANDLE
    imm32.ImmAssociateContext(hwnd, None)


def disable_process_ime():
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    imm32 = ctypes.WinDLL("imm32", use_last_error=True)
    imm32.ImmDisableIME.argtypes = (wintypes.DWORD,)
    imm32.ImmDisableIME.restype = wintypes.BOOL
    imm32.ImmDisableIME(0xFFFFFFFF)


def move_bullet_window(work_area, monitor):
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32")
    user32.SetWindowPos.argtypes = (
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, wintypes.UINT,
    )
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def position_window(window, _):
        if not user32.IsWindowVisible(window):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
        if pid.value != os.getpid():
            return True
        disable_ime_for_window(window)
        title = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(window, title, len(title))
        if "bullet" in title.value.lower() or "example" in title.value.lower():
            _, top, _, _ = work_area
            panel_right = monitor.root.winfo_rootx() + monitor.root.winfo_width()
            user32.SetWindowPos(window, None, panel_right + 20, top + 8, 0, 0, 0x0001 | 0x0004 | 0x0010)
        return True

    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.EnumWindows.argtypes = (callback_type, wintypes.LPARAM)
    user32.EnumWindows(position_window, 0)


class MiddleOrbitCamera:
    """Allow plain middle-mouse drag to orbit the PyBullet debug camera."""

    def __init__(self, p, client):
        self.p = p
        self.client = client
        self.dragging = False
        self.last_xy = None

    def poll(self):
        events = self.p.getMouseEvents(physicsClientId=self.client)
        dx_total = 0.0
        dy_total = 0.0
        for ev in events:
            ev_type, mx, my, btn, state = ev
            if ev_type == 2 and btn == 1:  # Middle mouse button
                if state & (self.p.KEY_WAS_TRIGGERED | self.p.KEY_IS_DOWN):
                    if not self.dragging:
                        self.dragging = True
                        self.last_xy = (mx, my)
                if state & self.p.KEY_WAS_RELEASED:
                    self.dragging = False
                    self.last_xy = None
            elif ev_type == 1 and self.dragging:
                if self.last_xy is not None:
                    dx_total += mx - self.last_xy[0]
                    dy_total += my - self.last_xy[1]
                self.last_xy = (mx, my)
        if dx_total or dy_total:
            cam = self.p.getDebugVisualizerCamera(physicsClientId=self.client)
            yaw, pitch, dist, target = cam[8], cam[9], cam[10], cam[11]
            new_yaw = yaw - dx_total * 0.35
            new_pitch = float(np.clip(pitch - dy_total * 0.35, -89.0, 89.0))
            self.p.resetDebugVisualizerCamera(dist, new_yaw, new_pitch, target, physicsClientId=self.client)


class PolicyMonitor:
    """Side panel showing segmented wrist view and real-time GA-DDPG telemetry."""

    def __init__(self, work_area):
        import tkinter as tk

        left, top, _, _ = work_area
        width = preview_width(work_area)
        self.img_size = width - 32
        self.root = tk.Tk()
        self.root.title("GA-DDPG 6-DoF Monitor")
        self.root.geometry(f"{width}x{width + 250}+{left + 8}+{top + 8}")
        self.root.configure(bg="#181a1f")
        self.root.protocol("WM_DELETE_WINDOW", self.root.withdraw)

        self.image_label = tk.Label(
            self.root, text="Wrist Camera (96x96 Segmented)", bg="#181a1f", fg="#e6e6e6"
        )
        self.image_label.pack(padx=12, pady=(10, 6))

        self.info_var = tk.StringVar(value="[加载中] 正在初始化...")
        self.info_label = tk.Label(
            self.root,
            textvariable=self.info_var,
            justify="left",
            anchor="nw",
            font=("Consolas", 10),
            bg="#22252c",
            fg="#f0f2f5",
            padx=10,
            pady=10,
            width=48,
            height=11,
        )
        self.info_label.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self.reset_view_requested = False
        self.switch_split_requested = None
        btn_row = tk.Frame(self.root, bg="#181a1f")
        btn_row.pack(fill="x", padx=12, pady=(0, 10))
        self.reset_btn = tk.Button(
            btn_row,
            text="默认视角 [V]",
            command=self._request_reset_view,
            bg="#2f3542",
            fg="#f0f2f5",
            activebackground="#404859",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            pady=5,
        )
        self.reset_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.other_btn = tk.Button(
            btn_row,
            text="其他物品 [O]",
            command=lambda: self._request_split("custom"),
            bg="#7a1c1c",
            fg="#f0f2f5",
            activebackground="#9c2626",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            pady=5,
        )
        self.other_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.photo = None
        self.update()
        disable_ime_for_window(self.root.winfo_id())

    def _request_reset_view(self):
        self.reset_view_requested = True

    def _request_split(self, target_split):
        self.switch_split_requested = target_split

    def show_step(self, rgb_overlay, telemetry_text):
        from PIL import Image, ImageTk

        resized = rgb_overlay.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(resized)
        self.image_label.configure(image=self.photo, text="")
        self.info_var.set(telemetry_text)
        self.root.deiconify()
        self.update()

    def set_text(self, telemetry_text):
        self.info_var.set(telemetry_text)
        self.update()

    def update(self):
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        self.root.destroy()


def reset_default_camera(p, client):
    dist, yaw, pitch, target = DEFAULT_CAM
    p.resetDebugVisualizerCamera(dist, yaw, pitch, target, physicsClientId=client)


CUSTOM_TEXTURES = {
    "pku/boya_tower": ROOT / "assets" / "boya_tower.png",
    "pku/blue_fat_fish": ROOT / "assets" / "blue_fat_fish.png",
    "pku/gta6_case": ROOT / "assets" / "gta6_case.png",
}
_TEXTURE_CACHE = {}


def style_gui_scene(env):
    """Keep original floor texture, apply object textures/shading, and sharpen shadows."""
    import pybullet as p

    tex_path = CUSTOM_TEXTURES.get(env.kind)
    if tex_path is not None and tex_path.is_file():
        cache_key = (env.client, str(tex_path))
        if cache_key not in _TEXTURE_CACHE:
            _TEXTURE_CACHE[cache_key] = p.loadTexture(str(tex_path), physicsClientId=env.client)
        p.changeVisualShape(
            env.obj,
            -1,
            rgbaColor=(1.0, 1.0, 1.0, 1.0),
            textureUniqueId=_TEXTURE_CACHE[cache_key],
            specularColor=(0.24, 0.24, 0.24),
            physicsClientId=env.client,
        )
    else:
        if env.kind.startswith("pku/"):
            color = OBJECT_COLORS["custom"]
        elif env.kind in HOLDOUT_KEYS:
            color = OBJECT_COLORS["holdout"]
        else:
            color = OBJECT_COLORS["train"]
        p.changeVisualShape(
            env.obj,
            -1,
            rgbaColor=color,
            specularColor=(0.24, 0.24, 0.24),
            physicsClientId=env.client,
        )
    p.changeVisualShape(
        env.support,
        -1,
        specularColor=(0.0, 0.0, 0.0),
        physicsClientId=env.client,
    )
    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI,
        0,
        lightPosition=(1.6, -1.2, 2.2),
        physicsClientId=env.client,
    )
    p.configureDebugVisualizer(
        p.COV_ENABLE_SHADOWS,
        1,
        shadowMapResolution=8192,
        shadowMapWorldSize=2,
        shadowMapIntensity=0.25,
        physicsClientId=env.client,
    )


def render_wrist_overlay(env):
    """Render the 96x96 wrist camera RGB and highlight segmented target pixels."""
    import pybullet as p
    from PIL import Image
    from ga_env import CAMERA_OFFSET, IMAGE_SIZE, matrix

    pos, quat = env.pose()
    rot = matrix(quat)
    eye = pos + rot @ CAMERA_OFFSET
    forward = rot @ np.array((-0.35, 0, math.sqrt(1 - 0.35**2)))
    up = -rot[:, 1]
    view = p.computeViewMatrix(eye, eye + forward, up)
    _, _, rgba, _, mask = p.getCameraImage(
        IMAGE_SIZE,
        IMAGE_SIZE,
        view,
        env.projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=env.client,
    )
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE, 4)[:, :, :3].copy()
    mask_arr = np.asarray(mask).reshape(IMAGE_SIZE, IMAGE_SIZE)
    target_px = (mask_arr >= 0) & ((mask_arr & 0xFFFFFF) == env.obj)
    if target_px.any():
        rgb[target_px] = np.clip(
            0.55 * rgb[target_px] + 0.45 * np.array((0, 220, 255), dtype=np.float32), 0, 255
        ).astype(np.uint8)
    return Image.fromarray(rgb), float(target_px.mean())


def load_custom_assets():
    from qdg_env import MeshAsset

    assets = {}
    inertia = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    boya_mesh = ROOT / "assets" / "boya_tower.obj"
    if boya_mesh.is_file():
        assets["pku/boya_tower"] = MeshAsset(
            key="pku/boya_tower",
            mesh=boya_mesh,
            bounds=np.array([[-0.0134, -0.0134, -0.045], [0.0134, 0.0134, 0.045]], dtype=np.float64),
            inertia_origin=inertia,
            mass=0.060,
            grasps=None,
        )
    fish_mesh = ROOT / "assets" / "blue_fat_fish.obj"
    if fish_mesh.is_file():
        assets["pku/blue_fat_fish"] = MeshAsset(
            key="pku/blue_fat_fish",
            mesh=fish_mesh,
            bounds=np.array([[-0.0344, -0.0408, -0.045], [0.0344, 0.0408, 0.045]], dtype=np.float64),
            inertia_origin=inertia,
            mass=0.060,
            grasps=None,
        )
    gta_mesh = ROOT / "assets" / "gta6_case.obj"
    if gta_mesh.is_file():
        assets["pku/gta6_case"] = MeshAsset(
            key="pku/gta6_case",
            mesh=gta_mesh,
            bounds=np.array([[-0.0065, -0.034, -0.046], [0.0065, 0.034, 0.046]], dtype=np.float64),
            inertia_origin=inertia,
            mass=0.055,
            grasps=None,
        )
    return assets


def select_validation_keys(assets, eval_log_path, split):
    if split == "train" and eval_log_path.is_file():
        rows = json.loads(eval_log_path.read_text(encoding="utf-8"))
        seen = []
        for row in rows:
            key = row["kind"]
            if key in assets and key not in seen:
                seen.append(key)
        if seen:
            return seen
    return sorted(assets.keys())


@contextmanager
def realtime_physics(p, monitor=None, orbit=None, enabled=True, speed=0.8):
    """Pace PyBullet physics substeps smoothly in real time during GUI playback."""
    if not enabled or speed <= 0:
        yield
        return
    orig_step = p.stepSimulation
    step_dt = (1.0 / 240.0) / speed
    state = {"next_tick": time.perf_counter(), "count": 0}

    def paced_step(*args, **kwargs):
        res = orig_step(*args, **kwargs)
        state["count"] += 1
        now = time.perf_counter()
        if now - state["next_tick"] > 0.05:
            # Never burst/fast-forward to catch up after inference or rendering.
            state["next_tick"] = now
        state["next_tick"] += step_dt
        delay = state["next_tick"] - now
        if delay > 0:
            time.sleep(delay)
        if orbit is not None and state["count"] % 4 == 0:
            orbit.poll()
        if monitor is not None and state["count"] % 8 == 0:
            monitor.update()
            if monitor.reset_view_requested:
                monitor.reset_view_requested = False
                reset_default_camera(p, orbit.client) if orbit is not None else None
        return res

    p.stepSimulation = paced_step
    try:
        yield
    finally:
        p.stepSimulation = orig_step


def run_episode(env, actor, critic, device, obs, *, kind, split, horizon=50, gui=False,
                monitor=None, orbit=None, speed=0.8):
    import pybullet as p
    import torch
    from ga_env import GOAL_SCALE, POS_SCALE, ROT_SCALE, matrix

    info = {}
    q_history = []
    t = 0
    with realtime_physics(p, monitor=monitor, orbit=orbit, enabled=gui, speed=speed):
        for t in range(1, horizon + 1):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.inference_mode():
                action_t, goal_t = actor(obs_t)
                q1_t, q2_t, _ = critic(obs_t, action_t)
            action = action_t.squeeze(0).cpu().numpy()
            goal = goal_t.squeeze(0).cpu().numpy()
            q1, q2 = float(q1_t.item()), float(q2_t.item())
            q_min = min(q1, q2)
            q_history.append(q_min)

            if gui and monitor is not None:
                dpos_cm = np.linalg.norm(action[:3]) * POS_SCALE * 100.0
                drot_deg = math.degrees(np.linalg.norm(action[3:6]) * ROT_SCALE)
                goal_dist_cm = np.linalg.norm(goal[:3]) * GOAL_SCALE * 100.0
                z_axis = matrix(env.pose()[1])[:, 2]
                tilt_deg = math.degrees(math.acos(np.clip(-z_axis[2], -1.0, 1.0)))
                overlay, mask_cov = render_wrist_overlay(env)
                text = (
                    f"Object   : {kind} ({split})\n"
                    f"Step     : {t:02d} / {horizon}   [6D APPROACH]\n"
                    f"Action   : dpos={dpos_cm:4.2f} cm  drot={drot_deg:4.1f} deg\n"
                    f"Critic Q : min={q_min:+.3f} (Q1={q1:+.3f}, Q2={q2:+.3f})\n"
                    f"Goal-Aux : dist={goal_dist_cm:4.1f} cm\n"
                    f"Wrist Cam: tilt={tilt_deg:4.1f} deg  mask={mask_cov * 100:4.1f}%\n"
                    f"Controls : [Space] next/run  [R] reset  [N] obj\n"
                    f"           [H] split  [A] auto"
                )
                monitor.show_step(overlay, text)

            obs, reward, term, trunc, info = env.step(action)

            if term or trunc:
                break

    success = bool(info.get("success", False))
    lift_cm = float(info.get("lift", 0.0)) * 100.0
    tilt_close = info.get("tilt_at_close_deg")
    tilt_str = f"{tilt_close:.1f} deg" if tilt_close is not None else "N/A"
    result = {
        "kind": kind,
        "split": split,
        "success": success,
        "grasped": bool(info.get("grasped", False)),
        "lift_cm": round(lift_cm, 2),
        "steps": t,
        "tilt_at_close_deg": None if tilt_close is None else round(float(tilt_close), 2),
        "max_approach_tilt_deg": round(float(info.get("max_approach_tilt_deg", 0.0)), 2),
        "visible_fraction": round(float(info.get("visible_fraction", 0.0)), 3),
        "final_q": None if not q_history else round(q_history[-1], 3),
    }
    if gui and monitor is not None:
        verdict = "SUCCESS" if success else "FAIL"
        overlay, mask_cov = render_wrist_overlay(env)
        text = (
            f"Object   : {kind} ({split})\n"
            f"Result   : {verdict}  (steps={t})\n"
            f"Lift     : {lift_cm:5.2f} cm   (target >= 10.0 cm)\n"
            f"CloseTilt: {tilt_str}   max={result['max_approach_tilt_deg']:.1f} deg\n"
            f"Visible  : {result['visible_fraction'] * 100:5.1f}%   Last Q={result['final_q']}\n"
            f"Next     : [Space] 新位姿再抓一次  [R] 重置  [N] 下个物体\n"
            f"           [M] 切换模式  [O] 其他物品  [A] 自动"
        )
        monitor.show_step(overlay, text)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("train", "holdout", "custom"), default="train")
    parser.add_argument("--object", help="Specific QDG500 object key (e.g. egad/egad_J00_1)")
    parser.add_argument("--episodes", type=int, help="Run N episodes automatically and exit")
    parser.add_argument("--seed", type=int, default=300001)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--cpu", action="store_true", help="Run PointNet++ inference on CPU")
    parser.add_argument("--no-gui", action="store_true", help="Run headless evaluation without windows")
    parser.add_argument("--auto", action="store_true", help="Auto-advance episodes in GUI mode")
    parser.add_argument("--speed", type=float, default=0.8, help="GUI playback speed multiplier (default: 0.8x)")
    args = parser.parse_args()

    gui = not args.no_gui
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not args.dataset.is_dir():
        parser.error(f"QDG500 dataset not found: {args.dataset}")

    import pybullet as p
    import torch
    from ga_model import Actor, Critic
    from qdg_env import QDGPointGraspEnv, load_manifest

    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --cpu to run on CPU.")
    device = torch.device("cpu" if args.cpu else "cuda")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    actor = Actor().to(device).eval()
    critic = Critic().to(device).eval()
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])

    splits = {
        "train": load_manifest(args.dataset, "train"),
        "holdout": load_manifest(args.dataset, "holdout"),
        "custom": load_custom_assets(),
    }
    split_keys = {
        "train": select_validation_keys(splits["train"], DEFAULT_EVAL_LOG, "train"),
        "holdout": sorted(splits["holdout"].keys()),
        "custom": sorted(splits["custom"].keys()),
    }

    split = args.split
    if args.object:
        if args.object in splits["train"]:
            split = "train"
        elif args.object in splits["holdout"]:
            split = "holdout"
        elif args.object in splits["custom"]:
            split = "custom"
        else:
            parser.error(f"Object {args.object!r} not found in train, holdout, or custom assets.")
    if not splits[split]:
        parser.error(f"No assets available for split {split!r}.")

    work_area = desktop_work_area() if gui else None
    if gui:
        disable_process_ime()
    monitor = PolicyMonitor(work_area) if gui else None

    HOLDOUT_KEYS.update(splits["holdout"].keys())

    class DemoQDGEnv(QDGPointGraspEnv):
        def _connect(self, use_gui):
            if use_gui and work_area is not None:
                return p.connect(p.GUI, options=desktop_window_options(work_area))
            return super()._connect(use_gui)

        def _is_custom(self):
            asset = getattr(self, "asset", None)
            return asset is not None and asset.key.startswith("pku/")

        def _with_fixed_rng(self, func, *args):
            saved = self.rng
            self.rng = np.random.default_rng(CUSTOM_SEED)
            try:
                return func(*args)
            finally:
                self.rng = saved

        def _spawn_object(self, x, y):
            if self._is_custom():
                if self.asset.key == "pku/gta6_case":
                    # Face the front cover (+X) toward the default camera (yaw=42 deg -> +X, -Y).
                    from ga_env import matrix
                    self.kind = self.asset.key
                    yaw = p.getQuaternionFromEuler((0, 0, math.radians(-42.0)))
                    xy = np.array((CUSTOM_XY[0], CUSTOM_XY[1], 0.0)) - matrix(yaw) @ np.r_[self.mesh_center[:2], 0.0]
                    xy[2] = -self.asset.bounds[0, 2] + 0.005
                    collision = p.createCollisionShape(p.GEOM_MESH, fileName=str(self.asset.mesh))
                    visual = p.createVisualShape(p.GEOM_MESH, fileName=str(self.asset.mesh),
                                                 rgbaColor=(1.0, 1.0, 1.0, 1.0))
                    return p.createMultiBody(
                        baseMass=self.asset.mass, baseCollisionShapeIndex=collision,
                        baseVisualShapeIndex=visual, basePosition=xy, baseOrientation=yaw,
                        baseInertialFramePosition=self.asset.inertia_origin[:3],
                        baseInertialFrameOrientation=self.asset.inertia_origin[3:],
                    )
                return self._with_fixed_rng(super()._spawn_object, *CUSTOM_XY)
            return super()._spawn_object(x, y)

        def _randomize_initial_pose(self):
            if self._is_custom():
                return self._with_fixed_rng(super()._randomize_initial_pose)
            return super()._randomize_initial_pose()

    env = DemoQDGEnv(assets=splits[split], gui=gui, max_steps=args.horizon, renderer="tiny")
    orbit = MiddleOrbitCamera(p, env.client) if gui else None
    if gui:
        style_gui_scene(env)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1, physicsClientId=env.client)
        reset_default_camera(p, env.client)
        move_bullet_window(work_area, monitor)

    try:
        if args.episodes is not None:
            keys = [args.object] if args.object else split_keys[split]
            results = []
            for idx in range(args.episodes):
                kind = keys[idx % len(keys)]
                obs = env.reset(seed=args.seed + idx, kind=kind)
                if gui:
                    style_gui_scene(env)
                res = run_episode(
                    env, actor, critic, device, obs, kind=kind, split=split,
                    horizon=args.horizon, gui=gui, monitor=monitor, speed=args.speed,
                )
                results.append(res)
                print(json.dumps(res), flush=True)
            successes = sum(int(r["success"]) for r in results)
            summary = {
                "split": split,
                "episodes": len(results),
                "successes": successes,
                "success_rate": successes / max(1, len(results)),
            }
            print(json.dumps(summary), flush=True)
            return 0 if successes == len(results) else 1

        # Interactive GUI loop
        keys = split_keys[split]
        obj_idx = keys.index(args.object) if args.object in keys else 0
        ep_seed = args.seed
        auto_run = args.auto
        split_order = [s for s in ("train", "holdout", "custom") if splits[s]]
        split_names = {"train": "验证", "holdout": "留出", "custom": "其他物品"}

        def load_scene(seed, kind, note):
            """Blocking scene (re)build with an explicit on-screen busy message."""
            monitor.set_text(f"[加载中] {note}\n正在构建场景与随机化机械臂起点...\n请稍候, 期间按键不会响应")
            new_obs = env.reset(seed=seed, kind=kind)
            style_gui_scene(env)
            overlay, _ = render_wrist_overlay(env)
            monitor.show_step(
                overlay,
                f"[就绪] {split_names[split]}模式 | {kind}\n按 Space 开始抓取\n"
                f"R:重置位姿  N:下个物体  M:切换模式  O:其他物品  V:视角  A:自动",
            )
            p.getKeyboardEvents(physicsClientId=env.client)  # Drop keys queued while busy.
            return new_obs

        from ga_env import OBS_DIM

        monitor.set_text("[预热中] 正在预热 GPU 网络(首次推理较慢)...")
        with torch.inference_mode():
            warm = torch.zeros(1, OBS_DIM, device=device)
            actor(warm)
            critic(warm, torch.zeros(1, 7, device=device))
        obs = load_scene(ep_seed, keys[obj_idx], "首个场景")
        ready = True
        last_hud = None
        hud = -1
        print(
            "GA-DDPG GUI ready: Space=run, R=new pose, N=next object, M/H=cycle mode, O=other items, V=view, A=auto",
            flush=True,
        )
        while p.isConnected(env.client):
            monitor.update()
            orbit.poll()
            kind = keys[obj_idx]
            hud_text = (
                f"GA-DDPG ({split_names[split]}) | {kind}\n"
                f"Space: run  R: reset  N: next  M: mode  O: other  V: view  A: auto({'ON' if auto_run else 'OFF'})"
            )
            if hud_text != last_hud:
                hud = p.addUserDebugText(
                    hud_text, (0.40, -0.22, 0.16), textColorRGB=(1, 1, 1), textSize=1.1,
                    replaceItemUniqueId=hud, physicsClientId=env.client,
                )
                last_hud = hud_text
            keys_evt = p.getKeyboardEvents(physicsClientId=env.client)
            pressed = {k for k, st in keys_evt.items() if st & p.KEY_WAS_TRIGGERED}
            if ord("q") in pressed:
                break
            if ord("v") in pressed or monitor.reset_view_requested:
                monitor.reset_view_requested = False
                reset_default_camera(p, env.client)
            if ord("a") in pressed:
                auto_run = not auto_run
            next_split = None
            if monitor.switch_split_requested is not None:
                req_split = monitor.switch_split_requested
                monitor.switch_split_requested = None
                if splits.get(req_split):
                    next_split = req_split
            elif ord("o") in pressed and splits.get("custom"):
                next_split = "custom" if split != "custom" else "train"
            elif ord("h") in pressed or ord("m") in pressed:
                next_split = split_order[(split_order.index(split) + 1) % len(split_order)]

            if next_split is not None and next_split != split:
                split = next_split
                env.assets = splits[split]
                env.asset_keys = tuple(splits[split])
                keys = split_keys[split]
                obj_idx = 0
                ep_seed += 1
                obs = load_scene(ep_seed, keys[obj_idx], f"切换到{split_names[split]}模式")
                ready = True
            elif ord("n") in pressed:
                obj_idx = (obj_idx + 1) % len(keys)
                ep_seed += 1
                obs = load_scene(ep_seed, keys[obj_idx], f"加载 {keys[obj_idx]}")
                ready = True
            elif ord("r") in pressed:
                ep_seed += 1
                obs = load_scene(ep_seed, kind, f"重置 {kind}")
                ready = True
            elif auto_run or ord(" ") in pressed:
                if not ready:
                    # Previous episode already finished: set up a fresh pose automatically.
                    ep_seed += 1
                    if auto_run:
                        obj_idx = (obj_idx + 1) % len(keys)
                        kind = keys[obj_idx]
                    obs = load_scene(ep_seed, kind, f"准备下一次抓取 {kind}")
                res = run_episode(
                    env, actor, critic, device, obs, kind=kind, split=split,
                    horizon=args.horizon, gui=True, monitor=monitor, orbit=orbit, speed=args.speed,
                )
                print(json.dumps(res), flush=True)
                ready = False
                p.getKeyboardEvents(physicsClientId=env.client)
                if auto_run:
                    time.sleep(0.8)
            time.sleep(1.0 / 60.0)
        return 0
    finally:
        if monitor is not None:
            monitor.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

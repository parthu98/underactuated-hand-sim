#!/home/namit/iitgn/mujoco_env/bin/python3
"""
interactive_gripper.py
======================
Two-finger tendon-driven gripper with a pop-up control panel.

Run it:

    python interactive_gripper.py

A Tk control panel opens alongside the MuJoCo viewer. Everything is live — no
restarts, no recompiles:

  * Movement mode .... "Simultaneous (link fingers)" couples both ΔL sliders;
                       uncheck to drive Finger A and Finger B alone.
  * Finger drive ..... per-finger tendon pull ΔL [mm], slider OR exact entry.
                       More ΔL → more flexion / grip (the string is stiff — creep it).
  * Joint stiffness .. live MCP/PIP/DIP sliders, shared by both fingers, for
                       on-the-go testing of the stiffness ratio. NOT saved —
                       each launch loads MCP/PIP/DIP_STIFFNESS from config.py.
  * Aperture ......... centre-to-centre gap between the fingers.
  * Probe object ..... enable/disable, shape (box / cylinder / sphere), size,
                       and DEPTH between the fingers — slide it in for an
                       enveloping grasp, out for a pinch grasp.
  * Scene ............ gravity toggle, Reset, Quit.

The object is WELDED in space (static), so the fingers press against an
immovable target — the live "grip force" readout is your handle on how the
joint-stiffness ratio affects gripping ability.

Physics is the tested high-fidelity finger (constant sheath moment arm,
constant-arm fixed tendon, hard joint limits) driven by ΔL exactly as in the
validation suite (tendon_lengthspring = L_rest - ΔL). Rigid contacts + a small
timestep keep it stable against the static object. See gripper/build_gripper.py.

Architecture: the Tk panel runs on the main thread; the MuJoCo viewer + physics
loop runs on a daemon thread. They share one lock-guarded GripperState.
"""
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import mujoco
import mujoco.viewer

import tkinter as tk
import customtkinter as ctk
from collections import deque

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure                                   # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import config                # noqa: E402  — single source of truth
import build_gripper as bg   # noqa: E402

MAX_PULL_MM = config.GRIPPER_MAX_PULL_MM
STIFF_MAX = config.GRIPPER_STIFFNESS_MAX
APER_MIN_MM, APER_MAX_MM = (1000 * x for x in config.GRIPPER_SEPARATION_RANGE)
DEPTH_MIN_MM, DEPTH_MAX_MM = (1000 * x for x in config.GRIPPER_OBJECT_DEPTH_RANGE)
SIZE_MIN_MM, SIZE_MAX_MM = config.GRIPPER_OBJECT_SIZE_RANGE_MM
LEN_MIN_MM, LEN_MAX_MM = config.GRIPPER_OBJECT_LENGTH_RANGE_MM
SHAPES = ("box", "cylinder", "sphere")
STIFF_DEFAULT = {"mcp": config.MCP_STIFFNESS, "pip": config.PIP_STIFFNESS,
                 "dip": config.DIP_STIFFNESS}
SMOOTH = 0.08                # ΔL low-pass (smooth, gentle closing)


# =====================================================================
#  Shared state (UI thread writes, sim thread reads)
# =====================================================================
class GripperState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.reset = False
        self.link = True
        self.dL = {"a": 0.0, "b": 0.0}                  # m (tendon pull)
        self.stiffness = dict(STIFF_DEFAULT)            # N·m/rad, shared by both fingers
        self.aperture = config.GRIPPER_SEPARATION       # m
        self.obj_enabled = config.GRIPPER_OBJECT_ENABLED
        self.obj_shape = config.GRIPPER_OBJECT_SHAPE
        self.obj_size_mm = config.GRIPPER_OBJECT_SIZE_MM
        self.obj_len_mm = config.GRIPPER_OBJECT_LENGTH_MM
        self.obj_depth = config.GRIPPER_OBJECT_DEPTH_Z  # m
        self.gravity = config.GRIPPER_GRAVITY_DEFAULT
        self.readout = {}

    def snapshot(self):
        with self.lock:
            snap = SimpleNamespace(
                reset=self.reset, link=self.link, dL=dict(self.dL),
                stiffness=dict(self.stiffness),
                aperture=self.aperture, obj_enabled=self.obj_enabled,
                obj_shape=self.obj_shape, obj_size_mm=self.obj_size_mm,
                obj_len_mm=self.obj_len_mm, obj_depth=self.obj_depth,
                gravity=self.gravity)
            self.reset = False
            return snap


# =====================================================================
#  Simulation side
# =====================================================================
def _make_ids(model):
    def jid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    def tid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, n)
    def gid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    def bid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

    ids = SimpleNamespace()
    ids.tendon = {f: tid(f"{f}_flexor") for f in "ab"}
    ids.jnt = {f: {n: jid(f"{f}_{n}") for n in bg.JOINT_NAMES} for f in "ab"}
    ids.base = {f: bid(f"{f}_base") for f in "ab"}
    ids.base_xyz = {f: model.body_pos[ids.base[f]].copy() for f in "ab"}
    ids.qadr = {f: {n: model.jnt_qposadr[ids.jnt[f][n]]
                    for n in bg.JOINT_NAMES} for f in "ab"}
    ids.obj_body = bid("object")
    ids.geom = {"box": gid("obj_box"), "cylinder": gid("obj_cyl"),
                "sphere": gid("obj_sph")}
    return ids


def _apply(model, data, ids, cur, snap):
    """Push the (snapshotted) panel state into the live model. Returns True if a
    reset was performed (caller should re-forward / skip a step's history)."""
    did_reset = False
    if snap.reset:
        mujoco.mj_resetData(model, data)
        cur["a"] = cur["b"] = 0.0
        did_reset = True

    # gravity
    model.opt.gravity[:] = (0, 0, -9.81) if snap.gravity else (0, 0, 0)

    # aperture: slide the two static finger bases symmetrically along ±Y
    for f, sgn in (("a", +1.0), ("b", -1.0)):
        p = ids.base_xyz[f].copy()
        p[1] = sgn * snap.aperture / 2.0
        model.body_pos[ids.base[f]] = p

    # probe object: depth (Z), shape visibility/contact, and size
    model.body_pos[ids.obj_body] = (0.0, 0.0, snap.obj_depth)
    active = snap.obj_shape if snap.obj_enabled else None
    s_m, L_m = snap.obj_size_mm / 1000.0, snap.obj_len_mm / 1000.0
    for shp, g in ids.geom.items():
        on = (shp == active)
        model.geom_rgba[g, 3] = 1.0 if on else 0.0
        model.geom_contype[g] = 1 if on else 0
        model.geom_conaffinity[g] = 1 if on else 0
        if on:
            if shp == "box":
                model.geom_size[g] = (s_m, s_m, L_m)
            elif shp == "cylinder":
                model.geom_size[g] = (s_m, L_m, 0.0)
            else:                       # sphere
                model.geom_size[g] = (s_m, 0.0, 0.0)

    # live joint-stiffness override (shared by both fingers); not persisted
    for f in "ab":
        for n in bg.JOINT_NAMES:
            model.jnt_stiffness[ids.jnt[f][n]] = snap.stiffness[n]

    # per-finger tendon pull ΔL (low-passed), driven exactly like validation.py:
    #   springlength = L_rest - ΔL  ->  the stiff string pulls the joints flexed
    for f in "ab":
        cur[f] += (snap.dL[f] - cur[f]) * SMOOTH
        ls = ids.Lrest[f] - cur[f]
        model.tendon_lengthspring[ids.tendon[f]] = (ls, ls)
    return did_reset


def _readout(model, data, ids):
    ro = {}
    for f in "ab":
        for n in bg.JOINT_NAMES:
            ro[f"{f}_{n}"] = float(np.degrees(data.qpos[ids.qadr[f][n]]))
        t = ids.tendon[f]
        ro[f"{f}_T"] = float(model.tendon_stiffness[t]) * max(   # tendon tension [N]
            0.0, float(data.ten_length[t]) - float(model.tendon_lengthspring[t, 0]))
    ro["grip"] = bg.grip_force_on_object(model, data)
    ro["ncon"] = int(data.ncon)
    return ro


def sim_thread(state, xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ids = _make_ids(model)
    ids.Lrest = {f: float(data.ten_length[ids.tendon[f]]) for f in "ab"}
    cur = {"a": 0.0, "b": 0.0}

    try:
        viewer_cm = mujoco.viewer.launch_passive(model, data)
    except Exception as exc:                       # pragma: no cover
        print(f"[gripper] could not open viewer: {exc}")
        with state.lock:
            state.running = False
        return

    # The gripper runs a small timestep (rigid contacts), so step several physics
    # sub-steps per rendered frame to stay real-time without syncing at kHz.
    n_sub = max(1, round((1.0 / 120.0) / model.opt.timestep))
    frame_dt = n_sub * model.opt.timestep

    with viewer_cm as viewer:
        # Look along +X so both fingers (separated along Y) and the gap between
        # them are visible, with the curl plane (Y–Z) facing the camera.
        viewer.cam.distance = 0.42
        viewer.cam.elevation = -10
        viewer.cam.azimuth = 0
        viewer.cam.lookat[:] = [0.0, 0.0, config.GRIPPER_MOUNT_HEIGHT + 0.05]

        while viewer.is_running():
            t0 = time.time()
            with state.lock:
                if not state.running:
                    break
            snap = state.snapshot()
            if _apply(model, data, ids, cur, snap):
                mujoco.mj_forward(model, data)
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            viewer.sync()

            ro = _readout(model, data, ids)
            with state.lock:
                state.readout = ro

            dt = frame_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    with state.lock:
        state.running = False


# =====================================================================
#  Control panel (Tk, main thread)
# =====================================================================
def _safe_get(var):
    try:
        return float(var.get())
    except (tk.TclError, ValueError):
        return None


def _slider_entry(parent, label, lo, hi, init, resolution, on_value):
    var = tk.DoubleVar(value=init)
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=2)
    ctk.CTkLabel(row, text=label, width=80, anchor="w").pack(side="left")
    
    scale = ctk.CTkSlider(row, from_=lo, to=hi)
    scale.set(init)
    scale.pack(side="left", fill="x", expand=True, padx=(5, 10))
    entry = ctk.CTkEntry(row, width=60, textvariable=var)
    entry.pack(side="left")

    def _cb(*_):
        v = _safe_get(var)
        if v is not None:
            v = max(lo, min(hi, v))
            scale.set(v)
            on_value(v)
            
    def _slider_cb(v):
        var.set(round(float(v), 3))
        
    scale.configure(command=_slider_cb)
    var.trace_add("write", _cb)
    return var


def build_ui(state):
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.title("Gripper Control")
    root.minsize(400, 0)

    pad = dict(padx=10, pady=(10, 0))
    guard = {"on": False}
    active_finger = {"f": "a"}

    def set_state(**kw):
        with state.lock:
            for k, v in kw.items():
                setattr(state, k, v)

    def set_dL(finger, mm):
        with state.lock:
            state.dL[finger] = max(0.0, mm) / 1000.0

    def set_stiffness(joint, val):
        with state.lock:
            state.stiffness[joint] = max(0.0, val)

    # ---- Movement mode -------------------------------------------------
    mode_f = ctk.CTkFrame(root)
    mode_f.pack(fill="x", **pad)
    ctk.CTkLabel(mode_f, text="Movement mode", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))
    link_var = tk.BooleanVar(value=state.link)

    # ---- Finger drive ---------------------------------
    flex_f = ctk.CTkFrame(root)
    flex_f.pack(fill="x", **pad)
    ctk.CTkLabel(flex_f, text="Finger drive — tendon pull ΔL (mm)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))

    def on_fa(v):
        set_dL("a", v)
        if link_var.get() and not guard["on"]:
            guard["on"] = True
            fb_var.set(round(v, 2))
            guard["on"] = False
            set_dL("b", v)

    def on_fb(v):
        set_dL("b", v)
        if link_var.get() and not guard["on"]:
            guard["on"] = True
            fa_var.set(round(v, 2))
            guard["on"] = False
            set_dL("a", v)

    fa_var = _slider_entry(flex_f, "Finger A", 0.0, MAX_PULL_MM, 0.0, 0.1, on_fa)
    fb_var = _slider_entry(flex_f, "Finger B", 0.0, MAX_PULL_MM, 0.0, 0.1, on_fb)

    def on_link():
        set_state(link=link_var.get())
        if link_var.get():
            v = _safe_get(fa_var) or 0.0
            fb_var.set(round(v, 2))
            set_dL("b", v)
    ctk.CTkCheckBox(mode_f, text="Simultaneous (link both fingers)", variable=link_var, command=on_link).pack(side="left", padx=10, pady=10)

    # ---- Joint stiffness ----------------
    stf_f = ctk.CTkFrame(root)
    stf_f.pack(fill="x", **pad)
    ctk.CTkLabel(stf_f, text="Joint stiffness — shared (N·m/rad)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))
    for jn in bg.JOINT_NAMES:
        _slider_entry(stf_f, jn.upper(), 0.0, STIFF_MAX, STIFF_DEFAULT[jn], 0.01, lambda v, j=jn: set_stiffness(j, v))
    ctk.CTkLabel(stf_f, text="(not saved — loads from config.py each launch)", text_color="gray").pack(anchor="w", padx=10, pady=(0, 5))

    # ---- Aperture ------------------------------------------------------
    aper_f = ctk.CTkFrame(root)
    aper_f.pack(fill="x", **pad)
    ctk.CTkLabel(aper_f, text="Aperture (mm)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))
    aper_var = _slider_entry(aper_f, "Gap", APER_MIN_MM, APER_MAX_MM, config.GRIPPER_SEPARATION * 1000, 1.0, lambda v: set_state(aperture=v / 1000.0))

    # ---- Probe object --------------------------------------------------
    obj_f = ctk.CTkFrame(root)
    obj_f.pack(fill="x", **pad)
    ctk.CTkLabel(obj_f, text="Probe object", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))

    top = ctk.CTkFrame(obj_f, fg_color="transparent")
    top.pack(fill="x", pady=2, padx=10)
    obj_en = tk.BooleanVar(value=state.obj_enabled)
    ctk.CTkCheckBox(top, text="Enabled", variable=obj_en, command=lambda: set_state(obj_enabled=obj_en.get())).pack(side="left", padx=(0, 10))
    
    ctk.CTkLabel(top, text="Shape:").pack(side="left", padx=(5, 2))
    shape_var = ctk.StringVar(value=state.obj_shape)
    shape_cb = ctk.CTkComboBox(top, variable=shape_var, values=SHAPES, width=100, command=lambda v: set_state(obj_shape=v))
    shape_cb.pack(side="left", padx=5)

    _slider_entry(obj_f, "Size r", SIZE_MIN_MM, SIZE_MAX_MM, config.GRIPPER_OBJECT_SIZE_MM, 0.5, lambda v: set_state(obj_size_mm=v))
    _slider_entry(obj_f, "Length", LEN_MIN_MM, LEN_MAX_MM, config.GRIPPER_OBJECT_LENGTH_MM, 0.5, lambda v: set_state(obj_len_mm=v))
    depth_var = _slider_entry(obj_f, "Depth", DEPTH_MIN_MM, DEPTH_MAX_MM, config.GRIPPER_OBJECT_DEPTH_Z * 1000, 1.0, lambda v: set_state(obj_depth=v / 1000.0))
    ctk.CTkLabel(obj_f, text="Depth: low = enveloping · high = pinch", text_color="gray").pack(anchor="w", padx=10, pady=(0, 5))

    # ---- Scene ---------------------------------------------------------
    scene_f = ctk.CTkFrame(root)
    scene_f.pack(fill="x", **pad)
    
    grav_var = tk.BooleanVar(value=state.gravity)
    ctk.CTkCheckBox(scene_f, text="Gravity", variable=grav_var, command=lambda: set_state(gravity=grav_var.get())).pack(side="left", padx=10, pady=10)

    def do_reset():
        guard["on"] = True
        fa_var.set(0.0)
        fb_var.set(0.0)
        guard["on"] = False
        set_dL("a", 0.0)
        set_dL("b", 0.0)
        set_state(reset=True)

    ctk.CTkButton(scene_f, text="Reset", width=80, command=do_reset).pack(side="left", padx=5, pady=10)
    def on_close():
        set_state(running=False)
        root.destroy()
    ctk.CTkButton(scene_f, text="Quit", width=80, fg_color="#C0392B", hover_color="#922B21", command=on_close).pack(side="left", padx=5, pady=10)

    hint = ("Hotkeys:  m link · 1/2 pick finger · ↑/↓ jog finger ΔL · "
            "←/→ aperture · [ / ] depth · g gravity · o object · r reset · q quit")
    ctk.CTkLabel(root, text=hint, text_color="gray", wraplength=360, justify="left").pack(fill="x", padx=10, pady=(10, 10))

    # ---- Separate Plot Window -----------------------------------------
    plot_win = ctk.CTkToplevel(root)
    plot_win.title("Gripper Dashboard")
    plot_win.geometry("500x400")
    plot_win.protocol("WM_DELETE_WINDOW", lambda: None) # Prevent accidental closing
    
    readout_var = ctk.StringVar(value="(starting…)")
    ctk.CTkLabel(plot_win, textvariable=readout_var, font=ctk.CTkFont(family="Courier", size=12), justify="left").pack(anchor="w", padx=10, pady=10)

    t_buf, grip_buf = deque(maxlen=300), deque(maxlen=300)
    a_buf, b_buf = deque(maxlen=300), deque(maxlen=300)
    fig = Figure(figsize=(5, 3.5), dpi=90)
    fig.patch.set_facecolor('#2b2b2b') # match dark theme
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.90, bottom=0.15, hspace=0.4)
    (grip_line,) = ax1.plot([], [], color="#E74C3C", lw=1.8)
    ax1.set_title("Grip force (N)", fontsize=9, color='white')
    (a_line,) = ax2.plot([], [], color="#3498DB", lw=1.6, label="A")
    (b_line,) = ax2.plot([], [], color="#2ECC71", lw=1.6, label="B")
    ax2.set_title("Closure (°)", fontsize=9, color='white')
    ax2.set_xlabel("time [s]", fontsize=8, color='white')
    ax2.legend(fontsize=8, loc="upper left", ncol=2)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8, colors='white')
        ax.set_facecolor('#333333')
        for spine in ax.spines.values():
            spine.set_color('gray')
    canvas = FigureCanvasTkAgg(fig, master=plot_win)
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
    plot_t0 = time.time()

    # ---- Keyboard ------------------------------------------------------
    def jog_finger(delta_mm):
        f = active_finger["f"]
        var = fa_var if f == "a" else fb_var
        v = max(0.0, min(MAX_PULL_MM, (_safe_get(var) or 0.0) + delta_mm))
        var.set(round(v, 2))

    def jog_var(var, delta, lo, hi):
        v = max(lo, min(hi, (_safe_get(var) or 0.0) + delta))
        var.set(round(v, 3))

    def bind(seq, fn):
        root.bind(seq, lambda e: fn())
        plot_win.bind(seq, lambda e: fn())

    bind("<Key-m>", lambda: (link_var.set(not link_var.get()), on_link()))
    bind("<Key-1>", lambda: active_finger.__setitem__("f", "a"))
    bind("<Key-2>", lambda: active_finger.__setitem__("f", "b"))
    bind("<Up>", lambda: jog_finger(+0.5))
    bind("<Down>", lambda: jog_finger(-0.5))
    bind("<Key-g>", lambda: (grav_var.set(not grav_var.get()), set_state(gravity=grav_var.get())))
    bind("<Key-o>", lambda: (obj_en.set(not obj_en.get()), set_state(obj_enabled=obj_en.get())))
    bind("<Left>", lambda: jog_var(aper_var, -2.0, APER_MIN_MM, APER_MAX_MM))
    bind("<Right>", lambda: jog_var(aper_var, +2.0, APER_MIN_MM, APER_MAX_MM))
    bind("<bracketleft>", lambda: jog_var(depth_var, -2.0, DEPTH_MIN_MM, DEPTH_MAX_MM))
    bind("<bracketright>", lambda: jog_var(depth_var, +2.0, DEPTH_MIN_MM, DEPTH_MAX_MM))
    bind("<Key-r>", do_reset)
    bind("<Key-q>", on_close)
    bind("<Escape>", on_close)

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Max grip force limit calculation:
    max_actuator_force = config.LOAD_TEST_MAX_TENDON_FORCE if hasattr(config, 'LOAD_TEST_MAX_TENDON_FORCE') else 441.0
    upper_limit = getattr(config, 'LOAD_TEST_MAX_FORCE_UI_MAX', max_actuator_force * 1.5)

    def poll():
        with state.lock:
            running = state.running
            ro = dict(state.readout)
        if not running:
            root.destroy()
            return
        if ro:
            readout_var.set(
                f"Finger A   MCP {ro['a_mcp']:+6.1f}°  PIP {ro['a_pip']:+6.1f}°  "
                f"DIP {ro['a_dip']:+6.1f}°   T {ro['a_T']:6.1f} N\n"
                f"Finger B   MCP {ro['b_mcp']:+6.1f}°  PIP {ro['b_pip']:+6.1f}°  "
                f"DIP {ro['b_dip']:+6.1f}°   T {ro['b_T']:6.1f} N\n"
                f"Grip force {ro['grip']:6.2f} N      contacts {ro['ncon']}")
            t_buf.append(time.time() - plot_t0)
            grip_buf.append(ro["grip"])
            a_buf.append(ro["a_mcp"] + ro["a_pip"] + ro["a_dip"])
            b_buf.append(ro["b_mcp"] + ro["b_pip"] + ro["b_dip"])
            grip_line.set_data(t_buf, grip_buf)
            a_line.set_data(t_buf, a_buf)
            b_line.set_data(t_buf, b_buf)
            
            # Auto-scale x, but clamp y for grip force
            ax1.set_xlim(max(0, t_buf[-1] - 30), t_buf[-1] + 1)
            current_max_grip = max(grip_buf) if grip_buf else 0
            ax1.set_ylim(0, min(max(current_max_grip * 1.1, 10), upper_limit))
            
            ax2.set_xlim(max(0, t_buf[-1] - 30), t_buf[-1] + 1)
            ax2.relim()
            ax2.autoscale_view(scalex=False, scaley=True)
            
            canvas.draw_idle()
        root.after(100, poll)

    root.after(200, poll)
    return root

def main():
    xml = bg.generate_gripper_xml()
    print(f"[gripper] built {xml}")
    state = GripperState()
    sim = threading.Thread(target=sim_thread, args=(state, xml), daemon=True)
    sim.start()
    root = build_ui(state)
    root.mainloop()
    with state.lock:
        state.running = False
    sim.join(timeout=3.0)     # let the viewer close its GL context before we exit
    print("[gripper] panel closed.")


def selftest():
    """Headless: exercise the sim-apply/readout path without a viewer or Tk."""
    xml = bg.generate_gripper_xml()
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = _make_ids(model)
    ids.Lrest = {f: float(data.ten_length[ids.tendon[f]]) for f in "ab"}
    cur = {"a": 0.0, "b": 0.0}

    # Individual mode: only finger A is pulled (B idle); stiffness softened live.
    snap = SimpleNamespace(reset=False, link=False, dL={"a": 0.016, "b": 0.0},
                           stiffness={"mcp": 0.5, "pip": 1.0, "dip": 1.0},
                           aperture=config.GRIPPER_SEPARATION, obj_enabled=True,
                           obj_shape="cylinder", obj_size_mm=config.GRIPPER_OBJECT_SIZE_MM,
                           obj_len_mm=config.GRIPPER_OBJECT_LENGTH_MM,
                           obj_depth=config.GRIPPER_OBJECT_DEPTH_Z, gravity=False)
    for _ in range(int(2.5 / model.opt.timestep)):
        _apply(model, data, ids, cur, snap)
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos)), "sim blew up"
    ro = _readout(model, data, ids)
    print(f"[selftest] A T={ro['a_T']:.0f}N  B T={ro['b_T']:.0f}N  "
          f"grip={ro['grip']:.2f}N  ncon={ro['ncon']}  (B idle as expected)")
    print("[selftest] OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()

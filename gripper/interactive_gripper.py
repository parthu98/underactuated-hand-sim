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
from tkinter import ttk
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
    """A labelled row: tk.Scale + ttk.Entry sharing one DoubleVar. Returns the
    var. `on_value(v)` is called with the float whenever it changes validly."""
    var = tk.DoubleVar(value=init)
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=1)
    ttk.Label(row, text=label, width=11).pack(side="left")
    scale = tk.Scale(row, from_=lo, to=hi, resolution=resolution,
                     orient="horizontal", variable=var, showvalue=False,
                     length=210, sliderlength=16)
    scale.pack(side="left", fill="x", expand=True, padx=(2, 4))
    ttk.Entry(row, width=7, textvariable=var).pack(side="left")

    def _cb(*_):
        v = _safe_get(var)
        if v is not None:
            on_value(v)
    var.trace_add("write", _cb)
    return var


def build_ui(state):
    root = tk.Tk()
    root.title("Gripper Control")
    root.minsize(360, 0)
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass

    pad = dict(padx=8, pady=(6, 0))
    guard = {"on": False}        # reentrancy guard for finger linking
    active_finger = {"f": "a"}   # which finger the keyboard jogs

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
    mode = ttk.LabelFrame(root, text="Movement mode")
    mode.pack(fill="x", **pad)
    link_var = tk.BooleanVar(value=state.link)

    # ---- Finger drive (tendon pull ΔL) ---------------------------------
    flex = ttk.LabelFrame(root, text="Finger drive  —  tendon pull ΔL (mm)")
    flex.pack(fill="x", **pad)

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

    fa_var = _slider_entry(flex, "Finger A", 0.0, MAX_PULL_MM, 0.0, 0.1, on_fa)
    fb_var = _slider_entry(flex, "Finger B", 0.0, MAX_PULL_MM, 0.0, 0.1, on_fb)

    def on_link():
        set_state(link=link_var.get())
        if link_var.get():                 # snap B to A on (re)link
            v = _safe_get(fa_var) or 0.0
            fb_var.set(round(v, 2))
            set_dL("b", v)
    ttk.Checkbutton(mode, text="Simultaneous  (link both fingers)",
                    variable=link_var, command=on_link).pack(side="left", padx=6, pady=4)

    # ---- Joint stiffness (live, shared by both fingers) ----------------
    stf = ttk.LabelFrame(root, text="Joint stiffness  —  live, shared (N·m/rad)")
    stf.pack(fill="x", **pad)
    for jn in bg.JOINT_NAMES:
        _slider_entry(stf, jn.upper(), 0.0, STIFF_MAX, STIFF_DEFAULT[jn], 0.01,
                      lambda v, j=jn: set_stiffness(j, v))
    ttk.Label(stf, text="(not saved — loads from config.py each launch)",
              foreground="#555").pack(anchor="w", padx=6, pady=(0, 4))

    # ---- Aperture ------------------------------------------------------
    aper = ttk.LabelFrame(root, text="Aperture  —  centre-to-centre gap (mm)")
    aper.pack(fill="x", **pad)
    aper_var = _slider_entry(aper, "Aperture", APER_MIN_MM, APER_MAX_MM,
                             config.GRIPPER_SEPARATION * 1000, 1.0,
                             lambda v: set_state(aperture=v / 1000.0))

    # ---- Probe object --------------------------------------------------
    obj = ttk.LabelFrame(root, text="Probe object  (static — fingers grip it)")
    obj.pack(fill="x", **pad)

    top = ttk.Frame(obj)
    top.pack(fill="x", pady=2)
    obj_en = tk.BooleanVar(value=state.obj_enabled)
    ttk.Checkbutton(top, text="Enabled", variable=obj_en,
                    command=lambda: set_state(obj_enabled=obj_en.get())
                    ).pack(side="left", padx=4)
    ttk.Label(top, text="Shape").pack(side="left", padx=(10, 2))
    shape_var = tk.StringVar(value=state.obj_shape)
    ttk.Combobox(top, textvariable=shape_var, values=SHAPES, width=9,
                 state="readonly").pack(side="left")
    shape_var.trace_add("write",
                        lambda *_: set_state(obj_shape=shape_var.get()))

    _slider_entry(obj, "Size r", SIZE_MIN_MM, SIZE_MAX_MM,
                  config.GRIPPER_OBJECT_SIZE_MM, 0.5,
                  lambda v: set_state(obj_size_mm=v))
    _slider_entry(obj, "Length", LEN_MIN_MM, LEN_MAX_MM,
                  config.GRIPPER_OBJECT_LENGTH_MM, 0.5,
                  lambda v: set_state(obj_len_mm=v))
    depth_var = _slider_entry(obj, "Depth", DEPTH_MIN_MM, DEPTH_MAX_MM,
                              config.GRIPPER_OBJECT_DEPTH_Z * 1000, 1.0,
                              lambda v: set_state(obj_depth=v / 1000.0))
    ttk.Label(obj, text="Depth: low = enveloping grasp · high = pinch grasp",
              foreground="#555").pack(anchor="w", padx=6, pady=(0, 4))

    # ---- Scene ---------------------------------------------------------
    scene = ttk.LabelFrame(root, text="Scene")
    scene.pack(fill="x", **pad)
    grav_var = tk.BooleanVar(value=state.gravity)
    ttk.Checkbutton(scene, text="Gravity", variable=grav_var,
                    command=lambda: set_state(gravity=grav_var.get())
                    ).pack(side="left", padx=6, pady=4)

    def do_reset():
        guard["on"] = True
        fa_var.set(0.0)
        fb_var.set(0.0)
        guard["on"] = False
        set_dL("a", 0.0)
        set_dL("b", 0.0)
        set_state(reset=True)

    ttk.Button(scene, text="Reset", command=do_reset).pack(side="left", padx=4)

    def on_close():
        set_state(running=False)
        root.destroy()
    ttk.Button(scene, text="Quit", command=on_close).pack(side="left", padx=4)

    # ---- Live readout + dashboard plot --------------------------------
    read = ttk.LabelFrame(root, text="Readout")
    read.pack(fill="both", expand=True, **pad)
    readout_var = tk.StringVar(value="(starting…)")
    ttk.Label(read, textvariable=readout_var, font=("TkFixedFont", 9),
              justify="left").pack(anchor="w", padx=6, pady=4)

    t_buf, grip_buf = deque(maxlen=300), deque(maxlen=300)
    a_buf, b_buf = deque(maxlen=300), deque(maxlen=300)
    fig = Figure(figsize=(4.2, 2.7), dpi=90)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.90, bottom=0.20, hspace=0.55)
    (grip_line,) = ax1.plot([], [], color="#C0392B", lw=1.6)
    ax1.set_title("Grip force", fontsize=8)
    ax1.set_ylabel("N", fontsize=8)
    (a_line,) = ax2.plot([], [], color="#1F4E79", lw=1.4, label="A")
    (b_line,) = ax2.plot([], [], color="#2E8B57", lw=1.4, label="B")
    ax2.set_ylabel("closure °", fontsize=8)
    ax2.set_xlabel("time [s]", fontsize=8)
    ax2.legend(fontsize=7, loc="upper left", ncol=2)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    canvas = FigureCanvasTkAgg(fig, master=read)
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=(0, 4))
    plot_t0 = time.time()

    hint = ("Hotkeys:  m link · 1/2 pick finger · ↑/↓ jog finger ΔL · "
            "←/→ aperture · [ / ] depth · g gravity · o object · r reset · q quit")
    ttk.Label(root, text=hint, foreground="#666", wraplength=360,
              justify="left").pack(fill="x", padx=8, pady=(4, 8))

    # ---- Keyboard ------------------------------------------------------
    def jog_finger(delta_mm):
        f = active_finger["f"]
        var = fa_var if f == "a" else fb_var
        v = max(0.0, min(MAX_PULL_MM, (_safe_get(var) or 0.0) + delta_mm))
        var.set(round(v, 2))

    def jog_var(var, delta, lo, hi, scale=1.0):
        v = max(lo, min(hi, (_safe_get(var) or 0.0) + delta))
        var.set(round(v, 3))

    def bind(seq, fn):
        root.bind(seq, lambda e: fn())

    bind("<Key-m>", lambda: (link_var.set(not link_var.get()), on_link()))
    bind("<Key-1>", lambda: active_finger.__setitem__("f", "a"))
    bind("<Key-2>", lambda: active_finger.__setitem__("f", "b"))
    bind("<Up>", lambda: jog_finger(+0.5))
    bind("<Down>", lambda: jog_finger(-0.5))
    bind("<Key-g>", lambda: (grav_var.set(not grav_var.get()),
                             set_state(gravity=grav_var.get())))
    bind("<Key-o>", lambda: (obj_en.set(not obj_en.get()),
                             set_state(obj_enabled=obj_en.get())))
    bind("<Left>", lambda: jog_var(aper_var, -2.0, APER_MIN_MM, APER_MAX_MM))
    bind("<Right>", lambda: jog_var(aper_var, +2.0, APER_MIN_MM, APER_MAX_MM))
    bind("<bracketleft>", lambda: jog_var(depth_var, -2.0, DEPTH_MIN_MM, DEPTH_MAX_MM))
    bind("<bracketright>", lambda: jog_var(depth_var, +2.0, DEPTH_MIN_MM, DEPTH_MAX_MM))
    bind("<Key-r>", do_reset)
    bind("<Key-q>", on_close)
    bind("<Escape>", on_close)

    root.protocol("WM_DELETE_WINDOW", on_close)

    # ---- Readout poll + lifecycle -------------------------------------
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
            for ax in (ax1, ax2):
                ax.relim()
                ax.autoscale_view()
            canvas.draw_idle()
        root.after(100, poll)

    root.after(200, poll)
    return root


# =====================================================================
#  Entry points
# =====================================================================
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

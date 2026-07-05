"""A small GUI to pick exactly which (video case) x (pipeline stage)
combinations to run, instead of always re-running the whole pipeline for
every case. Useful once you've already processed everything once and just
want to regenerate one stage (e.g. after tweaking a renderer's CONFIG) for
one case, without waiting on the others.

Usage:
    python dashboard.py
"""

from __future__ import annotations

from _bootstrap import ensure_venv
ensure_venv()

import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import run_all

STAGES = [
    ("track", "Track Points"),
    ("overlay", "Joint Overlay"),
    ("trajectory", "Tip Trajectory"),
    ("chart", "Angle Chart"),
]
STAGE_KEYS = [s for s, _ in STAGES]

# Which build_paths() keys count as "the output(s) of this stage" -- used to
# decide the default (pre-checked) state of each checkbox.
STAGE_OUTPUTS = {
    "track": ["cache"],
    "overlay": ["joint_video", "angle_csv"],
    "trajectory": ["tip_video", "workspace_png"],
    "chart": ["chart_video", "chart_png", "chart_pdf"],
}


class DashboardApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Finger Pipeline Dashboard")

        self.stems, self.labels, warnings = run_all.discover_cases()
        if not self.stems:
            warnings = warnings + [
                f"No raw videos found in {run_all.VIDEOS_DIR / 'raw'} -- "
                "drop a .mp4 there and restart the dashboard."
            ]

        self.vars: dict[tuple[str, str], tk.BooleanVar] = {}
        self.checkbuttons: dict[tuple[str, str], ttk.Checkbutton] = {}
        self.row_vars: dict[str, tk.BooleanVar] = {}
        self.col_vars: dict[str, tk.BooleanVar] = {}

        self._build_grid()

        self.run_button = ttk.Button(root, text="Run Selected", command=self._on_run_clicked)
        self.run_button.pack(pady=(4, 4))

        self.log_widget = ScrolledText(root, height=16, state="disabled")
        self.log_widget.pack(fill="both", expand=True, padx=4, pady=4)

        for w in warnings:
            self._append_log(f"[warn] {w}")

        self._refresh_defaults()

    # ---------- grid ----------

    def _build_grid(self):
        frame = ttk.Frame(self.root)
        frame.pack(fill="x", padx=4, pady=4)

        ttk.Label(frame, text="Case").grid(row=0, column=0, sticky="w", padx=4)
        for col, (stage, title) in enumerate(STAGES, start=1):
            col_var = tk.BooleanVar(value=False)
            self.col_vars[stage] = col_var
            cb = ttk.Checkbutton(frame, text=title, variable=col_var,
                                  command=lambda s=stage: self._toggle_column(s))
            cb.grid(row=0, column=col, padx=4, pady=2)

        for row, stem in enumerate(self.stems, start=1):
            label = self.labels.get(stem, stem)
            row_var = tk.BooleanVar(value=False)
            self.row_vars[stem] = row_var
            row_cb = ttk.Checkbutton(frame, text=label, variable=row_var,
                                      command=lambda s=stem: self._toggle_row(s))
            row_cb.grid(row=row, column=0, sticky="w", padx=4, pady=1)

            for col, (stage, _title) in enumerate(STAGES, start=1):
                var = tk.BooleanVar(value=False)
                self.vars[(stem, stage)] = var
                cb = ttk.Checkbutton(frame, variable=var)
                cb.grid(row=row, column=col, padx=4, pady=1)
                self.checkbuttons[(stem, stage)] = cb

    def _toggle_row(self, stem):
        val = self.row_vars[stem].get()
        for stage, _ in STAGES:
            self.vars[(stem, stage)].set(val)

    def _toggle_column(self, stage):
        val = self.col_vars[stage].get()
        for stem in self.stems:
            self.vars[(stem, stage)].set(val)

    # ---------- default checkbox state ----------

    def _refresh_defaults(self):
        for stem in self.stems:
            paths = run_all.build_paths(stem)
            for stage, _ in STAGES:
                outputs = STAGE_OUTPUTS[stage]
                needs_run = any(not Path(paths[key]).exists() for key in outputs)
                self.vars[(stem, stage)].set(needs_run)
        # Row/column "select all" boxes are momentary toggles, not a
        # reflection of partial selection state -- always reset to unchecked.
        for stem in self.stems:
            self.row_vars[stem].set(False)
        for stage, _ in STAGES:
            self.col_vars[stage].set(False)

    # ---------- run ----------

    def _on_run_clicked(self):
        selected = [
            (stem, stage)
            for stem in self.stems
            for stage in STAGE_KEYS
            if self.vars[(stem, stage)].get()
        ]
        if not selected:
            self._append_log("[info] nothing selected.")
            return

        self.run_button.config(state="disabled")
        threading.Thread(target=self._worker, args=(selected,), daemon=True).start()

    def _worker(self, pairs):
        ok_count = 0
        fail_count = 0
        for stem, stage in pairs:
            try:
                rc = run_all.run_stage(
                    stage, stem,
                    log=lambda line: self.root.after(0, self._append_log, line),
                )
                if rc == 0:
                    ok_count += 1
                else:
                    fail_count += 1
                    self.root.after(0, self._append_log,
                                     f"[FAILED] {stem} / {stage}: exit code {rc}")
            except Exception as e:  # noqa: BLE001 -- one bad pair shouldn't stop the batch
                fail_count += 1
                self.root.after(0, self._append_log, f"[FAILED] {stem} / {stage}: {e}")

        self.root.after(0, self._append_log, f"Done: {ok_count} ok, {fail_count} failed.")
        self.root.after(0, lambda: self.run_button.config(state="normal"))
        self.root.after(0, self._refresh_defaults)

    # ---------- logging ----------

    def _append_log(self, line: str):
        self.log_widget.config(state="normal")
        self.log_widget.insert("end", line + "\n")
        self.log_widget.see("end")
        self.log_widget.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()

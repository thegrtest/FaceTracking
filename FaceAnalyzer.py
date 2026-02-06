"""
mp_visualizer.py

Visualizer + analytics for MediaPipe JSONL logs produced by your tracker app.
Goal: identify where on a machine it is being worked on most frequently.

Key features
- Load JSONL (run_*.jsonl) and parse "sample" records
- Filter by:
    * time window
    * camera index
    * require pose present
- Compute statistics:
    * total time observed
    * "people seen" as presence episodes
    * stillness intervals (no movement)
    * top visited regions (heatmap)
- Heatmap overlay on a reference image loaded from the root directory (or any path).
  Assumes the reference image matches the camera viewpoint.
  If its resolution differs from the logged frame size, we scale the heatmap to match.

Run:
  python mp_visualizer.py --log mp_logs/run_XXXXXXXXXX.jsonl --ref reference.png

Notes:
- Uses torso center (derived in logs) as the visit point.
- Stillness uses pixel movement threshold (in the reference/heatmap coordinate system).

Dependencies:
  pip install PySide6 opencv-python numpy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Sample:
    ts: float
    ts_iso: str
    run_id: str
    frame_idx: int
    camera_index: int
    frame_w: int
    frame_h: int
    fps: float

    pose_present: bool
    face_present: bool

    torso_x: Optional[int]
    torso_y: Optional[int]

    bbox: Optional[Dict[str, int]]


# -----------------------------
# Helpers
# -----------------------------
def parse_iso_utc(s: str) -> Optional[float]:
    # expects "...Z"
    try:
        if s.endswith("Z"):
            s = s[:-1]
        # datetime.fromisoformat handles milliseconds
        dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def load_jsonl_samples(path: str) -> Tuple[List[Sample], Dict[str, Any]]:
    """
    Returns:
      samples: list of Sample
      meta: session_start settings if found
    """
    samples: List[Sample] = []
    meta: Dict[str, Any] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            rtype = rec.get("type")
            if rtype == "session_start" and not meta:
                meta = rec
                continue

            if rtype != "sample":
                continue

            frame = rec.get("frame", {})
            flags = rec.get("flags", {})
            pose_derived = rec.get("pose_derived", {})

            torso = safe_get(pose_derived, ["torso_center_px"], None)
            bbox = safe_get(pose_derived, ["bbox_px"], None)

            samples.append(
                Sample(
                    ts=float(rec.get("ts", 0.0)),
                    ts_iso=str(rec.get("ts_iso", "")),
                    run_id=str(rec.get("run_id", "")),
                    frame_idx=int(rec.get("frame_idx", -1)),
                    camera_index=int(rec.get("camera_index", -1)),
                    frame_w=int(frame.get("w", 0)),
                    frame_h=int(frame.get("h", 0)),
                    fps=float(rec.get("fps", 0.0)),
                    pose_present=bool(flags.get("pose_present", False)),
                    face_present=bool(flags.get("face_present", False)),
                    torso_x=int(torso.get("x")) if isinstance(torso, dict) and "x" in torso else None,
                    torso_y=int(torso.get("y")) if isinstance(torso, dict) and "y" in torso else None,
                    bbox=bbox if isinstance(bbox, dict) else None,
                )
            )

    # Ensure time order
    samples.sort(key=lambda s: s.ts)
    return samples, meta


def compute_presence_episodes(samples: List[Sample], gap_s: float = 2.0) -> int:
    """
    "People seen" approximation: count pose-present episodes separated by >= gap_s without pose.
    """
    count = 0
    in_episode = False
    last_present_ts = None

    for s in samples:
        if s.pose_present and s.torso_x is not None and s.torso_y is not None:
            if not in_episode:
                count += 1
                in_episode = True
            last_present_ts = s.ts
        else:
            if in_episode and last_present_ts is not None and (s.ts - last_present_ts) >= gap_s:
                in_episode = False

    return count


@dataclass
class StillInterval:
    start_ts: float
    end_ts: float
    duration_s: float
    avg_x: float
    avg_y: float


def compute_stillness_intervals(
    samples: List[Sample],
    move_thresh_px: float = 8.0,
    min_still_s: float = 10.0,
    require_pose: bool = True,
) -> List[StillInterval]:
    """
    Detect intervals where the torso center moves less than move_thresh_px over time.
    Simple rule:
      - Track torso positions when (pose_present if require_pose)
      - When displacement between consecutive valid samples stays <= thresh, we accumulate still time
      - When it exceeds thresh, we close interval if it is long enough.
    """
    intervals: List[StillInterval] = []
    active = False
    start_ts = 0.0
    last_ts = 0.0
    last_xy: Optional[Tuple[int, int]] = None
    xs: List[float] = []
    ys: List[float] = []

    def close_interval(end_ts: float):
        nonlocal active, start_ts, last_ts, last_xy, xs, ys
        dur = end_ts - start_ts
        if active and dur >= min_still_s and xs and ys:
            intervals.append(
                StillInterval(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    duration_s=dur,
                    avg_x=float(np.mean(xs)),
                    avg_y=float(np.mean(ys)),
                )
            )
        active = False
        last_xy = None
        xs = []
        ys = []

    for s in samples:
        ok = (s.torso_x is not None and s.torso_y is not None)
        if require_pose:
            ok = ok and s.pose_present
        if not ok:
            # if we lose tracking, end the still interval
            if active:
                close_interval(last_ts if last_ts > 0 else s.ts)
            continue

        xy = (int(s.torso_x), int(s.torso_y))
        if last_xy is None:
            # start candidate interval
            active = True
            start_ts = s.ts
            last_ts = s.ts
            last_xy = xy
            xs = [xy[0]]
            ys = [xy[1]]
            continue

        dx = xy[0] - last_xy[0]
        dy = xy[1] - last_xy[1]
        dist = math.hypot(dx, dy)

        if dist <= move_thresh_px:
            # still
            if not active:
                active = True
                start_ts = s.ts
                xs = []
                ys = []
            xs.append(xy[0])
            ys.append(xy[1])
            last_ts = s.ts
            last_xy = xy
        else:
            # moved
            if active:
                close_interval(s.ts)
            # reset to new point
            active = True
            start_ts = s.ts
            last_ts = s.ts
            last_xy = xy
            xs = [xy[0]]
            ys = [xy[1]]

    if active:
        close_interval(last_ts)

    # Sort longest first (useful)
    intervals.sort(key=lambda it: it.duration_s, reverse=True)
    return intervals


def make_heatmap_overlay(
    ref_bgr: np.ndarray,
    points_xy: np.ndarray,
    sigma: int = 31,
    alpha: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create heatmap overlay over ref_bgr.

    points_xy: Nx2 array in pixel coords (same coordinate system as ref image)
    sigma: gaussian blur kernel size (odd)
    returns: (overlay_bgr, heatmap_color_bgr)
    """
    h, w = ref_bgr.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    if points_xy.size > 0:
        xs = np.clip(points_xy[:, 0].astype(np.int32), 0, w - 1)
        ys = np.clip(points_xy[:, 1].astype(np.int32), 0, h - 1)
        for x, y in zip(xs, ys):
            heat[y, x] += 1.0

    # Blur to form density
    sigma = int(sigma)
    if sigma % 2 == 0:
        sigma += 1
    if sigma >= 3:
        heat = cv2.GaussianBlur(heat, (sigma, sigma), 0)

    # Normalize to 0..255
    mx = float(heat.max()) if heat.size else 0.0
    if mx > 0:
        heat_u8 = np.clip((heat / mx) * 255.0, 0, 255).astype(np.uint8)
    else:
        heat_u8 = heat.astype(np.uint8)

    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    # Overlay
    overlay = ref_bgr.copy()
    overlay = cv2.addWeighted(overlay, 1.0, heat_color, float(alpha), 0)

    return overlay, heat_color


def to_qimage_bgr(img_bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()


def fmt_ts(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


# -----------------------------
# Visualizer UI
# -----------------------------
class HeatmapView(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background:#111; color:#ddd; border:1px solid #333;")
        self._pix: Optional[QtGui.QPixmap] = None

    def set_image(self, qimg: QtGui.QImage):
        self._pix = QtGui.QPixmap.fromImage(qimg)
        self._render_scaled()

    def resizeEvent(self, ev: QtGui.QResizeEvent):
        super().resizeEvent(ev)
        self._render_scaled()

    def _render_scaled(self):
        if self._pix is None:
            return
        scaled = self._pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(scaled)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, log_path: Optional[str] = None, ref_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("MediaPipe Workpoint Visualizer (JSONL -> Heatmap + Stats)")
        self.resize(1400, 900)

        self.samples: List[Sample] = []
        self.meta: Dict[str, Any] = {}
        self.ref_bgr: Optional[np.ndarray] = None
        self.ref_path: Optional[str] = None

        self._filtered: List[Sample] = []
        self._last_frame_wh: Optional[Tuple[int, int]] = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # Left: image view + stats
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, stretch=2)

        self.view = HeatmapView()
        left.addWidget(self.view, stretch=3)

        self.stats = QtWidgets.QTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setStyleSheet("background:#0f1116; color:#d6d6d6; border:1px solid #2a2f3a;")
        left.addWidget(self.stats, stretch=1)

        # Right: controls + intervals table
        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, stretch=1)

        # File controls
        file_box = QtWidgets.QGroupBox("Inputs")
        fb = QtWidgets.QGridLayout(file_box)

        self.txt_log = QtWidgets.QLineEdit()
        self.btn_log = QtWidgets.QPushButton("Browse JSONL")
        self.btn_log.clicked.connect(self._pick_log)

        self.txt_ref = QtWidgets.QLineEdit()
        self.btn_ref = QtWidgets.QPushButton("Browse Ref Image")
        self.btn_ref.clicked.connect(self._pick_ref)

        fb.addWidget(QtWidgets.QLabel("Log:"), 0, 0)
        fb.addWidget(self.txt_log, 0, 1)
        fb.addWidget(self.btn_log, 0, 2)

        fb.addWidget(QtWidgets.QLabel("Ref:"), 1, 0)
        fb.addWidget(self.txt_ref, 1, 1)
        fb.addWidget(self.btn_ref, 1, 2)

        self.btn_reload = QtWidgets.QPushButton("Load / Reload")
        self.btn_reload.clicked.connect(self._reload_all)
        fb.addWidget(self.btn_reload, 2, 0, 1, 3)

        right.addWidget(file_box)

        # Filters
        filt_box = QtWidgets.QGroupBox("Filters")
        fg = QtWidgets.QGridLayout(filt_box)

        self.cmb_camera = QtWidgets.QComboBox()
        self.cmb_camera.addItem("All", -1)
        self.cmb_camera.currentIndexChanged.connect(self._recompute)

        self.chk_pose = QtWidgets.QCheckBox("Require Pose Present")
        self.chk_pose.setChecked(True)
        self.chk_pose.toggled.connect(lambda _: self._recompute())

        self.dt_start = QtWidgets.QDateTimeEdit()
        self.dt_start.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.dt_start.setCalendarPopup(True)
        self.dt_start.dateTimeChanged.connect(lambda _: self._recompute())

        self.dt_end = QtWidgets.QDateTimeEdit()
        self.dt_end.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.dt_end.setCalendarPopup(True)
        self.dt_end.dateTimeChanged.connect(lambda _: self._recompute())

        self.spin_sigma = QtWidgets.QSpinBox()
        self.spin_sigma.setRange(3, 301)
        self.spin_sigma.setValue(41)
        self.spin_sigma.setSingleStep(2)
        self.spin_sigma.valueChanged.connect(lambda _: self._recompute())

        self.spin_alpha = QtWidgets.QDoubleSpinBox()
        self.spin_alpha.setRange(0.05, 0.95)
        self.spin_alpha.setSingleStep(0.05)
        self.spin_alpha.setValue(0.45)
        self.spin_alpha.valueChanged.connect(lambda _: self._recompute())

        fg.addWidget(QtWidgets.QLabel("Camera:"), 0, 0)
        fg.addWidget(self.cmb_camera, 0, 1, 1, 2)

        fg.addWidget(QtWidgets.QLabel("Start (local):"), 1, 0)
        fg.addWidget(self.dt_start, 1, 1, 1, 2)

        fg.addWidget(QtWidgets.QLabel("End (local):"), 2, 0)
        fg.addWidget(self.dt_end, 2, 1, 1, 2)

        fg.addWidget(self.chk_pose, 3, 0, 1, 3)

        fg.addWidget(QtWidgets.QLabel("Heat Blur (sigma):"), 4, 0)
        fg.addWidget(self.spin_sigma, 4, 1)

        fg.addWidget(QtWidgets.QLabel("Overlay alpha:"), 4, 2)
        fg.addWidget(self.spin_alpha, 4, 3)

        right.addWidget(filt_box)

        # Stillness settings
        still_box = QtWidgets.QGroupBox("Stillness Detection")
        sg = QtWidgets.QGridLayout(still_box)

        self.spin_move = QtWidgets.QDoubleSpinBox()
        self.spin_move.setRange(1.0, 100.0)
        self.spin_move.setSingleStep(1.0)
        self.spin_move.setValue(8.0)
        self.spin_move.valueChanged.connect(lambda _: self._recompute())

        self.spin_min_still = QtWidgets.QDoubleSpinBox()
        self.spin_min_still.setRange(1.0, 3600.0)
        self.spin_min_still.setSingleStep(5.0)
        self.spin_min_still.setValue(15.0)
        self.spin_min_still.valueChanged.connect(lambda _: self._recompute())

        sg.addWidget(QtWidgets.QLabel("Move thresh (px):"), 0, 0)
        sg.addWidget(self.spin_move, 0, 1)
        sg.addWidget(QtWidgets.QLabel("Min still (s):"), 1, 0)
        sg.addWidget(self.spin_min_still, 1, 1)

        right.addWidget(still_box)

        # Stillness intervals table
        self.tbl = QtWidgets.QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["Duration(s)", "Start(UTC)", "End(UTC)", "Avg XY"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        right.addWidget(self.tbl, stretch=1)

        # Export buttons
        exp = QtWidgets.QHBoxLayout()
        self.btn_export_points = QtWidgets.QPushButton("Export Points CSV")
        self.btn_export_points.clicked.connect(self._export_points_csv)
        self.btn_export_intervals = QtWidgets.QPushButton("Export Stillness CSV")
        self.btn_export_intervals.clicked.connect(self._export_intervals_csv)
        exp.addWidget(self.btn_export_points)
        exp.addWidget(self.btn_export_intervals)
        right.addLayout(exp)

        # Defaults from args
        if log_path:
            self.txt_log.setText(log_path)
        if ref_path:
            self.txt_ref.setText(ref_path)

        # Auto load if both provided
        if log_path or ref_path:
            self._reload_all()

    def _pick_log(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pick JSONL log", "", "JSONL (*.jsonl);;All Files (*)")
        if p:
            self.txt_log.setText(p)

    def _pick_ref(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pick reference image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if p:
            self.txt_ref.setText(p)

    def _reload_all(self):
        log_path = self.txt_log.text().strip()
        ref_path = self.txt_ref.text().strip()

        if log_path:
            try:
                self.samples, self.meta = load_jsonl_samples(log_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load error", f"Failed to read log:\n{e}")
                return
        else:
            self.samples = []
            self.meta = {}

        if ref_path:
            ref_bgr = cv2.imread(ref_path, cv2.IMREAD_COLOR)
            if ref_bgr is None:
                QtWidgets.QMessageBox.critical(self, "Load error", "Failed to load reference image.")
                return
            self.ref_bgr = ref_bgr
            self.ref_path = ref_path
        else:
            self.ref_bgr = None
            self.ref_path = None

        # Populate camera list
        self._populate_cameras()

        # Set filter time bounds from samples (local time in widget, but we store UTC ts)
        self._init_time_bounds()

        self._recompute()

    def _populate_cameras(self):
        self.cmb_camera.blockSignals(True)
        self.cmb_camera.clear()
        self.cmb_camera.addItem("All", -1)
        cams = sorted({s.camera_index for s in self.samples}) if self.samples else []
        for c in cams:
            self.cmb_camera.addItem(f"{c}", c)
        self.cmb_camera.blockSignals(False)

    def _init_time_bounds(self):
        if not self.samples:
            now = QtCore.QDateTime.currentDateTime()
            self.dt_start.setDateTime(now.addSecs(-3600))
            self.dt_end.setDateTime(now)
            return

        t0 = self.samples[0].ts
        t1 = self.samples[-1].ts

        # Convert to local datetime for widget display
        self.dt_start.blockSignals(True)
        self.dt_end.blockSignals(True)
        self.dt_start.setDateTime(QtCore.QDateTime.fromSecsSinceEpoch(int(t0), QtCore.Qt.LocalTime))
        self.dt_end.setDateTime(QtCore.QDateTime.fromSecsSinceEpoch(int(t1), QtCore.Qt.LocalTime))
        self.dt_start.blockSignals(False)
        self.dt_end.blockSignals(False)

    def _recompute(self):
        if not self.samples:
            self.stats.setText("No samples loaded.")
            self.tbl.setRowCount(0)
            self.view.setText("Load a log and reference image.")
            return

        # Filter time window (widgets in local time)
        start_ts = self.dt_start.dateTime().toSecsSinceEpoch()
        end_ts = self.dt_end.dateTime().toSecsSinceEpoch()
        if end_ts < start_ts:
            start_ts, end_ts = end_ts, start_ts

        cam = int(self.cmb_camera.currentData())
        require_pose = bool(self.chk_pose.isChecked())

        filtered: List[Sample] = []
        for s in self.samples:
            if s.ts < start_ts or s.ts > end_ts:
                continue
            if cam != -1 and s.camera_index != cam:
                continue
            if require_pose and not s.pose_present:
                continue
            if s.torso_x is None or s.torso_y is None:
                continue
            filtered.append(s)

        self._filtered = filtered

        # Compute stats
        self._update_stats(start_ts, end_ts, cam, require_pose)

        # Heatmap overlay
        self._update_heatmap()

        # Stillness intervals
        self._update_stillness()

    def _update_stats(self, start_ts: float, end_ts: float, cam: int, require_pose: bool):
        filt = self._filtered
        n = len(filt)

        # Estimate observed duration from sample timestamps
        observed_s = 0.0
        if n >= 2:
            observed_s = max(0.0, filt[-1].ts - filt[0].ts)

        episodes = compute_presence_episodes(filt, gap_s=2.0)

        # Basic region statistics (top grid cells)
        grid = self._grid_counts(filt, grid_w=24, grid_h=14)
        top_cells = sorted(grid.items(), key=lambda kv: kv[1], reverse=True)[:5]

        run_id = self.samples[0].run_id if self.samples else ""
        fw = self.samples[0].frame_w if self.samples else 0
        fh = self.samples[0].frame_h if self.samples else 0

        lines = []
        lines.append(f"Run ID: {run_id}")
        lines.append(f"Samples loaded: {len(self.samples)}  |  Filtered: {n}")
        lines.append(f"Time window (local): {self.dt_start.dateTime().toString('yyyy-MM-dd HH:mm:ss')}  ->  {self.dt_end.dateTime().toString('yyyy-MM-dd HH:mm:ss')}")
        lines.append(f"Camera filter: {'All' if cam == -1 else cam} | Require pose: {require_pose}")
        lines.append("")
        lines.append(f"Frame size in log: {fw}x{fh}")
        if self.ref_bgr is not None:
            rh, rw = self.ref_bgr.shape[:2]
            lines.append(f"Reference image: {Path(self.ref_path).name if self.ref_path else ''} ({rw}x{rh})")
        lines.append("")
        lines.append(f"Observed duration (filtered): {observed_s:.1f}s")
        lines.append(f"People seen (presence episodes): {episodes}")
        lines.append("")
        lines.append("Top visited grid cells (coarse):")
        if not top_cells:
            lines.append("  - (none)")
        else:
            for (gx, gy), c in top_cells:
                lines.append(f"  - cell({gx},{gy}) : {c} samples")
        self.stats.setText("\n".join(lines))

    def _grid_counts(self, samples: List[Sample], grid_w: int, grid_h: int) -> Dict[Tuple[int, int], int]:
        # Coarse grid over reference coordinate system.
        if not samples:
            return {}
        frame_w = samples[0].frame_w
        frame_h = samples[0].frame_h

        # If reference image exists, use its size as coordinate system; otherwise use frame size.
        if self.ref_bgr is not None:
            h, w = self.ref_bgr.shape[:2]
        else:
            w, h = frame_w, frame_h

        counts: Dict[Tuple[int, int], int] = {}
        for s in samples:
            x, y = self._map_to_ref_coords(s.torso_x, s.torso_y, frame_w, frame_h, w, h)
            gx = int(np.clip((x / max(1, w)) * grid_w, 0, grid_w - 1))
            gy = int(np.clip((y / max(1, h)) * grid_h, 0, grid_h - 1))
            counts[(gx, gy)] = counts.get((gx, gy), 0) + 1
        return counts

    def _map_to_ref_coords(self, x: int, y: int, frame_w: int, frame_h: int, ref_w: int, ref_h: int) -> Tuple[int, int]:
        """
        Map logged frame coordinates to reference image coordinates via simple scaling.
        Assumes same viewpoint; if you need true machine-plane mapping, add homography calibration later.
        """
        if frame_w <= 0 or frame_h <= 0:
            return x, y
        rx = int((x / frame_w) * ref_w)
        ry = int((y / frame_h) * ref_h)
        rx = int(np.clip(rx, 0, ref_w - 1))
        ry = int(np.clip(ry, 0, ref_h - 1))
        return rx, ry

    def _update_heatmap(self):
        if self.ref_bgr is None:
            # fallback: build a blank reference using the log frame size
            fw = self.samples[0].frame_w
            fh = self.samples[0].frame_h
            blank = np.zeros((fh, fw, 3), dtype=np.uint8)
            self.ref_bgr = blank

        ref = self.ref_bgr.copy()
        rh, rw = ref.shape[:2]

        if not self._filtered:
            self.view.set_image(to_qimage_bgr(ref))
            return

        # Collect points mapped to reference coordinates
        frame_w = self._filtered[0].frame_w
        frame_h = self._filtered[0].frame_h

        pts = []
        for s in self._filtered:
            if s.torso_x is None or s.torso_y is None:
                continue
            rx, ry = self._map_to_ref_coords(s.torso_x, s.torso_y, frame_w, frame_h, rw, rh)
            pts.append((rx, ry))

        points_xy = np.array(pts, dtype=np.int32) if pts else np.zeros((0, 2), dtype=np.int32)

        sigma = int(self.spin_sigma.value())
        alpha = float(self.spin_alpha.value())

        overlay, _heat = make_heatmap_overlay(ref, points_xy, sigma=sigma, alpha=alpha)
        self.view.set_image(to_qimage_bgr(overlay))

    def _update_stillness(self):
        if not self._filtered:
            self.tbl.setRowCount(0)
            return

        move_thresh = float(self.spin_move.value())
        min_still = float(self.spin_min_still.value())
        require_pose = bool(self.chk_pose.isChecked())

        # Stillness is computed in reference coordinates so it aligns with "workpoint"
        frame_w = self._filtered[0].frame_w
        frame_h = self._filtered[0].frame_h
        if self.ref_bgr is not None:
            ref_h, ref_w = self.ref_bgr.shape[:2]
        else:
            ref_w, ref_h = frame_w, frame_h

        remapped: List[Sample] = []
        for s in self._filtered:
            rx, ry = self._map_to_ref_coords(s.torso_x, s.torso_y, frame_w, frame_h, ref_w, ref_h)
            remapped.append(
                Sample(
                    ts=s.ts,
                    ts_iso=s.ts_iso,
                    run_id=s.run_id,
                    frame_idx=s.frame_idx,
                    camera_index=s.camera_index,
                    frame_w=ref_w,
                    frame_h=ref_h,
                    fps=s.fps,
                    pose_present=s.pose_present,
                    face_present=s.face_present,
                    torso_x=rx,
                    torso_y=ry,
                    bbox=s.bbox,
                )
            )

        intervals = compute_stillness_intervals(
            remapped,
            move_thresh_px=move_thresh,
            min_still_s=min_still,
            require_pose=require_pose,
        )

        self._last_intervals = intervals

        self.tbl.setRowCount(0)
        for it in intervals[:200]:  # cap
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)

            self.tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{it.duration_s:.1f}"))
            self.tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(fmt_ts(it.start_ts)))
            self.tbl.setItem(r, 2, QtWidgets.QTableWidgetItem(fmt_ts(it.end_ts)))
            self.tbl.setItem(r, 3, QtWidgets.QTableWidgetItem(f"({it.avg_x:.0f},{it.avg_y:.0f})"))

        self.tbl.resizeColumnsToContents()

    def _export_points_csv(self):
        if not self._filtered:
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export points CSV", "points.csv", "CSV (*.csv)")
        if not p:
            return

        # Export in reference coordinates
        frame_w = self._filtered[0].frame_w
        frame_h = self._filtered[0].frame_h
        if self.ref_bgr is not None:
            ref_h, ref_w = self.ref_bgr.shape[:2]
        else:
            ref_w, ref_h = frame_w, frame_h

        import csv
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts", "ts_iso", "run_id", "camera", "frame_idx", "x_ref", "y_ref", "pose_present", "face_present"])
            for s in self._filtered:
                xr, yr = self._map_to_ref_coords(s.torso_x, s.torso_y, frame_w, frame_h, ref_w, ref_h)
                w.writerow([s.ts, s.ts_iso, s.run_id, s.camera_index, s.frame_idx, xr, yr, int(s.pose_present), int(s.face_present)])

    def _export_intervals_csv(self):
        intervals = getattr(self, "_last_intervals", [])
        if not intervals:
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export stillness CSV", "stillness.csv", "CSV (*.csv)")
        if not p:
            return
        import csv
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start_ts", "end_ts", "duration_s", "avg_x", "avg_y"])
            for it in intervals:
                w.writerow([it.start_ts, it.end_ts, it.duration_s, it.avg_x, it.avg_y])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="", help="Path to run_*.jsonl")
    ap.add_argument("--ref", default="", help="Reference image path (png/jpg)")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(log_path=args.log or None, ref_path=args.ref or None)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    import sys
    main()

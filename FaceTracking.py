import sys
import time
import threading
import queue
import json
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# MediaPipe
import mediapipe as mp


# -----------------------------
# Utilities
# -----------------------------
def list_working_cameras(max_index: int = 10, warmup_frames: int = 3) -> list[int]:
    """Probe camera indices [0..max_index] and return those that can actually grab frames."""
    working = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        ok = False
        for _ in range(warmup_frames):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                ok = True
                break
            time.sleep(0.03)

        cap.release()
        if ok:
            working.append(idx)
    return working


def cv_bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    """Convert BGR OpenCV image to QImage (RGB888)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


def iso_utc(ts: float) -> str:
    """UTC ISO timestamp with milliseconds."""
    # Avoid datetime import overhead; keep it simple but precise enough for analytics.
    # If you prefer local time, replace with datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")
    import datetime as dt

    return dt.datetime.utcfromtimestamp(ts).isoformat(timespec="milliseconds") + "Z"


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# -----------------------------
# Settings
# -----------------------------
@dataclass
class TrackerSettings:
    # Camera
    camera_index: int = 0
    desired_width: int = 1280
    desired_height: int = 720

    enable_face: bool = True
    enable_pose: bool = True

    # FaceMesh settings
    face_max_faces: int = 1
    face_refine_landmarks: bool = True
    face_min_det_conf: float = 0.5
    face_min_track_conf: float = 0.5

    # Pose settings
    pose_model_complexity: int = 1  # 0,1,2
    pose_min_det_conf: float = 0.5
    pose_min_track_conf: float = 0.5

    draw_face: bool = True
    draw_pose: bool = True

    # Logging controls
    log_enabled: bool = True
    log_dir: str = "mp_logs"
    log_sample_hz: float = 5.0          # rate for "sample" records (pose + compact face)
    log_full_face_hz: float = 1.0       # rate for full 468-point face mesh records (0 disables)
    log_full_pose: bool = True          # log all 33 pose landmarks each sample
    log_compact_face: bool = True       # log compact face keypoints each sample
    log_full_face: bool = False         # log full face mesh at log_full_face_hz
    log_world_landmarks: bool = False   # if MP provides world landmarks, log them (pose)


# -----------------------------
# Analytics-friendly logger (JSONL)
# -----------------------------
class JsonlLogger:
    """
    Writes newline-delimited JSON (JSONL) with a small queue and a single writer thread.
    Each line is one record; easy to load into pandas, Spark, DuckDB, etc.
    """

    def __init__(self, out_dir: str, run_id: str, max_queue: int = 5000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

        self.path = self.out_dir / f"run_{run_id}.jsonl"
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue)
        self._stop_evt = threading.Event()
        self._thr = threading.Thread(target=self._writer, name="jsonl-writer", daemon=True)

        # Open file in line-buffered mode for safety
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)

        self._dropped = 0
        self._thr.start()

    def close(self):
        self._stop_evt.set()
        try:
            self._thr.join(timeout=1.5)
        except Exception:
            pass
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    @property
    def dropped(self) -> int:
        return self._dropped

    def log(self, record: Dict[str, Any]):
        if self._stop_evt.is_set():
            return
        try:
            self._q.put_nowait(record)
        except queue.Full:
            # Drop newest under load; keep UI/inference real-time
            self._dropped += 1

    def _writer(self):
        while not self._stop_evt.is_set() or not self._q.empty():
            try:
                rec = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._fh.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
            except Exception:
                # Last-resort: ignore write failures to not kill the app
                pass
            finally:
                try:
                    self._q.task_done()
                except Exception:
                    pass


# -----------------------------
# MediaPipe landmark packing
# -----------------------------
# FaceMesh: compact subset indices that give useful analytics without 468 points every sample.
# (These are commonly used stable points: nose tip, chin, outer/inner eyes, mouth corners)
_FACE_COMPACT_IDXS = [
    1,    # nose tip-ish
    152,  # chin
    33,   # left eye outer
    133,  # left eye inner
    362,  # right eye outer
    263,  # right eye inner
    61,   # mouth left
    291,  # mouth right
    13,   # upper lip
    14,   # lower lip
]

# Pose landmarks of interest for derived metrics
# (Indices taken from mp.solutions.pose.PoseLandmark enum)
# We'll reference by value at runtime.


def pack_norm_landmarks(lms, max_count: Optional[int] = None) -> List[List[float]]:
    """
    Pack MediaPipe normalized landmarks into a list of [x,y,z,vis,pres?].
    If presence isn't available, set -1.0.
    """
    out: List[List[float]] = []
    if lms is None:
        return out
    if max_count is None:
        max_count = len(lms)
    for i in range(min(max_count, len(lms))):
        lm = lms[i]
        x = float(lm.x)
        y = float(lm.y)
        z = float(getattr(lm, "z", 0.0))
        vis = float(getattr(lm, "visibility", -1.0))
        pres = float(getattr(lm, "presence", -1.0))
        out.append([x, y, z, vis, pres])
    return out


def pack_world_landmarks(lms, max_count: Optional[int] = None) -> List[List[float]]:
    """
    Pack MediaPipe world landmarks into list of [x,y,z,vis,pres?].
    """
    out: List[List[float]] = []
    if lms is None:
        return out
    if max_count is None:
        max_count = len(lms)
    for i in range(min(max_count, len(lms))):
        lm = lms[i]
        x = float(lm.x)
        y = float(lm.y)
        z = float(getattr(lm, "z", 0.0))
        vis = float(getattr(lm, "visibility", -1.0))
        pres = float(getattr(lm, "presence", -1.0))
        out.append([x, y, z, vis, pres])
    return out


def compute_pose_bbox_px(pose_lms_norm: List[List[float]], w: int, h: int, vis_gate: float = 0.2) -> Optional[Dict[str, int]]:
    """
    Compute bbox in pixel coords from pose landmarks, gating by visibility if available.
    pose_lms_norm: list of [x,y,z,vis,pres]
    """
    if not pose_lms_norm:
        return None

    xs: List[float] = []
    ys: List[float] = []
    for x, y, _z, vis, _pres in pose_lms_norm:
        if vis >= 0.0 and vis < vis_gate:
            continue
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return None

    x0 = int(max(0, min(w - 1, min(xs) * w)))
    y0 = int(max(0, min(h - 1, min(ys) * h)))
    x1 = int(max(0, min(w - 1, max(xs) * w)))
    y1 = int(max(0, min(h - 1, max(ys) * h)))
    if x1 <= x0 or y1 <= y0:
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "w": x1 - x0, "h": y1 - y0}


def compute_torso_center_px(pose_lms_norm: List[List[float]], w: int, h: int) -> Optional[Dict[str, int]]:
    """
    Torso center from shoulders+hips if available (normalized -> px).
    Uses PoseLandmark indices: L/R SHOULDER (11,12), L/R HIP (23,24)
    """
    if len(pose_lms_norm) < 25:
        return None
    idxs = [11, 12, 23, 24]
    pts = []
    for i in idxs:
        x, y, _z, vis, _pres = pose_lms_norm[i]
        if vis >= 0.0 and vis < 0.2:
            continue
        pts.append((x, y))
    if len(pts) < 2:
        return None
    cx = int(np.mean([p[0] for p in pts]) * w)
    cy = int(np.mean([p[1] for p in pts]) * h)
    return {"x": int(max(0, min(w - 1, cx))), "y": int(max(0, min(h - 1, cy)))}


# -----------------------------
# Worker thread (camera + inference + logging)
# -----------------------------
class TrackerWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(QtGui.QImage)
    stats_ready = QtCore.Signal(str)
    camera_list_ready = QtCore.Signal(list)
    log_status_ready = QtCore.Signal(str)

    def __init__(self, settings: TrackerSettings):
        super().__init__()
        self.settings = settings

        self._stop_evt = threading.Event()
        self._settings_lock = threading.Lock()
        self._cap = None

        # Mediapipe modules
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_pose = mp.solutions.pose
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._face_mesh = None
        self._pose = None

        # Logging
        self._logger: Optional[JsonlLogger] = None
        self._run_id = uuid.uuid4().hex[:10]
        self._last_sample_t = 0.0
        self._last_full_face_t = 0.0
        self._frame_idx = 0

        # Session metadata
        self._session_started = False

    def request_stop(self):
        self._stop_evt.set()

    def update_settings(self, **kwargs):
        with self._settings_lock:
            for k, v in kwargs.items():
                if hasattr(self.settings, k):
                    setattr(self.settings, k, v)

    def _open_camera(self, index: int):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.settings.desired_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.settings.desired_height))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _rebuild_models_if_needed(self, prev_settings: TrackerSettings, cur_settings: TrackerSettings):
        face_changed = (
            prev_settings.enable_face != cur_settings.enable_face
            or prev_settings.face_max_faces != cur_settings.face_max_faces
            or prev_settings.face_refine_landmarks != cur_settings.face_refine_landmarks
            or prev_settings.face_min_det_conf != cur_settings.face_min_det_conf
            or prev_settings.face_min_track_conf != cur_settings.face_min_track_conf
        )

        pose_changed = (
            prev_settings.enable_pose != cur_settings.enable_pose
            or prev_settings.pose_model_complexity != cur_settings.pose_model_complexity
            or prev_settings.pose_min_det_conf != cur_settings.pose_min_det_conf
            or prev_settings.pose_min_track_conf != cur_settings.pose_min_track_conf
        )

        if face_changed:
            if self._face_mesh is not None:
                try:
                    self._face_mesh.close()
                except Exception:
                    pass
                self._face_mesh = None

            if cur_settings.enable_face:
                self._face_mesh = self._mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=int(cur_settings.face_max_faces),
                    refine_landmarks=bool(cur_settings.face_refine_landmarks),
                    min_detection_confidence=float(cur_settings.face_min_det_conf),
                    min_tracking_confidence=float(cur_settings.face_min_track_conf),
                )

        if pose_changed:
            if self._pose is not None:
                try:
                    self._pose.close()
                except Exception:
                    pass
                self._pose = None

            if cur_settings.enable_pose:
                self._pose = self._mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=int(cur_settings.pose_model_complexity),
                    enable_segmentation=False,
                    min_detection_confidence=float(cur_settings.pose_min_det_conf),
                    min_tracking_confidence=float(cur_settings.pose_min_track_conf),
                )

    def _ensure_logger(self, cur: TrackerSettings):
        if not cur.log_enabled:
            if self._logger is not None:
                self._logger.close()
                self._logger = None
            return

        if self._logger is None:
            self._logger = JsonlLogger(cur.log_dir, self._run_id)
            self.log_status_ready.emit(f"Logging ON -> {self._logger.path}")
            # Session start record
            self._logger.log(
                {
                    "type": "session_start",
                    "ts": time.time(),
                    "ts_iso": iso_utc(time.time()),
                    "run_id": self._run_id,
                    "settings": asdict(cur),
                    "platform": {"opencv": cv2.__version__, "mediapipe": getattr(mp, "__version__", "unknown")},
                }
            )
            self._session_started = True

    def _log_sample(
        self,
        cur: TrackerSettings,
        ts: float,
        fps: float,
        frame_w: int,
        frame_h: int,
        pose_norm: Optional[List[List[float]]],
        pose_world: Optional[List[List[float]]],
        face_compact: Optional[List[Dict[str, Any]]],
        face_full: Optional[List[Dict[str, Any]]],
    ):
        if self._logger is None:
            return

        rec: Dict[str, Any] = {
            "type": "sample",
            "ts": ts,
            "ts_iso": iso_utc(ts),
            "run_id": self._run_id,
            "frame_idx": int(self._frame_idx),
            "camera_index": int(cur.camera_index),
            "frame": {"w": int(frame_w), "h": int(frame_h)},
            "fps": float(fps),
            "flags": {
                "pose_enabled": bool(cur.enable_pose),
                "face_enabled": bool(cur.enable_face),
                "pose_present": bool(pose_norm and len(pose_norm) >= 25),
                "face_present": bool(face_compact and len(face_compact) > 0),
            },
        }

        # Derived pose features (helpful for visualizers)
        if pose_norm:
            bbox = compute_pose_bbox_px(pose_norm, frame_w, frame_h)
            torso = compute_torso_center_px(pose_norm, frame_w, frame_h)
            rec["pose_derived"] = {
                "bbox_px": bbox,
                "torso_center_px": torso,
            }

        # Pose landmarks
        if cur.enable_pose and pose_norm is not None and cur.log_full_pose:
            rec["pose_landmarks_norm"] = pose_norm
            if cur.log_world_landmarks and pose_world is not None:
                rec["pose_landmarks_world"] = pose_world

        # Compact face (keypoints)
        if cur.enable_face and face_compact is not None and cur.log_compact_face:
            rec["face_compact"] = face_compact

        # Full face mesh is put in separate record (or optionally attached, but it’s big)
        self._logger.log(rec)

        if face_full is not None:
            full_rec = {
                "type": "face_full",
                "ts": ts,
                "ts_iso": iso_utc(ts),
                "run_id": self._run_id,
                "frame_idx": int(self._frame_idx),
                "camera_index": int(cur.camera_index),
                "frame": {"w": int(frame_w), "h": int(frame_h)},
                "faces": face_full,
            }
            self._logger.log(full_rec)

        # Periodically report drops
        if (self._frame_idx % 150) == 0 and self._logger is not None:
            d = self._logger.dropped
            if d > 0:
                self.log_status_ready.emit(f"Log queue overflow: dropped={d}")

    def run(self):
        cams = list_working_cameras(max_index=10)
        self.camera_list_ready.emit(cams)

        with self._settings_lock:
            cam_idx = self.settings.camera_index
        self._cap = self._open_camera(cam_idx)

        prev = TrackerSettings(**vars(self.settings))
        self._rebuild_models_if_needed(prev, self.settings)
        prev = TrackerSettings(**vars(self.settings))

        last_t = time.time()
        fps = 0.0
        frame_count = 0
        fps_window_start = time.time()

        while not self._stop_evt.is_set():
            self._frame_idx += 1

            # Snapshot settings
            with self._settings_lock:
                cur = TrackerSettings(**vars(self.settings))

            # Logger setup
            self._ensure_logger(cur)

            # Camera switch handling
            if prev.camera_index != cur.camera_index:
                self._cap = self._open_camera(cur.camera_index)
                prev.camera_index = cur.camera_index

            # Model rebuilds
            self._rebuild_models_if_needed(prev, cur)
            prev = TrackerSettings(**vars(cur))

            if self._cap is None:
                self.stats_ready.emit("Camera: NOT OPEN | Retry...")
                time.sleep(0.3)
                self._cap = self._open_camera(cur.camera_index)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                self.stats_ready.emit("Camera: NO FRAME | Retry...")
                time.sleep(0.05)
                continue

            # Mirror for user-facing webcams
            frame = cv2.flip(frame, 1)

            # Inference expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Results placeholders
            pose_norm: Optional[List[List[float]]] = None
            pose_world: Optional[List[List[float]]] = None
            face_compact: Optional[List[Dict[str, Any]]] = None
            face_full: Optional[List[Dict[str, Any]]] = None

            # ---- Face tracking ----
            if cur.enable_face and self._face_mesh is not None:
                face_res = self._face_mesh.process(rgb)
                if face_res.multi_face_landmarks:
                    # Compact face points each sample (small)
                    if cur.log_compact_face:
                        face_compact = []
                        for fi, face_lms in enumerate(face_res.multi_face_landmarks):
                            pts = face_lms.landmark
                            compact_pts = []
                            for idx in _FACE_COMPACT_IDXS:
                                if idx < len(pts):
                                    lm = pts[idx]
                                    compact_pts.append(
                                        {
                                            "i": int(idx),
                                            "x": float(lm.x),
                                            "y": float(lm.y),
                                            "z": float(getattr(lm, "z", 0.0)),
                                        }
                                    )
                            face_compact.append({"face_i": int(fi), "pts": compact_pts})

                    # Full face mesh at a slower rate (optional)
                    if cur.log_full_face and cur.log_full_face_hz > 0:
                        ts_now = time.time()
                        interval = 1.0 / float(max(0.1, cur.log_full_face_hz))
                        if (ts_now - self._last_full_face_t) >= interval:
                            self._last_full_face_t = ts_now
                            face_full = []
                            for fi, face_lms in enumerate(face_res.multi_face_landmarks):
                                face_full.append(
                                    {
                                        "face_i": int(fi),
                                        "landmarks_norm": pack_norm_landmarks(face_lms.landmark),
                                    }
                                )

                    # Draw face
                    if cur.draw_face:
                        for face_landmarks in face_res.multi_face_landmarks:
                            self._mp_draw.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=self._mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self._mp_styles.get_default_face_mesh_tesselation_style(),
                            )
                            self._mp_draw.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=self._mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self._mp_styles.get_default_face_mesh_contours_style(),
                            )

            # ---- Body tracking ----
            if cur.enable_pose and self._pose is not None:
                pose_res = self._pose.process(rgb)
                if pose_res.pose_landmarks:
                    pose_norm = pack_norm_landmarks(pose_res.pose_landmarks.landmark)
                if cur.log_world_landmarks and getattr(pose_res, "pose_world_landmarks", None):
                    pose_world = pack_world_landmarks(pose_res.pose_world_landmarks.landmark)

                if cur.draw_pose and pose_res.pose_landmarks:
                    self._mp_draw.draw_landmarks(
                        image=frame,
                        landmark_list=pose_res.pose_landmarks,
                        connections=self._mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
                    )

            # FPS calc
            frame_count += 1
            now = time.time()
            if now - fps_window_start >= 0.5:
                fps = frame_count / (now - fps_window_start)
                fps_window_start = now
                frame_count = 0

            h, w = frame.shape[:2]
            stats = f"Camera {cur.camera_index} | {w}x{h} | FPS: {fps:.1f} | RunID: {self._run_id}"
            self.stats_ready.emit(stats)

            # Logging at fixed sample rate (analytics-friendly)
            if cur.log_enabled and self._logger is not None and cur.log_sample_hz > 0:
                interval = 1.0 / float(max(0.1, cur.log_sample_hz))
                if (now - self._last_sample_t) >= interval:
                    self._last_sample_t = now
                    self._log_sample(
                        cur=cur,
                        ts=now,
                        fps=fps,
                        frame_w=w,
                        frame_h=h,
                        pose_norm=pose_norm,
                        pose_world=pose_world,
                        face_compact=face_compact,
                        face_full=face_full,  # this will be None unless scheduled
                    )

            # Emit frame
            qimg = cv_bgr_to_qimage(frame)
            self.frame_ready.emit(qimg)

            # Small sleep to reduce CPU spin
            dt_loop = time.time() - last_t
            last_t = time.time()
            if dt_loop < 0.001:
                time.sleep(0.001)

        # Cleanup
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass
            self._face_mesh = None

        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass
            self._pose = None

        # Log session end
        if self._logger is not None:
            try:
                self._logger.log(
                    {
                        "type": "session_end",
                        "ts": time.time(),
                        "ts_iso": iso_utc(time.time()),
                        "run_id": self._run_id,
                        "frame_idx": int(self._frame_idx),
                        "dropped": int(self._logger.dropped),
                    }
                )
            except Exception:
                pass
            self._logger.close()
            self._logger = None


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Face + Body Tracking (MediaPipe + OpenCV) + JSONL Logging")
        self.resize(1200, 860)

        self.settings = TrackerSettings()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.video_label = QtWidgets.QLabel("No video yet")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background: #111; color: #ddd; border: 1px solid #333;")

        self.stats_label = QtWidgets.QLabel("Stats: -")
        self.stats_label.setStyleSheet("color: #ddd;")

        self.log_label = QtWidgets.QLabel("Log: -")
        self.log_label.setStyleSheet("color: #9ad;")

        controls = self._build_controls()

        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.stats_label)
        layout.addWidget(self.log_label)
        layout.addWidget(controls)

        # Worker
        self.worker = TrackerWorker(self.settings)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.stats_ready.connect(self.on_stats)
        self.worker.camera_list_ready.connect(self.on_camera_list)
        self.worker.log_status_ready.connect(self.on_log_status)

        self.worker.start()

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.worker.request_stop()
            self.worker.wait(1500)
        except Exception:
            pass
        event.accept()

    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(w)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        # Camera dropdown
        self.cmb_camera = QtWidgets.QComboBox()
        self.cmb_camera.addItem("Probing cameras...")
        self.cmb_camera.currentIndexChanged.connect(self._on_camera_changed)

        # Resolution
        self.spin_w = QtWidgets.QSpinBox()
        self.spin_w.setRange(160, 4096)
        self.spin_w.setValue(self.settings.desired_width)
        self.spin_w.valueChanged.connect(lambda v: self.worker.update_settings(desired_width=int(v)))

        self.spin_h = QtWidgets.QSpinBox()
        self.spin_h.setRange(120, 2160)
        self.spin_h.setValue(self.settings.desired_height)
        self.spin_h.valueChanged.connect(lambda v: self.worker.update_settings(desired_height=int(v)))

        # Toggles
        self.chk_face = QtWidgets.QCheckBox("Enable Face")
        self.chk_face.setChecked(self.settings.enable_face)
        self.chk_face.toggled.connect(lambda v: self.worker.update_settings(enable_face=bool(v)))

        self.chk_pose = QtWidgets.QCheckBox("Enable Pose")
        self.chk_pose.setChecked(self.settings.enable_pose)
        self.chk_pose.toggled.connect(lambda v: self.worker.update_settings(enable_pose=bool(v)))

        self.chk_draw_face = QtWidgets.QCheckBox("Draw Face")
        self.chk_draw_face.setChecked(self.settings.draw_face)
        self.chk_draw_face.toggled.connect(lambda v: self.worker.update_settings(draw_face=bool(v)))

        self.chk_draw_pose = QtWidgets.QCheckBox("Draw Pose")
        self.chk_draw_pose.setChecked(self.settings.draw_pose)
        self.chk_draw_pose.toggled.connect(lambda v: self.worker.update_settings(draw_pose=bool(v)))

        # Face settings
        self.spin_faces = QtWidgets.QSpinBox()
        self.spin_faces.setRange(1, 4)
        self.spin_faces.setValue(self.settings.face_max_faces)
        self.spin_faces.valueChanged.connect(lambda v: self.worker.update_settings(face_max_faces=int(v)))

        self.chk_refine = QtWidgets.QCheckBox("Refine Landmarks")
        self.chk_refine.setChecked(self.settings.face_refine_landmarks)
        self.chk_refine.toggled.connect(lambda v: self.worker.update_settings(face_refine_landmarks=bool(v)))

        self.face_det = QtWidgets.QDoubleSpinBox()
        self.face_det.setRange(0.0, 1.0)
        self.face_det.setSingleStep(0.05)
        self.face_det.setValue(self.settings.face_min_det_conf)
        self.face_det.valueChanged.connect(lambda v: self.worker.update_settings(face_min_det_conf=float(v)))

        self.face_track = QtWidgets.QDoubleSpinBox()
        self.face_track.setRange(0.0, 1.0)
        self.face_track.setSingleStep(0.05)
        self.face_track.setValue(self.settings.face_min_track_conf)
        self.face_track.valueChanged.connect(lambda v: self.worker.update_settings(face_min_track_conf=float(v)))

        # Pose settings
        self.pose_complex = QtWidgets.QComboBox()
        self.pose_complex.addItems(["0 (fast)", "1 (balanced)", "2 (accurate)"])
        self.pose_complex.setCurrentIndex(self.settings.pose_model_complexity)
        self.pose_complex.currentIndexChanged.connect(lambda i: self.worker.update_settings(pose_model_complexity=int(i)))

        self.pose_det = QtWidgets.QDoubleSpinBox()
        self.pose_det.setRange(0.0, 1.0)
        self.pose_det.setSingleStep(0.05)
        self.pose_det.setValue(self.settings.pose_min_det_conf)
        self.pose_det.valueChanged.connect(lambda v: self.worker.update_settings(pose_min_det_conf=float(v)))

        self.pose_track = QtWidgets.QDoubleSpinBox()
        self.pose_track.setRange(0.0, 1.0)
        self.pose_track.setSingleStep(0.05)
        self.pose_track.setValue(self.settings.pose_min_track_conf)
        self.pose_track.valueChanged.connect(lambda v: self.worker.update_settings(pose_min_track_conf=float(v)))

        # Logging controls
        self.chk_log = QtWidgets.QCheckBox("Enable Logging (JSONL)")
        self.chk_log.setChecked(self.settings.log_enabled)
        self.chk_log.toggled.connect(lambda v: self.worker.update_settings(log_enabled=bool(v)))

        self.spin_log_hz = QtWidgets.QDoubleSpinBox()
        self.spin_log_hz.setRange(0.2, 60.0)
        self.spin_log_hz.setDecimals(1)
        self.spin_log_hz.setSingleStep(0.5)
        self.spin_log_hz.setValue(self.settings.log_sample_hz)
        self.spin_log_hz.valueChanged.connect(lambda v: self.worker.update_settings(log_sample_hz=float(v)))

        self.chk_full_face = QtWidgets.QCheckBox("Log Full Face Mesh (468)")
        self.chk_full_face.setChecked(self.settings.log_full_face)
        self.chk_full_face.toggled.connect(lambda v: self.worker.update_settings(log_full_face=bool(v)))

        self.spin_full_face_hz = QtWidgets.QDoubleSpinBox()
        self.spin_full_face_hz.setRange(0.2, 10.0)
        self.spin_full_face_hz.setDecimals(1)
        self.spin_full_face_hz.setSingleStep(0.2)
        self.spin_full_face_hz.setValue(self.settings.log_full_face_hz)
        self.spin_full_face_hz.valueChanged.connect(lambda v: self.worker.update_settings(log_full_face_hz=float(v)))

        self.chk_world = QtWidgets.QCheckBox("Log Pose World Landmarks")
        self.chk_world.setChecked(self.settings.log_world_landmarks)
        self.chk_world.toggled.connect(lambda v: self.worker.update_settings(log_world_landmarks=bool(v)))

        # Layout
        r = 0
        grid.addWidget(QtWidgets.QLabel("Camera:"), r, 0)
        grid.addWidget(self.cmb_camera, r, 1, 1, 3)
        r += 1

        grid.addWidget(QtWidgets.QLabel("Desired Resolution:"), r, 0)
        grid.addWidget(self.spin_w, r, 1)
        grid.addWidget(QtWidgets.QLabel("x"), r, 2)
        grid.addWidget(self.spin_h, r, 3)
        r += 1

        grid.addWidget(self.chk_face, r, 0)
        grid.addWidget(self.chk_draw_face, r, 1)
        grid.addWidget(self.chk_pose, r, 2)
        grid.addWidget(self.chk_draw_pose, r, 3)
        r += 1

        face_box = QtWidgets.QGroupBox("Face Tracking (FaceMesh)")
        fg = QtWidgets.QGridLayout(face_box)
        fg.addWidget(QtWidgets.QLabel("Max Faces:"), 0, 0)
        fg.addWidget(self.spin_faces, 0, 1)
        fg.addWidget(self.chk_refine, 0, 2, 1, 2)
        fg.addWidget(QtWidgets.QLabel("Min Det Conf:"), 1, 0)
        fg.addWidget(self.face_det, 1, 1)
        fg.addWidget(QtWidgets.QLabel("Min Track Conf:"), 1, 2)
        fg.addWidget(self.face_track, 1, 3)

        pose_box = QtWidgets.QGroupBox("Body Tracking (Pose)")
        pg = QtWidgets.QGridLayout(pose_box)
        pg.addWidget(QtWidgets.QLabel("Model:"), 0, 0)
        pg.addWidget(self.pose_complex, 0, 1, 1, 3)
        pg.addWidget(QtWidgets.QLabel("Min Det Conf:"), 1, 0)
        pg.addWidget(self.pose_det, 1, 1)
        pg.addWidget(QtWidgets.QLabel("Min Track Conf:"), 1, 2)
        pg.addWidget(self.pose_track, 1, 3)

        log_box = QtWidgets.QGroupBox("Logging (Analytics)")
        lg = QtWidgets.QGridLayout(log_box)
        lg.addWidget(self.chk_log, 0, 0, 1, 2)
        lg.addWidget(QtWidgets.QLabel("Sample Hz:"), 1, 0)
        lg.addWidget(self.spin_log_hz, 1, 1)
        lg.addWidget(self.chk_full_face, 2, 0, 1, 2)
        lg.addWidget(QtWidgets.QLabel("Full Face Hz:"), 3, 0)
        lg.addWidget(self.spin_full_face_hz, 3, 1)
        lg.addWidget(self.chk_world, 4, 0, 1, 2)

        grid.addWidget(face_box, r, 0, 1, 4)
        r += 1
        grid.addWidget(pose_box, r, 0, 1, 4)
        r += 1
        grid.addWidget(log_box, r, 0, 1, 4)

        return w

    @QtCore.Slot(list)
    def on_camera_list(self, cams: list[int]):
        self.cmb_camera.blockSignals(True)
        self.cmb_camera.clear()
        if not cams:
            self.cmb_camera.addItem("No cameras found")
            self.cmb_camera.setEnabled(False)
            self.cmb_camera.blockSignals(False)
            return

        for idx in cams:
            self.cmb_camera.addItem(f"Camera {idx}", idx)

        found = False
        for i in range(self.cmb_camera.count()):
            if self.cmb_camera.itemData(i) == self.settings.camera_index:
                self.cmb_camera.setCurrentIndex(i)
                found = True
                break
        if not found:
            self.cmb_camera.setCurrentIndex(0)
            self.worker.update_settings(camera_index=int(self.cmb_camera.itemData(0)))

        self.cmb_camera.setEnabled(True)
        self.cmb_camera.blockSignals(False)

    def _on_camera_changed(self, idx: int):
        data = self.cmb_camera.itemData(idx)
        if data is None:
            return
        self.worker.update_settings(camera_index=int(data))

    @QtCore.Slot(QtGui.QImage)
    def on_frame(self, qimg: QtGui.QImage):
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    @QtCore.Slot(str)
    def on_stats(self, text: str):
        self.stats_label.setText(f"Stats: {text}")

    @QtCore.Slot(str)
    def on_log_status(self, text: str):
        self.log_label.setText(f"Log: {text}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

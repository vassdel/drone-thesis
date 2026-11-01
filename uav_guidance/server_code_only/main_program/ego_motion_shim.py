"""
Moving-camera pixel <-> metric world conversion (ego-motion compensation).

Drop-in replacement for `PixelMetricShim` when the UAV is moving rather than
hovering. Implements the same four-method API plus one lifecycle method:

    status = shim.update_frame(frame_bgr, mask_bboxes_ltwh=None)

Call `update_frame` once per video frame (after detection so the bbox mask is
ready, before any pixel<->metric calls for that frame). It estimates the
homography H_live -> K0 (current frame to canonical keyframe-zero pixel space)
via ORB keypoints + USAC_MAGSAC, then composes it with the static H_K0 ->
world from the base shim.

Reference-frame strategy: frame-to-keyframe with sliding re-anchoring. Pairwise
chaining drifts catastrophically; a fixed K0 breaks once the UAV pans away from
the original view. Instead we re-anchor to a fresh keyframe when match quality
degrades, keeping the number of chained multiplications small over a clip.

Detector / matcher / RANSAC: ORB + BFMatcher(NORM_HAMMING) + KNN k=2 + Lowe
ratio, with USAC_MAGSAC for the homography fit. Tuned values are in the module
constants below; they require cv2 >= 4.5.2 for USAC_MAGSAC.

Public API contract (mirrors PixelMetricShim):
    shim = MovingCameraShim(map_pickle_path)
    shim.update_frame(frame_bgr, mask_bboxes_ltwh=[...])   # per frame
    x_m, y_m = shim.pixel_to_metric(u, v)
    u, v     = shim.metric_to_pixel(x_m, y_m)
    xy_m  = shim.pixels_to_metric(uv_array)    # (N, 2) -> (N, 2)
    uv    = shim.metrics_to_pixel(xy_m_array)  # (N, 2) -> (N, 2)
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from pixel_metric_shim import PixelMetricShim


# Detector / matcher / RANSAC knobs.
# nfeatures=5000 (vs a 3000 baseline): after masking moving agents, the Lowe
# ratio test, and RANSAC inlier removal, we still want >=200 clean inliers for
# a tight fit. A 1.5 px reproj threshold keeps corner lever-arm error under
# ~3 px (USAC_MAGSAC uses a dataset-invariant threshold).
_ORB_DEFAULTS = dict(
    nfeatures=5000, scaleFactor=1.2, nlevels=8,
    edgeThreshold=31, fastThreshold=20, patchSize=31,
)
_LOWE_RATIO = 0.75
_MIN_GOOD_MATCHES = 50
_RANSAC_REPROJ_THRESH_PX = 1.5
_RANSAC_MAX_ITERS = 2000
_RANSAC_CONFIDENCE = 0.999
_MIN_INLIERS_OK = 50
_MIN_INLIER_RATIO = 0.4
_REANCHOR_TRANSLATION_FRAC = 0.40         # of frame width
# Quality-based re-anchor floor: promote Kc to a fresher frame when the
# live->Kc inlier ratio drops below this. On a slowly panning camera the
# match quality against a pinned keyframe decays into the _MIN_INLIER_RATIO
# acceptance gate long before translation reaches 40% of frame width, so
# without this trigger most frames would go FROZEN. 0.55 keeps a comfortable
# margin above the 0.4 gate.
_REANCHOR_INLIER_RATIO_FLOOR = 0.55
_DILATE_FRAC = 0.18                       # of max(bbox_w, bbox_h)
_DILATE_MIN_PX = 20


class MovingCameraShim:
    """
    Stateful pixel <-> metric converter for moving-UAV footage.

    Composes a static base shim (PixelMetricShim or ScaleOnlyShim) for the
    K0 -> world step. Each `update_frame` caches the per-frame estimate
    H_live -> K0; the four conversion methods are then O(1) homography applies.
    """

    def __init__(
        self,
        map_pickle_path: Optional[str] = None,
        *,
        base_shim: Optional[object] = None,
        detector: str = "ORB",
        **detector_kwargs,
    ):
        if detector != "ORB":
            # Only ORB is implemented; raise rather than silently downgrade.
            raise NotImplementedError(
                f"detector={detector!r} not supported; ORB only for now."
            )
        # Static K0->world base shim. Supply exactly one of:
        #   - map_pickle_path -> wrap a fresh PixelMetricShim (inD/uniD/rounD);
        #   - base_shim       -> a caller-built four-method shim (e.g.
        #     ScaleOnlyShim for VisDrone). Duck-typed, not checked explicitly.
        if base_shim is not None and map_pickle_path is not None:
            raise ValueError(
                "MovingCameraShim: specify exactly one of `map_pickle_path` "
                "or `base_shim`, not both."
            )
        if base_shim is None and map_pickle_path is None:
            raise ValueError(
                "MovingCameraShim: must specify either `map_pickle_path` "
                "(constructs a PixelMetricShim) or `base_shim` (any "
                "four-method shim, e.g. ScaleOnlyShim)."
            )
        self._base = (
            base_shim if base_shim is not None else PixelMetricShim(map_pickle_path)
        )
        orb_kwargs = {**_ORB_DEFAULTS, **detector_kwargs}
        self._orb = cv2.ORB_create(**orb_kwargs)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Keyframe state, set lazily on first update_frame(). K0's
        # keypoints/descriptors are kept permanently (separate from Kc) so a
        # re-anchor can refit H_Knew -> K0 by matching directly against K0,
        # avoiding the cascading drift of pure-chain re-anchoring. Kc is the
        # current keyframe used for the per-frame H_live -> Kc fit.
        self._kp_K0: Optional[List[cv2.KeyPoint]] = None
        self._des_K0: Optional[np.ndarray] = None
        self._kp_Kc: Optional[List[cv2.KeyPoint]] = None
        self._des_Kc: Optional[np.ndarray] = None
        self._H_Kc_to_K0: Optional[np.ndarray] = None   # 3x3
        self._K0_shape_hw: Optional[Tuple[int, int]] = None  # (H, W) of K0 image

        # Per-frame state. Set on every successful update_frame().
        self._H_live_to_K0: Optional[np.ndarray] = None
        self._H_K0_to_live: Optional[np.ndarray] = None  # inverse, cached
        self._last_status: str = "UNINITIALIZED"

        # Diagnostic counters, exposed for the test harness.
        self.n_frames_seen = 0
        self.n_reanchors = 0
        self.n_frozen = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def update_frame(
        self,
        frame_bgr: np.ndarray,
        mask_bboxes_ltwh: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> str:
        """
        Estimate H_live -> K0 for this frame; return status:
          - "OK"          : fit succeeded, no re-anchor
          - "REANCHORED"  : fit succeeded AND we promoted live to new keyframe
          - "FROZEN"      : fit failed; previous H_live_to_K0 is kept

        First call ever bootstraps K0 (no estimation; status "OK").
        """
        self.n_frames_seen += 1
        H_img, W_img = frame_bgr.shape[:2]

        # Build the keypoint-detection mask: exclude dilated bbox interiors.
        kp_mask = self._build_mask(H_img, W_img, mask_bboxes_ltwh)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp_live, des_live = self._orb.detectAndCompute(gray, kp_mask)

        # Bootstrap: this frame IS K0 (and the first Kc).
        if self._kp_Kc is None:
            if des_live is None or len(kp_live) < _MIN_GOOD_MATCHES:
                # Accept as K0 regardless — there is nothing to match against
                # yet. Too few features here only hurts later-frame fits.
                pass
            self._kp_K0 = kp_live
            self._des_K0 = des_live
            self._kp_Kc = kp_live
            self._des_Kc = des_live
            self._H_Kc_to_K0 = np.eye(3, dtype=np.float64)
            self._K0_shape_hw = (H_img, W_img)
            self._H_live_to_K0 = np.eye(3, dtype=np.float64)
            self._H_K0_to_live = np.eye(3, dtype=np.float64)
            self._last_status = "OK"
            return "OK"

        # Not enough live descriptors -> FROZEN.
        if des_live is None or len(kp_live) < _MIN_GOOD_MATCHES:
            self._last_status = "FROZEN"
            self.n_frozen += 1
            return "FROZEN"

        # Per-frame fit H_live -> Kc; drives both the conversion API and the
        # re-anchor decision.
        H_live_to_Kc, n_inliers_Kc, n_good_Kc = self._fit_homography(
            kp_live, des_live, self._kp_Kc, self._des_Kc,
        )
        if H_live_to_Kc is None:
            self._last_status = "FROZEN"
            self.n_frozen += 1
            return "FROZEN"
        inlier_ratio_Kc = (n_inliers_Kc / n_good_Kc) if n_good_Kc > 0 else 0.0

        # Compose to K0 frame via the cached Kc->K0 chain.
        H_live_to_K0 = (
            self._H_Kc_to_K0.astype(np.float64)
            @ H_live_to_Kc.astype(np.float64)
        )

        # Re-anchor if EITHER trigger fires:
        #   (1) translation > 40% frame width — coarse "drifted a lot" signal;
        #   (2) live->Kc inlier ratio below the quality floor — catches slow
        #       pans where translation stays under 40% W but match quality
        #       decays into the rejection gate, which would otherwise freeze
        #       most frames on long natural-pan clips.
        # We deliberately do NOT trigger on absolute inlier count: it
        # oscillates near the gate and causes re-anchor cascades.
        translation_mag = float(
            np.hypot(H_live_to_Kc[0, 2], H_live_to_Kc[1, 2])
        )
        do_reanchor = (
            translation_mag > _REANCHOR_TRANSLATION_FRAC * W_img
            or inlier_ratio_Kc < _REANCHOR_INLIER_RATIO_FLOOR
        )

        if do_reanchor:
            # Refit H_newKc -> K0 directly against K0's descriptors (kept since
            # init) rather than reuse the chained estimate, which would bake in
            # per-frame RANSAC noise and amplify it on each re-anchor.
            H_new_Kc_to_K0, _, _ = self._fit_homography(
                kp_live, des_live, self._kp_K0, self._des_K0,
            )
            if H_new_Kc_to_K0 is not None:
                self._H_Kc_to_K0 = H_new_Kc_to_K0
                H_live_to_K0 = H_new_Kc_to_K0   # current frame == new Kc
            else:
                # K0 out of view; fall back to the chained estimate (with its
                # accumulated error) as the new anchor.
                self._H_Kc_to_K0 = H_live_to_K0.copy()
            self._kp_Kc = kp_live
            self._des_Kc = des_live
            self._H_live_to_K0 = H_live_to_K0
            try:
                self._H_K0_to_live = np.linalg.inv(H_live_to_K0)
            except np.linalg.LinAlgError:
                self._last_status = "FROZEN"
                self.n_frozen += 1
                return "FROZEN"
            self._last_status = "REANCHORED"
            self.n_reanchors += 1
            return "REANCHORED"

        self._H_live_to_K0 = H_live_to_K0
        try:
            self._H_K0_to_live = np.linalg.inv(H_live_to_K0)
        except np.linalg.LinAlgError:
            self._last_status = "FROZEN"
            self.n_frozen += 1
            return "FROZEN"
        self._last_status = "OK"
        return "OK"

    def _fit_homography(
        self,
        kp_src, des_src, kp_dst, des_dst,
    ) -> Tuple[Optional[np.ndarray], int, int]:
        """
        KNN+Lowe+USAC_MAGSAC fit of `H src -> dst`.

        Returns
        -------
        (H, n_inliers, n_good) — H is None when the fit fails any gate.
        `n_good` is the post-Lowe match count BEFORE RANSAC; callers use
        `n_inliers / n_good` as the inlier ratio for re-anchor decisions.
        """
        if des_src is None or des_dst is None:
            return None, 0, 0
        try:
            knn = self._bf.knnMatch(des_src, des_dst, k=2)
        except cv2.error:
            return None, 0, 0
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < _LOWE_RATIO * n.distance:
                good.append(m)
        if len(good) < _MIN_GOOD_MATCHES:
            return None, 0, len(good)
        src_pts = np.array(
            [kp_src[m.queryIdx].pt for m in good], dtype=np.float32
        ).reshape(-1, 1, 2)
        dst_pts = np.array(
            [kp_dst[m.trainIdx].pt for m in good], dtype=np.float32
        ).reshape(-1, 1, 2)
        H, inlier_mask = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=_RANSAC_REPROJ_THRESH_PX,
            maxIters=_RANSAC_MAX_ITERS,
            confidence=_RANSAC_CONFIDENCE,
        )
        if H is None or inlier_mask is None:
            return None, 0, len(good)
        n_inliers = int(inlier_mask.sum())
        if n_inliers < _MIN_INLIERS_OK or n_inliers / len(good) < _MIN_INLIER_RATIO:
            return None, 0, len(good)
        return H, n_inliers, len(good)

    @staticmethod
    def _build_mask(
        H: int, W: int,
        bboxes_ltwh: Optional[List[Tuple[float, float, float, float]]],
    ) -> np.ndarray:
        mask = np.full((H, W), 255, dtype=np.uint8)
        if not bboxes_ltwh:
            return mask
        for ltwh in bboxes_ltwh:
            x, y, w, h = ltwh
            pad = max(_DILATE_MIN_PX, int(_DILATE_FRAC * max(w, h)))
            x0 = max(int(x) - pad, 0)
            y0 = max(int(y) - pad, 0)
            x1 = min(int(x + w) + pad, W)
            y1 = min(int(y + h) + pad, H)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 0
        return mask

    # ------------------------------------------------------------------
    # Conversion API (PixelMetricShim-compatible)
    # ------------------------------------------------------------------

    def _require_initialized(self) -> None:
        if self._H_live_to_K0 is None:
            raise RuntimeError(
                "MovingCameraShim.update_frame() must be called at least once "
                "before any pixel<->metric conversion."
            )

    def pixel_to_metric(self, u: float, v: float) -> Tuple[float, float]:
        self._require_initialized()
        # live (u, v) -> K0, then K0 -> world via the base shim.
        p = self._H_live_to_K0 @ np.array([u, v, 1.0], dtype=np.float64)
        if abs(p[2]) < 1e-12:
            raise RuntimeError(
                f"pixel_to_metric: degenerate homog coord at live ({u}, {v})."
            )
        u_k0, v_k0 = float(p[0] / p[2]), float(p[1] / p[2])
        return self._base.pixel_to_metric(u_k0, v_k0)

    def metric_to_pixel(self, x_m: float, y_m: float) -> Tuple[float, float]:
        self._require_initialized()
        u_k0, v_k0 = self._base.metric_to_pixel(x_m, y_m)
        p = self._H_K0_to_live @ np.array([u_k0, v_k0, 1.0], dtype=np.float64)
        if abs(p[2]) < 1e-12:
            raise RuntimeError(
                f"metric_to_pixel: degenerate homog coord at K0 ({u_k0}, {v_k0})."
            )
        return float(p[0] / p[2]), float(p[1] / p[2])

    def pixels_to_metric(self, uv: np.ndarray) -> np.ndarray:
        self._require_initialized()
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        homog = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
        k0 = (self._H_live_to_K0 @ homog.T).T   # (N, 3)
        w = k0[:, 2:3]
        if np.any(np.abs(w) < 1e-12):
            raise RuntimeError("pixels_to_metric: degenerate homog coord in batch.")
        k0_2d = k0[:, :2] / w
        return self._base.pixels_to_metric(k0_2d)

    def metrics_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        self._require_initialized()
        k0_2d = self._base.metrics_to_pixel(xy)              # (N, 2)
        homog = np.concatenate([k0_2d, np.ones((k0_2d.shape[0], 1))], axis=1)
        live = (self._H_K0_to_live @ homog.T).T              # (N, 3)
        w = live[:, 2:3]
        if np.any(np.abs(w) < 1e-12):
            raise RuntimeError("metrics_to_pixel: degenerate homog coord in batch.")
        return live[:, :2] / w

    # ------------------------------------------------------------------
    # Introspection (used by tests / debugging)
    # ------------------------------------------------------------------

    @property
    def H_live_to_K0(self) -> Optional[np.ndarray]:
        return None if self._H_live_to_K0 is None else self._H_live_to_K0.copy()

    @property
    def last_status(self) -> str:
        return self._last_status

    @property
    def k0_shape_hw(self) -> Optional[Tuple[int, int]]:
        return self._K0_shape_hw

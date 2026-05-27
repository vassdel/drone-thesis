import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """Faithful port of Ntousis' `tr_pred_lstm` adapted to ContextVAE's data interface.

    Original (uav_guidance, commit ce91bae, scene_recog_socket_yolov8.py:30-39):

        class tr_pred_lstm(nn.Module):
            def __init__(self, out_steps):
                super().__init__()
                self.lstm   = nn.LSTM(input_size=10, hidden_size=64, batch_first=True)
                self.linear = nn.Linear(64, out_steps*2)
            def forward(self, x):
                x, _ = self.lstm(x); x = self.linear(x); return x

    Architectural choices preserved 1:1:
      - LSTM input_size=10, hidden_size=64, single layer, batch_first=True
      - Linear head: hidden_dim -> out_steps*2 (interleaved [x1,y1,...,xm,ym])
      - Linear is applied to ALL timesteps; the prediction is `out[..., -1, :]`
        (cf. scene_recog_socket_yolov8.py:588 `track_predict_model(...)[-1,:]`)

    Per-timestep 10-D input layout (cf. find_neighbors_from_detections at line 380):

        [up_x, up_y, left_x, left_y, target_x, target_y, right_x, right_y, down_x, down_y]

    Quadrants follow Ntousis' detect_up/down/left/right partition of the (rel_x, rel_y)
    plane by the y=x and y=-x diagonals (lines 40-90). In ContextVAE's agent-centered
    heading-rotated metric frame the visual labels "up"/"down" do not literally agree
    with Ntousis' pixel-space +y-down convention, but the 4-quadrant partition is the
    same — what matters for the LSTM is the consistent feature layout.

    Adaptations vs. the deployed Ntousis pipeline (all locked thesis decisions):
      - out_steps = PRED_HORIZON = 25 (5 s @ 5 Hz) instead of his 6, so eval matches
        ContextVAE's protocol head-to-head.
      - Inputs are raw metric coords (NO min-max scaling). See thesis_plan.md:114.
      - Neighbor source is the radius-masked, 1e9-padded `neighbor` tensor from
        contextvae/data.py — not the per-frame 4-direction pixel lookup. The
        quadrant assignment is computed at every observation timestep from those
        per-step neighbor positions, then fed to the LSTM as a 10-D stream.
      - Empty-quadrant fallback is (0, 0), matching Ntousis' `min_track=[0,0]`.
    """

    PAD_THRESHOLD = 1e8  # neighbor positions >= this come from the loader's 1e9 padding
    BIG = 1e18           # masked-distance sentinel for invalid candidates

    def __init__(self, horizon, hidden_dim=64, input_dim=10, **kwargs):
        super().__init__()
        del kwargs  # swallow ContextVAE-only config keys (ob_radius, map_model, ...)
        assert input_dim == 10, "Ntousis architecture is fixed at input_size=10"
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.use_map = False
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, horizon * 2)

    def _build_10feat_input(self, x, neighbor):
        # x        : [L1, N, 6]
        # neighbor : [L_total, N, Nn, 6] with 1e9 padding at the Nn axis
        # returns  : [N, L1, 10]
        L1, N, _ = x.shape
        nbh = neighbor[:L1]                          # [L1, N, Nn, 6]
        tx = x[..., 0]                               # [L1, N]
        ty = x[..., 1]                               # [L1, N]
        nx = nbh[..., 0]                             # [L1, N, Nn]
        ny = nbh[..., 1]                             # [L1, N, Nn]

        rel_x = nx - tx.unsqueeze(-1)                # [L1, N, Nn]
        rel_y = ny - ty.unsqueeze(-1)
        abs_rx = rel_x.abs()
        abs_ry = rel_y.abs()
        # Quadrant masks (Ntousis verbatim — see lines 40-90):
        up_mask    = (rel_y <= -abs_rx)
        down_mask  = (rel_y >=  abs_rx)
        left_mask  = (rel_x <  -abs_ry)
        right_mask = (rel_x >   abs_ry)

        # 1e9-padded slots: huge nx/ny — mask out so they aren't picked when a real
        # neighbor exists in the quadrant.
        padded = (nx.abs() >= self.PAD_THRESHOLD) | (ny.abs() >= self.PAD_THRESHOLD)
        dist = rel_x.square() + rel_y.square()       # [L1, N, Nn]

        def pick(qmask):
            valid = qmask & ~padded                  # [L1, N, Nn]
            d = torch.where(valid, dist, dist.new_full((), self.BIG))
            idx = d.argmin(dim=-1, keepdim=True)     # [L1, N, 1]
            qx = nx.gather(-1, idx).squeeze(-1)      # [L1, N]
            qy = ny.gather(-1, idx).squeeze(-1)
            none = ~valid.any(dim=-1)                # [L1, N]
            qx = torch.where(none, torch.zeros_like(qx), qx)
            qy = torch.where(none, torch.zeros_like(qy), qy)
            return qx, qy

        ux, uy = pick(up_mask)
        lx, ly = pick(left_mask)
        rx, ry = pick(right_mask)
        dx, dy = pick(down_mask)

        # Ntousis ordering: [up, left, target, right, down]
        feats = torch.stack(
            [ux, uy, lx, ly, tx, ty, rx, ry, dx, dy], dim=-1
        )                                            # [L1, N, 10]
        return feats.permute(1, 0, 2).contiguous()   # batch_first=True -> [N, L1, 10]

    def _predict(self, x, neighbor):
        # x:        [L1, N, 6]
        # neighbor: [L>=L1, N, Nn, 6]
        # returns:  [L2, N, 2]
        inp = self._build_10feat_input(x, neighbor)  # [N, L1, 10]
        h, _ = self.lstm(inp)                         # [N, L1, hidden_dim]
        out = self.linear(h)                          # [N, L1, horizon*2]
        out = out[:, -1, :]                           # last-timestep projection, [N, horizon*2]
        out = out.view(-1, self.horizon, 2)           # [N, L2, 2]
        return out.permute(1, 0, 2).contiguous()      # [L2, N, 2]

    def forward(self, *args, **kwargs):
        # Training:   forward(x, y, neighbor)                    -> (err, kl=0)
        # Eval det:   forward(x, neighbor, n_predictions=0)      -> [L2, N, 2]
        # Eval stoch: forward(x, neighbor, n_predictions=K, K>0) -> [K, L2, N, 2]
        if self.training:
            x, y, neighbor, *_ = args
            pred = self._predict(x, neighbor)
            err = (pred - y).square()
            kl = torch.zeros((), device=err.device, dtype=err.dtype)
            return err, kl

        ai = iter(args)
        x = kwargs["x"] if "x" in kwargs else next(ai)
        neighbor = kwargs.get("neighbor", None)
        if neighbor is None:
            try:
                neighbor = next(ai)
            except StopIteration:
                pass
        n_predictions = kwargs.get("n_predictions", 0)

        pred = self._predict(x, neighbor)             # [L2, N, 2]
        if n_predictions > 0:
            # Unimodal predictor: tile the deterministic output so min-over-K in
            # main.py is well-defined. minADE_K and minFDE_K collapse to ADE_d/FDE_d.
            pred = pred.unsqueeze(0).expand(n_predictions, -1, -1, -1)
        return pred

    def loss(self, err, kl):
        rec = err.mean()
        return {
            "loss": rec + kl,
            "rec": rec,
            "kl": kl,
        }

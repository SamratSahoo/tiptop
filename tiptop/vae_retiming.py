"""Stroke re-timing driven entirely by the VAE motion-manifold cost -- no flow model, no search.

WHY THIS EXISTS, AND WHY IT IS NOT THE `vae_retiming` TAMP OVERRIDE
------------------------------------------------------------------
Both put the VAE manifold cost in charge of the clock; they differ in what a "segment" is, and that
difference is the whole ballgame.

The in-trajopt `vae_retiming` override optimizes duration knots inside each cuRobo ``plan_single``.
But cuTAMP issues 2-3 ``plan_single`` calls per gripper-to-gripper motion (retract, approach, grasp
-- see cutamp/motion_solver.py), while DROID's latent cluster was built on whole human strokes. The
cost then pulls every LEG to a human STROKE duration. Measured on a 3-toy pick-and-place: 19 legs,
each emitted at 4.58-4.72 s (a 1.03x spread) while their joint path lengths spanned 12.4x -- a 5 cm
retract given the same 4.6 s as a 3.5 rad transit. Seven strokes took 13.9 s each against teleop's
6.8 s median, for an 88 s episode against teleop's 48.8 s.

This module scores the unit DROID actually has: one gripper-delimited stroke. The grouping,
vel/accel caps and endpoint pinning are the existing blender's (see trajectory_blending); only the
TIME LAW is ours.

WHAT IS OPTIMIZED, AND WHY THERE IS NO DURATION SEARCH
------------------------------------------------------
One batched Adam run over two things at once:

* ``theta`` -- one duration knot per arc-length interval, the SHAPE of the speed profile.
* ``nu``    -- a single scalar squashed into the allowed duration range: how many 15 Hz frames the
               stroke spans, and hence its duration.

Duration used to be a grid search (a coarse geomspace sweep plus a refine bracket, each candidate
getting its own independently-converged ``theta``) because the obvious parameterization makes it
non-differentiable. If the clock is normalized by duration -- knot shares scaled to sum to D -- then
every sampled position is scale-invariant in D, and D reaches the score ONLY through the integer
sample count ``round(D * 15) + 1``. The objective is then a step function of duration: literally zero
gradient inside a bin, so nothing to descend, and a finite-difference secant across bins is
noise-dominated (measured: the sign of the adjacent-bin difference flips on ~55% of steps).

Here the stroke is instead sampled on a FIXED 15 Hz grid whose frame COUNT is what duration
controls. Spreading the same path over more frames shrinks the per-frame step, so velocity,
acceleration and jerk -- and therefore the manifold score -- vary smoothly and analytically with
duration. The leftover discreteness (a stroke ends between frames) is absorbed by a soft mask, exact
thanks to ``_FilterbankVAE.encode_mu_masked``.

That same masking fix is what lets ``_N_STARTS`` initial durations share ONE padded forward pass, so
multi-start costs no more wall-clock than a single start. It also disposes of the old grid's most
delicate property: candidate durations no longer need hand-matched optimization budgets to be
comparable, because they now literally step together in one optimizer.

THREE THINGS THAT HAD TO BE RIGHT (each was measured, each alternative was worse)
--------------------------------------------------------------------------------
1. TRUE 15 Hz ENCODING. ``droid_mean``/``droid_prec`` were built from motion sampled at 15 Hz
   (vae/data.py COMMON_RATE), and the encoder is a filterbank whose kernels therefore mean a fixed
   number of SECONDS. Holding the frame count fixed and letting the spacing float instead (which
   would make duration differentiable the easy way, through the 1/h scaling of the derivative
   channels) feeds that filterbank the wrong rate: against a true 15 Hz encode it correlates
   anywhere from +0.02 to +0.99 depending on the stroke, moves the best duration by up to 3.9 s, and
   compresses the score range 10-100x. Do not "simplify" the clock that way.
2. ARC-LENGTH CANVAS, NOT INDEX. Resampling the joined stroke by index inherits cuRobo's own profile
   -- including the full stop at each interior leg join -- so theta can only rescale it. Arc length
   hands the speed profile to theta.
3. THETA, NOT JUST A DURATION. Arc length alone is worse than index: a constant-speed traversal is
   less human than cuRobo's bell. The cost wants an accelerate/cruise/decelerate shape, and theta is
   what lets it build one.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

_log = logging.getLogger(__name__)

#: Arc-length control knots the stroke's clock is parameterized over. One duration knot per interval,
#: so this also sets how finely the speed profile can be shaped. 64 is well above the ~10-25 samples a
#: 15 Hz encoding of a human stroke has, so the profile is never the limiting resolution.
_KNOTS = 64

#: Rail on each interval's duration: within exp(+-_RAIL) of the stroke's uniform-arc-length pace.
#: Wider than the in-trajopt cost's 0.7 because here it has to build the whole bell from a
#: constant-speed canvas, not nudge an existing profile -- exp(1.5) ~ 4.5x either way.
_RAIL = 1.5

#: One joint optimization over (theta, nu). Adam steps, and the two learning rates.
#:
#: The duration rate is NOT a free tuning knob to raise: duration is a single scalar competing with
#: 63 profile knots, so on its own a larger step just walks the duration somewhere bad before theta
#: has shaped anything. Measured on a 7-stroke plan, raising it alone monotonically hurt (total
#: maha^2 13.4 -> 17.7 -> 26.1 -> 205 for 0.05/0.15/0.30/0.60 at a single start). What actually fixes
#: the imbalance is starting from several durations at once -- see _N_STARTS.
_ITERS = 500
_LR_THETA = 0.08
_LR_DURATION = 0.15

#: Initial durations, log-spaced across the allowed range, optimized SIMULTANEOUSLY as one batch.
#: Strokes differ in where their basin sits -- most land near the middle of the range, but a Place
#: whose score explodes past ~6 s has its optimum near 3.5 s -- and a single start converges into
#: whichever basin it began in. Because the batch shares one forward pass this is nearly free: 16
#: starts cost the same wall-clock as 1, and took a 7-stroke plan from 13.4 to 5.9 total maha^2.
_N_STARTS = 16

#: Width (in frames) of the soft mask that ends the stroke between 15 Hz samples.
_MASK_TAU = 1.5

#: Weight on the velocity/acceleration hinge inside the optimization. This is a SOFT steer only; the
#: binding check is the hard cap test on the emitted samples, which rejects a start outright (_emit).
_GUARD_WEIGHT = 500.0
#: Weight pulling each stroke end onto the requested lead/trail boundary speed (a two-sided squared
#: target -- the end speed is held AT it, not merely above it).
#:
#: This term governs the arm's dwell at gripper events, which is the dominant stall in the emitted
#: data. Every cuTAMP gripper event is a near-exact direction reversal -- measured cos(v_before,
#: v_after) = -0.75 to -0.95, because the arm approaches along the tool axis and the Place retract
#: ladder lifts back out along it -- so |v| has to pass through a minimum there no matter how the
#: stroke is timed. Human teleop, which does NOT reverse (cos +0.85, only 6% of events), holds a
#: 0.168 rad/s median minimum through its own gripper events. What the timing controls is not
#: whether the dip happens but how long it lasts, and that is set by `blend_boundary_speed`.
#:
#: A/B'd on one raw 19-leg plan, counting emitted frames at 15 Hz:
#:    target @ 0.01   58 frames < 0.10 rad/s, longest near-zero run 17   junction acc 0.48x limit
#:    target @ 0.10    6 frames < 0.10 rad/s, longest run 5             junction acc 0.91x limit
#:    band   @ 0.10    7 frames < 0.10 rad/s, longest run 5             junction acc 1.16x limit
#: So the VALUE is the whole fix. Replacing the target with a floor (or a [floor, 3x floor] band) was
#: tried and rejected: it does not reduce the stall further, and the extra end speed it permits pushes
#: the reversal's acceleration -- which spans TWO strokes and is therefore invisible to _emit's
#: per-stroke cap check -- past the FR3's 15 rad/s^2 at two of six junctions. 0.10 already sits at
#: 0.91x, so it is close to the ceiling: raising `blend_boundary_speed` further trades a stall the
#: robot can track for a commanded reversal it cannot.
_BOUNDARY_WEIGHT = 50.0

#: Allowed duration range, as multiples of the stroke's own cuRobo wall-clock, and the absolute
#: seconds it is clipped to. These are BOX BOUNDS on the optimized duration (nu is squashed into
#: them), not a set of candidates. The lower multiple is what lets the VAE speed a stroke up; the
#: upper one is `blend_max_duration_mult`, passed in.
_D_LO_MULT = 0.2
_D_ABS_LO, _D_ABS_HI = 0.8, 20.0

_VAE_RATE_HZ = 15.0


class _Scorer:
    """Loads the VAE manifold pack once and scores 15 Hz-sampled strokes against the DROID cluster."""

    def __init__(self, checkpoint_path: str | None, n_joints: int):
        # Imported lazily: cuRobo is heavy and this module is only reached when blend_mode is "vae".
        from curobo.rollout.cost.vae_manifold_cost import load_vae_manifold
        from curobo.types.base import TensorDeviceType

        self.tp = TensorDeviceType()
        self.device = self.tp.device
        self.pack = dict(load_vae_manifold(checkpoint_path, self.tp), n_joints=n_joints)
        self.n_joints = n_joints

    def features(self, q: torch.Tensor):
        """q: [B, T, J] sampled at exactly 15 Hz -> ([B, C, T] standardized feats, vel, acc)."""
        from curobo.rollout.cost.vae_manifold_cost import _grad_time

        h = 1.0 / _VAE_RATE_HZ
        v = _grad_time(q, h)
        a = _grad_time(v, h)
        jk = _grad_time(a, h)
        feats = torch.cat([q, v, a, jk], dim=-1)
        feats = (feats - self.pack["chan_mu"]) / self.pack["chan_sd"]
        return feats.transpose(1, 2).contiguous(), v, a

    def maha2(self, x: torch.Tensor, m: torch.Tensor):
        """Squared Mahalanobis distance to the DROID cluster, on a masked (zero-padded) window."""
        dz = self.pack["model"].encode_mu_masked(x, m) - self.pack["droid_mean"]
        return torch.einsum("ni,ij,nj->n", dz, self.pack["droid_prec"], dz)

    def score_emitted(self, positions: np.ndarray, duration: float) -> float:
        """Score an EMITTED stroke: resample to exactly 15 Hz and encode it unpadded.

        Deliberately independent of the optimizer's own machinery -- this is what actually ships, so
        it is what starts are ranked by.
        """
        n = max(6, int(round(duration * _VAE_RATE_HZ)) + 1)
        src = np.linspace(0.0, 1.0, len(positions))
        tgt = np.linspace(0.0, 1.0, n)
        q = np.stack([np.interp(tgt, src, positions[:, j]) for j in range(positions.shape[1])], axis=1)
        qt = torch.as_tensor(q[None], device=self.device, dtype=torch.float32)
        x, _, _ = self.features(qt)
        with torch.no_grad():
            dz = self.pack["model"].encode_mu(x, torch.ones(1, 1, n, device=self.device))
            dz = dz - self.pack["droid_mean"]
            return float(torch.einsum("ni,ij,nj->n", dz, self.pack["droid_prec"], dz))


_SCORER: _Scorer | None = None


def _scorer(checkpoint_path: str | None, n_joints: int) -> _Scorer:
    """Process-wide scorer. Rebuilt if the checkpoint or dof changes (they don't, within a run)."""
    global _SCORER
    if _SCORER is None or _SCORER.n_joints != n_joints:
        _SCORER = _Scorer(checkpoint_path, n_joints)
    return _SCORER


def _sample_at_progress(q_knots: torch.Tensor, theta: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
    """Read the arc-length canvas ``q_knots`` [B, M, J] at normalized clock progress ``frac`` [B, T].

    ``theta`` [B, M-1] sets each arc interval's share of the stroke, so it owns the SHAPE of the
    speed profile. Separating progress from the frame grid lets the 15 Hz scoring pass and the
    executor-rate emit pass read the same clock.
    """
    d = torch.exp(_RAIL * torch.tanh(theta))
    d = d / d.sum(-1, keepdim=True)
    tau = F.pad(torch.cumsum(d, dim=-1), (1, 0))                                  # [B, M]
    idx = ((frac.unsqueeze(-1) >= tau[:, :-1].unsqueeze(1)).sum(-1) - 1).clamp(0, tau.shape[1] - 2)
    t0 = torch.gather(tau, 1, idx)
    t1 = torch.gather(tau, 1, idx + 1)
    u = ((frac - t0) / (t1 - t0).clamp(min=1e-9)).clamp(0.0, 1.0).unsqueeze(-1)
    gi = idx.unsqueeze(-1).expand(-1, -1, q_knots.shape[-1])
    q0 = torch.gather(q_knots, 1, gi)
    q1 = torch.gather(q_knots, 1, gi + 1)
    return q0 + (q1 - q0) * u


def _optimize(scorer, q_knots, dt, d_lo, d_hi, vel_cap, acc_cap, lead_speed, trail_speed):
    """Joint Adam over (theta, duration) from ``_N_STARTS`` durations at once. -> [(duration, theta)]."""
    dev = q_knots.device
    n_lo, n_hi = d_lo * _VAE_RATE_HZ + 1.0, d_hi * _VAE_RATE_HZ + 1.0
    n_frames = int(np.ceil(n_hi)) + 8

    starts = np.geomspace(d_lo, d_hi, _N_STARTS + 2)[1:-1]
    p = np.clip((starts * _VAE_RATE_HZ + 1.0 - n_lo) / (n_hi - n_lo), 1e-3, 1 - 1e-3)
    nu = torch.tensor(np.log(p / (1 - p)), device=dev, dtype=torch.float32, requires_grad=True)
    theta = torch.zeros(_N_STARTS, q_knots.shape[1] - 1, device=dev, requires_grad=True)
    knots = q_knots.expand(_N_STARTS, -1, -1)
    frames = torch.arange(n_frames, device=dev, dtype=q_knots.dtype)

    opt = torch.optim.Adam([{"params": [theta], "lr": _LR_THETA},
                            {"params": [nu], "lr": _LR_DURATION}])
    for _ in range(_ITERS):
        opt.zero_grad(set_to_none=True)
        n_eff = n_lo + (n_hi - n_lo) * torch.sigmoid(nu)                          # [B]
        duration = (n_eff - 1.0) / _VAE_RATE_HZ

        # 15 Hz scoring pass: the path is fully traversed by frame n_eff - 1, and the soft mask ends
        # the stroke between frames. Zeroing the features past the mask is what makes the padded
        # window score identically to the prefix alone (encode_mu_masked).
        q = _sample_at_progress(knots, theta, (frames[None] / (n_eff[:, None] - 1.0)).clamp(max=1.0))
        x, vel, acc = scorer.features(q)
        mask = torch.sigmoid((n_eff[:, None, None] - 1.0 - frames[None, None]) / _MASK_TAU)
        loss = scorer.maha2(x * mask, mask).sum()

        span = mask.transpose(1, 2)                                               # [B, T, 1]
        loss = loss + _GUARD_WEIGHT * (
            (F.relu(vel.abs() / vel_cap - 1.0).pow(2) * span).sum()
            + (F.relu(acc.abs() / acc_cap - 1.0).pow(2) * span).sum()
        )

        # Boundary speeds at the EXECUTOR timestep, read analytically off the same clock -- |q(dt) -
        # q(0)| / dt -- so no integer frame count enters and the term stays differentiable in
        # duration. This matches np.gradient's one-sided ends on the emitted samples.
        step = (dt / duration).clamp(max=0.5)
        zero, one = torch.zeros_like(step), torch.ones_like(step)
        ends = _sample_at_progress(knots, theta, torch.stack([zero, step, one - step, one], dim=1))
        bnd = q.new_zeros(())
        if lead_speed > 0.0:
            bnd = bnd + (((ends[:, 1] - ends[:, 0]).norm(dim=-1) / dt - lead_speed) ** 2).sum()
        if trail_speed > 0.0:
            bnd = bnd + (((ends[:, 3] - ends[:, 2]).norm(dim=-1) / dt - trail_speed) ** 2).sum()

        (loss + _BOUNDARY_WEIGHT * bnd).backward()
        opt.step()

    with torch.no_grad():
        n_eff = n_lo + (n_hi - n_lo) * torch.sigmoid(nu)
        durations = ((n_eff - 1.0) / _VAE_RATE_HZ).cpu().numpy()
    return [(float(durations[b]), theta[b : b + 1].detach()) for b in range(_N_STARTS)]


def _emit(q_knots, theta, duration, dt, vel_cap_np, acc_cap_np):
    """Resample one optimized stroke at the control timestep. Returns (pos, vel, acc) or None if the
    EMITTED signal violates the caps -- the hard check the soft guard only steers toward."""
    n_out = max(3, int(round(duration / dt)) + 1)
    dt_out = duration / (n_out - 1)
    frac = torch.linspace(0.0, 1.0, n_out, device=q_knots.device, dtype=q_knots.dtype)[None]
    with torch.no_grad():
        pos = _sample_at_progress(q_knots, theta, frac)[0].cpu().numpy().astype(np.float64)
    vel = np.gradient(pos, dt_out, axis=0)
    acc = np.gradient(vel, dt_out, axis=0)
    if (np.abs(vel) > vel_cap_np).any() or (np.abs(acc) > acc_cap_np).any():
        return None
    return pos, vel, acc


def vae_retime_group(
    positions: np.ndarray,
    dt: float,
    orig_duration: float,
    vel_cap: np.ndarray,
    acc_cap: np.ndarray,
    smoothing: float,
    lead_speed: float,
    trail_speed: float,
    max_duration_mult: float,
    checkpoint_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Re-time one joined stroke so its motion sits as close as possible to the DROID manifold.

    Same contract as :func:`trajectory_blending.blend_group` -- (pos, vel, acc, dt_out) -- so it drops
    into the same call site. ``orig_duration`` is the stroke's cuRobo wall-clock, used only to bound
    the duration range; unlike the spline/flow laws it is NOT a target, because the point here is to
    let the cost pick the pace.

    Geometry is the planner's, re-parameterized by arc length and smoothed with the shared
    ``blend_smoothing`` spline (``smoothing`` = 0 keeps the exact planner polyline, hence its exact
    collision status, at the price of the polyline's corner accelerations).
    """
    from tiptop.trajectory_blending import _dedup_path, _eval_geometry, _fit_geometry

    t_start = time.perf_counter()
    pos = _dedup_path(np.asarray(positions, dtype=np.float64))
    if len(pos) < 3:
        raise ValueError(f"VAE re-timing needs at least 3 distinct waypoints, got {len(pos)}")
    dof = pos.shape[1]

    # Arc-length canvas: the same curve, sampled uniformly in distance rather than in cuRobo's time.
    # Index sampling would carry cuRobo's stop at every interior leg join into the stroke, leaving
    # theta able only to rescale a profile that already comes to rest twice in the middle.
    chord = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(chord)])
    geom = _fit_geometry(u, pos, smoothing)
    knots_np = _eval_geometry(geom, np.linspace(0.0, u[-1], _KNOTS), 0)
    # The spline is fit through, not to, the endpoints; pin them so the stroke still starts and ends
    # exactly where cuTAMP's plan says (the next step's start position and any gripper action depend
    # on it).
    knots_np[0] = pos[0]
    knots_np[-1] = pos[-1]

    if dof != 7:
        # The VAE encodes a 28-D [q|v|a|j] metric for 7 joints; there is no meaningful score for a
        # 12-DOF bimanual chain, and silently scoring its first 7 columns would re-time BOTH arms
        # from one arm's motion. blend_cutamp_plan catches this per stroke and keeps the original
        # segments, so an unsupported embodiment degrades to "no re-timing" rather than to bad timing.
        raise ValueError(
            f"VAE stroke re-timing is 7-DOF only (the VAE was trained on 7-DOF Franka joint "
            f"metrics); this plan has dof={dof}"
        )
    scorer = _scorer(checkpoint_path, dof)
    dev = scorer.device
    q_knots = torch.as_tensor(knots_np[None], device=dev, dtype=torch.float32)
    v_cap = torch.as_tensor(np.abs(vel_cap), device=dev, dtype=torch.float32)
    a_cap = torch.as_tensor(np.abs(acc_cap), device=dev, dtype=torch.float32)

    d_lo = max(_D_ABS_LO, _D_LO_MULT * orig_duration)
    d_hi = min(_D_ABS_HI, max_duration_mult * orig_duration)
    if d_hi <= d_lo:
        d_hi = d_lo * 1.5

    best = None
    for duration, theta in _optimize(scorer, q_knots, dt, d_lo, d_hi, v_cap, a_cap,
                                     lead_speed, trail_speed):
        emitted = _emit(q_knots, theta, duration, dt, np.abs(vel_cap), np.abs(acc_cap))
        if emitted is None:
            continue
        m2 = scorer.score_emitted(emitted[0], duration)
        if best is None or m2 < best[0]:
            best = (m2, duration, emitted)

    if best is None:
        raise RuntimeError(
            f"VAE re-timing found no duration in [{d_lo:.2f}, {d_hi:.2f}]s that meets the velocity/"
            f"acceleration caps for this stroke"
        )
    m2, duration, (out_pos, out_vel, out_acc) = best
    _log.debug(
        "VAE stroke re-timing: %.2f s -> %.2f s (maha2 %.2f, %d waypoints, %.1f s to fit)",
        orig_duration, duration, m2, len(out_pos), time.perf_counter() - t_start,
    )
    from tiptop.trajectory_blending import _finish_stroke

    return _finish_stroke(out_pos, out_vel, out_acc, duration, lead_speed, trail_speed)

"""Stroke re-timing driven entirely by the VAE motion-manifold cost -- no flow model.

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
TIME LAW is ours. On the same plan it re-times 88.0 s -> 57.0 s at maha^2 0.12-5.39 per stroke,
against 2.8-9.4 for real teleop strokes measured the same way -- i.e. the emitted motion sits at
least as close to the DROID manifold as the human data does.

THREE THINGS THAT HAD TO BE RIGHT (each was measured, each alternative was worse)
--------------------------------------------------------------------------------
1. TRUE 15 Hz ENCODING. ``droid_mean``/``droid_prec`` were built from motion sampled at 15 Hz
   (vae/data.py COMMON_RATE). The in-trajopt cost cannot honour that -- it holds the sample COUNT
   fixed at ``round(nominal * 15) + 1`` so the integer count never enters its gradient, which means
   a 3 s segment is fed to the 15 Hz-trained filterbank at 23 Hz. Post-hoc we do not need that
   gradient: D is searched on an outer grid, so N = round(D * 15) + 1 is a CONSTANT for the inner
   problem and theta stays fully differentiable. Do not "fix" one encoding to match the other; they
   are deliberately different, for that reason.
2. ARC-LENGTH CANVAS, NOT INDEX. Resampling the joined stroke by index inherits cuRobo's own profile
   -- including the full stop at each interior leg join -- so theta can only rescale it (best
   maha^2 1.87-13.93). Arc length hands the speed profile to theta.
3. THETA, NOT JUST A GLOBAL SCALE. Arc length ALONE is worse than index (maha^2 7.0-34.2): a
   constant-speed traversal is less human than cuRobo's bell. The cost wants an accelerate/cruise/
   decelerate shape, and theta is what lets it build one.
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

#: Adam steps per candidate duration, IDENTICAL for every candidate. That uniformity is load-bearing,
#: not incidental: the outer search compares maha^2 across durations, so an unequal optimization
#: budget biases the comparison toward whichever candidates got more steps. Warm-starting each
#: duration from the previous one's profile was tried and does exactly that -- it cut the fit to 25 s
#: but the plan came out at 74.4 s instead of 49.9 s, because the under-converged candidates lost to
#: the well-converged ones on optimizer budget rather than on motion quality. Batching the durations
#: to share one budget was tried too: the encoder's masked pooling is NOT padding-invariant (12-6400%
#: score error against an unpadded encode), so a padded batch optimizes a different objective.
_ITERS = 200
_LR = 0.08

#: Weight on the velocity/acceleration/jerk hinge inside the optimization. This is a SOFT steer only;
#: the binding check is the hard cap test on the emitted samples, which rejects the candidate duration
#: outright (see _emit).
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

#: Duration search range, as multiples of the stroke's own cuRobo wall-clock, and the absolute
#: seconds it is clipped to. The lower multiple is what lets the VAE actually speed a stroke up;
#: the upper one is `blend_max_duration_mult`, passed in.
_D_LO_MULT = 0.2
_D_ABS_LO, _D_ABS_HI = 0.8, 20.0
#: Coarse sweep, then a refine pass bracketing the coarse winner. maha^2 is smooth and single-basin
#: in the duration direction on every stroke measured, so a bracket beats a finer uniform grid.
_D_GRID_COARSE = 7
_D_REFINE = 3

_VAE_RATE_HZ = 15.0


class _Scorer:
    """Loads the VAE manifold pack once and scores a 15 Hz-sampled stroke against the DROID cluster."""

    def __init__(self, checkpoint_path: str | None, n_joints: int):
        # Imported lazily: cuRobo is heavy and this module is only reached when blend_mode is "vae".
        from curobo.rollout.cost.vae_manifold_cost import load_vae_manifold
        from curobo.types.base import TensorDeviceType

        self.tp = TensorDeviceType()
        self.device = self.tp.device
        self.pack = dict(load_vae_manifold(checkpoint_path, self.tp), n_joints=n_joints)
        self.n_joints = n_joints

    def maha2(self, q: torch.Tensor):
        """q: [1, N, J] joint positions sampled at exactly 15 Hz -> (maha^2, (vel, acc, jerk))."""
        from curobo.rollout.cost.vae_manifold_cost import _grad_time

        h = 1.0 / _VAE_RATE_HZ
        v = _grad_time(q, h)
        a = _grad_time(v, h)
        jk = _grad_time(a, h)
        feats = torch.cat([q, v, a, jk], dim=-1)
        feats = (feats - self.pack["chan_mu"]) / self.pack["chan_sd"]
        x = feats.transpose(1, 2).contiguous()
        mask = torch.ones(1, 1, q.shape[1], device=q.device, dtype=q.dtype)
        dz = self.pack["model"].encode_mu(x, mask) - self.pack["droid_mean"]
        return torch.einsum("ni,ij,nj->n", dz, self.pack["droid_prec"], dz), (v, a, jk)


_SCORER: _Scorer | None = None


def _scorer(checkpoint_path: str | None, n_joints: int) -> _Scorer:
    """Process-wide scorer. Rebuilt if the checkpoint or dof changes (they don't, within a run)."""
    global _SCORER
    if _SCORER is None or _SCORER.n_joints != n_joints:
        _SCORER = _Scorer(checkpoint_path, n_joints)
    return _SCORER


def _sample_on_clock(q_knots: torch.Tensor, theta: torch.Tensor, duration: float, n_out: int) -> torch.Tensor:
    """Read the arc-length canvas ``q_knots`` [1, M, J] off the clock ``theta`` defines, at ``n_out``
    uniformly spaced times over [0, duration].

    ``theta`` sets each arc interval's share of the stroke's duration, normalized so the total is
    exactly ``duration`` -- so theta owns the SHAPE of the speed profile and the outer grid owns its
    SCALE. The two are separated on purpose: the scale changes the 15 Hz sample count (an integer) and
    the shape does not, so only the shape needs a gradient.
    """
    d = torch.exp(_RAIL * torch.tanh(theta))
    d = d / d.sum(-1, keepdim=True) * duration
    tau = F.pad(torch.cumsum(d, dim=-1), (1, 0))                                  # [1, M]
    t = torch.linspace(0.0, 1.0, n_out, device=q_knots.device, dtype=q_knots.dtype)[None] * tau[:, -1:]
    idx = ((t.unsqueeze(-1) >= tau[:, :-1].unsqueeze(1)).sum(-1) - 1).clamp(0, tau.shape[1] - 2)
    t0 = torch.gather(tau, 1, idx)
    t1 = torch.gather(tau, 1, idx + 1)
    u = ((t - t0) / (t1 - t0).clamp(min=1e-9)).clamp(0.0, 1.0).unsqueeze(-1)
    gi = idx.unsqueeze(-1).expand(-1, -1, q_knots.shape[-1])
    q0 = torch.gather(q_knots, 1, gi)
    q1 = torch.gather(q_knots, 1, gi + 1)
    return q0 + (q1 - q0) * u


def _boundary(speed: torch.Tensor, target: float) -> torch.Tensor:
    """Squared pull holding one stroke end at ``target``.

    ``target == 0`` means a genuine rest end (the episode's first and last stroke), where the arm
    really is stationary and any pull away from zero would be wrong -- so the term is skipped there
    and _finish_stroke pins it exactly.
    """
    if target <= 0.0:
        return speed.new_zeros(())
    return (speed - target) ** 2


def _optimize_theta(scorer, q_knots, duration, dt, vel_cap, acc_cap, lead_speed, trail_speed):
    """Fit the speed profile for one candidate duration. Returns (theta, maha2).

    Two resolutions, deliberately. maha^2 is scored on a TRUE 15 Hz sampling because that is what the
    DROID cluster was built from. The boundary band is evaluated on the EMITTED sampling (``dt``,
    ~0.02 s), because that is the velocity the plan carries and the executor and LeRobot export see.
    Scoring the band at 15 Hz instead measures a first difference over a 0.067 s window -- a
    different quantity on a profile that is ramping into a boundary, which left the emitted end
    speeds unconstrained in practice.
    """
    n_score = max(6, int(round(duration * _VAE_RATE_HZ)) + 1)
    n_emit = max(3, int(round(duration / dt)) + 1)
    theta = torch.zeros(1, q_knots.shape[1] - 1, device=q_knots.device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=_LR)
    for _ in range(_ITERS):
        opt.zero_grad(set_to_none=True)
        q = _sample_on_clock(q_knots, theta, duration, n_score)
        m2, (v, a, _) = scorer.maha2(q)
        over = (
            F.relu(v.abs() / vel_cap - 1.0).pow(2).sum()
            + F.relu(a.abs() / acc_cap - 1.0).pow(2).sum()
        )
        q_emit = _sample_on_clock(q_knots, theta, duration, n_emit)
        dt_out = duration / (n_emit - 1)
        v0 = (q_emit[0, 1] - q_emit[0, 0]).norm() / dt_out      # matches np.gradient's one-sided ends
        v1 = (q_emit[0, -1] - q_emit[0, -2]).norm() / dt_out
        bnd = _boundary(v0, lead_speed) + _boundary(v1, trail_speed)
        (m2.sum() + _GUARD_WEIGHT * over + _BOUNDARY_WEIGHT * bnd).backward()
        opt.step()
    with torch.no_grad():
        q = _sample_on_clock(q_knots, theta, duration, n_score)
        m2, _ = scorer.maha2(q)
    return theta.detach(), float(m2)


def _emit(q_knots, theta, duration, dt, vel_cap_np, acc_cap_np):
    """Resample the optimized stroke at the control timestep. Returns (pos, vel, acc) or None if the
    EMITTED signal violates the caps -- the hard check the soft guard only steers toward."""
    n_out = max(2, int(round(duration / dt)) + 1)
    with torch.no_grad():
        q = _sample_on_clock(q_knots, theta, duration, n_out)[0].cpu().numpy().astype(np.float64)
    dt_out = duration / (n_out - 1)
    vel = np.gradient(q, dt_out, axis=0)
    acc = np.gradient(vel, dt_out, axis=0)
    if (np.abs(vel) > vel_cap_np).any() or (np.abs(acc) > acc_cap_np).any():
        return None
    return q, vel, acc


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
    the duration search; unlike the spline/flow laws it is NOT a target, because the point here is to
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
    coarse = np.geomspace(d_lo, d_hi, _D_GRID_COARSE)

    best = None
    scored: dict[float, float] = {}

    def evaluate(duration: float):
        nonlocal best
        duration = float(duration)
        if duration in scored:
            return
        theta, m2 = _optimize_theta(
            scorer, q_knots, duration, dt, v_cap, a_cap, lead_speed, trail_speed
        )
        scored[duration] = m2
        emitted = _emit(q_knots, theta, duration, dt, np.abs(vel_cap), np.abs(acc_cap))
        if emitted is not None and (best is None or m2 < best[0]):
            best = (m2, duration, emitted)

    for duration in coarse:
        evaluate(duration)
    # Refine inside the bracket around the coarse winner (by score, not by feasibility -- a rejected
    # winner still tells us where the basin is, and a feasible neighbour may sit just beside it).
    if scored:
        centre = min(scored, key=scored.get)
        i = int(np.argmin(np.abs(coarse - centre)))
        lo, hi = coarse[max(0, i - 1)], coarse[min(len(coarse) - 1, i + 1)]
        for duration in np.geomspace(lo, hi, _D_REFINE + 2)[1:-1]:
            evaluate(duration)

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

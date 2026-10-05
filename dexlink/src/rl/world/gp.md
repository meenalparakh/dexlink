I think the GP idea is reasonable here, with one important adjustment: use it as a smooth residual model over the current reference, not as a free-form generator of the whole trajectory.

That gives you three useful properties at once:

smoothness over time
propagation of trustworthy late-phase improvements backward into earlier phases
hard or soft pinning at specific timesteps
For a 40-step horizon, the compute is small enough that a full GP posterior over time is practical.

Why it fits this problem

Your issue is a temporal stitching problem. A GP over phase can help because it turns a sparse set of “good suffix evidence” into a smooth target over all timesteps. If the last 10-15 steps are consistently improved, the posterior mean can pull nearby earlier steps toward a compatible continuation instead of leaving them aimed at the stale original reference.

I would not model the absolute trajectory directly. I would model:

input x: normalized timestep or phase in [0, 1]
target y: residual from the base reference
So for wrist position:

delta_p(t) = p_rollout(t) - p_base(t)
For fingers:

delta_q(t) = q_rollout(t) - q_base(t)
For wrist rotation, do not GP quaternions directly. Use a tangent-space residual:

delta_r(t) = log( R_base(t)^T R_rollout(t) ) as a 3D axis-angle residual
Then reconstruct:

p_opt(t) = p_base(t) + mean_delta_p(t)
R_opt(t) = R_base(t) exp(mean_delta_r(t))
q_opt(t) = q_base(t) + mean_delta_q(t)
This is much safer than fitting absolute states, and it naturally keeps you near the original reference unless the data strongly supports deviation.

What incorporation would look like

I would add a dedicated reference owner, something like ReferenceTrajectoryManager, instead of spreading this logic into reset/reward/commands.

It should own:

base_reference
optimized_reference_mean
optimized_reference_var
rollout samples used for updates
update logic from top-percentile trajectories
pinned constraints
Then all of these should query it as the single source of truth:

reset state construction in reset_manager.py (line 109)
command generation in commands_manager.py (line 166)
reward reference queries in reward_manager.py (line 501)
That part matters. If GP-smoothed motion is only used for resets, but rewards or commands still use the old retargeted motion, training will become inconsistent.

A good update loop

I would do this:

Collect rollouts for the current curriculum window.
Rank trajectories by suffix return.
Keep the top p%.
For each timestep, compute weighted residual observations from those rollouts.
Fit GP posterior for each channel over time.
Use posterior mean as the new optimized reference.
Use posterior variance as confidence.
Two practical choices matter here:

update only every K PPO iterations, not every batch
freeze the optimized reference between updates
Otherwise PPO sees a target that moves too often.

How to use the top percentile

You do not need every rollout point equally. Better:

only use successful or near-successful trajectories
weight each sample by normalized return
optionally weight later timesteps more during early curriculum stages
For timestep t, you can aggregate observations from selected rollouts as:

weighted mean residual
weighted empirical noise estimate
That empirical noise can become the GP observation noise, which is a clean way to downweight inconsistent rollout segments.

Pinning specific points

This is where the GP formulation is genuinely useful.

You can pin:

start configuration at t=0
known grasp/contact pose at t=t_grasp
terminal pose at t=T
In GP terms these are just observations with very low noise. Hard pinning is effectively zero-noise conditioning. Soft pinning is small-noise conditioning.

That gives you the behavior you want:

exact start state if needed
smooth interpolation between fixed anchors and learned improvements
What I would not do

I would not run independent unconstrained GPs over raw quaternion components. That will produce invalid rotations and ugly interpolation.

I also would not let the GP fully replace trust controls. Even with smoothing, you still want:

max residual magnitude limits
per-update step size limits
optional blend with previous optimized reference
So the actual deployed reference update should look more like:

ref_new = ref_prev + beta * (gp_mean - ref_prev), with small beta
That keeps the target from jumping.

Variance is useful, not just the mean

The posterior variance gives you a very clean curriculum signal:

low variance in late timesteps: safe to expand the reset window backward
high variance in a segment: do not trust that segment yet
So instead of expanding based only on average return, you can expand when both hold:

return over suffix is good
GP variance in that suffix is below threshold
That is stronger than using return alone.

Recommended kernel and representation

For your case, I would start simple:

kernel: RBF over normalized phase
optional small white-noise term
fit separate GPs for:
wrist position: 3 channels
wrist rotation residual: 3 channels
finger joints: one per joint
Given only 40 timesteps, separate per-channel GPs are fine. You do not need a fancy multi-output GP initially.

If you want even more structure later, use one shared kernel hyperparameter set per group:

one lengthscale for wrist translation
one for rotation residual
one for fingers
That is easier to tune and usually enough.

My overall recommendation

Yes, this is a coherent direction, but I would frame it as:

reverse curriculum for reset windows
top-percentile rollout selection
GP smoothing over residual trajectory
anchored constraints through pinning
posterior variance used as confidence for curriculum expansion
single reference provider used by reset, rewards, commands, and eval

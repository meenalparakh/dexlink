from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, sleep
from typing import TYPE_CHECKING

import numpy as np
import torch

from dexlink.src.rl.agent.ppo import PPO
from dexlink.utils.rl.action_probe_utils import finalize_action_tracking_episode
from dexlink.utils.rl.contact_probe_utils import (
    finalize_contact_tracking_episode,
    save_contact_axis_tracking_plot,
)

if TYPE_CHECKING:
    from dexlink.src.rl.core_agent.wandb_logger import WandbLogger
    from dexlink.src.rl.config.base import AgentConfig, EvalConfig, RealEvalConfig
    from dexlink.src.rl.world.env import Env


def _clone_probe_value_for_env(value, env_idx: int):
    if isinstance(value, torch.Tensor):
        return value[env_idx].detach().cpu().clone()
    if isinstance(value, dict):
        return {
            key: _clone_probe_value_for_env(nested_value, env_idx)
            for key, nested_value in value.items()
        }
    return value


def _split_action_traces_by_side(
    episode_action_traces: list[dict],
) -> dict[str, list[dict]]:
    if len(episode_action_traces) == 0:
        raise ValueError("Expected at least one action trace.")
    first_trace = episode_action_traces[0]
    if "joint_names" in first_trace:
        side = str(first_trace.get("side", "right"))
        return {side: episode_action_traces}
    side_names = tuple(first_trace)
    if len(side_names) == 0:
        raise ValueError("Expected action trace dictionary to contain hand probes.")
    traces_by_side: dict[str, list[dict]] = {}
    for side in side_names:
        side_traces = []
        for step_idx, trace in enumerate(episode_action_traces):
            side_trace = trace.get(side)
            if not isinstance(side_trace, dict):
                raise ValueError(
                    f"Expected action trace step {step_idx} to contain probe for hand {side!r}."
                )
            side_traces.append(side_trace)
        traces_by_side[side] = side_traces
    return traces_by_side


def _finalize_action_tracking_by_side(
    *,
    episode_idx: int,
    episode_dir: Path,
    episode_action_traces: list[dict],
) -> dict[str, dict[str, float | int | str]]:
    traces_by_side = _split_action_traces_by_side(episode_action_traces)
    use_side_subdirs = len(traces_by_side) > 1
    return {
        side: finalize_action_tracking_episode(
            episode_idx=episode_idx,
            episode_dir=episode_dir / side if use_side_subdirs else episode_dir,
            episode_action_traces=side_traces,
        )
        for side, side_traces in traces_by_side.items()
    }


def _format_tracking_summary_by_side(
    tracking_summaries: dict[str, dict[str, float | int | str]],
) -> str:
    return ", ".join(
        (
            f"{side}: mean_alpha_l2={summary['mean_alpha_l2']:.4f}, "
            f"median_alpha_l2={summary['median_alpha_l2']:.4f}, "
            f"mean_alpha_finger_l2={summary['mean_alpha_finger_l2']:.4f}, "
            f"mean_alpha_wrist_pos_l2={summary['mean_alpha_wrist_pos_l2']:.4f}, "
            f"mean_alpha_wrist_rot_l2={summary['mean_alpha_wrist_rot_l2']:.4f}"
        )
        for side, summary in tracking_summaries.items()
    )


class PPOEval(PPO):
    def __init__(
        self,
        env: Env,
        logger: WandbLogger,
        agent_cfg: AgentConfig,
        eval_cfg: EvalConfig | None = None,
    ) -> None:
        super().__init__(
            env=env,
            logger=logger,
            agent_cfg=agent_cfg,
            eval_cfg=eval_cfg,
        )

    def eval_policy_action(
        self,
        obs: torch.Tensor,
        *,
        history: tuple[torch.Tensor, torch.Tensor] | None,
        greedy_actions: bool,
    ) -> torch.Tensor:
        if greedy_actions:
            if history is None:
                action, _ = self.policy_net.act_greedy(obs)
            else:
                proc_obs = self.policy_obs_processor(obs, update_stats=False)
                history_obs, history_dones = history
                action, _ = self.policy_net.net.act_greedy(
                    proc_obs,
                    command_timesteps=self.policy_obs_processor.command_timesteps(obs),
                    history_obs=history_obs,
                    history_dones=history_dones,
                )
        else:
            if history is None:
                action, _ = self.policy_net.act(obs, update_stats=False)
            else:
                proc_obs = self.policy_obs_processor(obs, update_stats=False)
                history_obs, history_dones = history
                action, _ = self.policy_net.net.act(
                    proc_obs,
                    command_timesteps=self.policy_obs_processor.command_timesteps(obs),
                    history_obs=history_obs,
                    history_dones=history_dones,
                )
        return action

    def eval_policy_real(
        self,
        rl_run_dir,
        ckpt_timestep,
        seed,
        real_eval_cfg: RealEvalConfig,
    ):
        run_dir = Path(rl_run_dir)
        if (run_dir / "policy.pt").exists():
            ckpt_dir = run_dir
        else:
            ckpt_dir = run_dir / "models" / str(ckpt_timestep)

        policy_ckpt = ckpt_dir / "policy.pt"
        assert policy_ckpt.exists(), f"Policy checkpoint not found: {policy_ckpt}"

        self.load(policy_ckpt)
        self.policy_net.eval()
        self.policy_obs_processor.eval()
        self.value_obs_processor.eval()
        self.value_net.eval()
        if self.use_discriminator:
            self.discriminator.eval()

        torch.manual_seed(seed)
        np.random.seed(seed)

        eval_dir = Path(run_dir) / real_eval_cfg.logdir
        eval_dir.mkdir(parents=True, exist_ok=True)
        print(f"[real-eval] output directory: {eval_dir}")

        if self.env.num_envs != 1:
            raise ValueError(
                "eval_policy_real currently supports exactly one real environment for step dumps, "
                f"got {self.env.num_envs}."
            )

        def _clone_named_obs(named_obs):
            if named_obs is None:
                return None
            return {
                key: value.detach().cpu().clone()
                for key, value in named_obs.items()
                if isinstance(value, torch.Tensor)
            }

        previous_mode = getattr(self.env, "mode", "train")
        previous_command_timestep_substeps = int(self.env.command_timestep_substeps)
        self.env.set_mode("eval")
        command_substeps = int(real_eval_cfg.command_substeps)
        if command_substeps <= 0:
            raise ValueError(
                "Expected real_eval.command_substeps > 0, got "
                f"{real_eval_cfg.command_substeps}."
            )
        self.env.command_timestep_substeps = command_substeps

        session_metadata = {
            "rl_run_dir": str(run_dir),
            "checkpoint_dir": str(ckpt_dir),
            "policy_checkpoint": str(policy_ckpt),
            "checkpoint_timestep": int(ckpt_timestep),
            "seed": int(seed),
            "real_eval_cfg": asdict(real_eval_cfg),
        }

        if self.env.physics_manager.is_recording:
            pre_eval_video = eval_dir / "pre_real_eval_recording.mp4"
            self.env.physics_manager.stop_recording(
                save_to_filename=str(pre_eval_video), fps=30
            )

        if not real_eval_cfg.initialization_only:
            self._print_real_startup_command_timing(num_loops=10)
        if real_eval_cfg.startup_timing_only:
            session_metadata["rollout"] = {
                "rollout_seed": int(seed),
                "num_steps": 0,
                "ended_early": False,
                "checkpoint_timestep": int(ckpt_timestep),
                "policy_checkpoint": str(policy_ckpt),
                "video_path": None,
                "trajectory_path": None,
                "mode": "startup_timing_only",
                "eval_config": asdict(real_eval_cfg),
            }
            print("Completed real startup timing only.")
            session_path = eval_dir / "session_metadata.json"
            session_path.write_text(json.dumps(session_metadata, indent=2))
            self.env.command_timestep_substeps = previous_command_timestep_substeps
            self.env.set_mode(previous_mode)
            return session_metadata

        print(
            f"Starting real evaluation for checkpoint {ckpt_timestep} "
            f"for one rollout."
        )
        real_control_freq_hz = float(self.env.real_cfg.control_freq_hz)
        if real_control_freq_hz <= 0.0:
            raise ValueError(
                "Expected env.real_cfg.control_freq_hz > 0, got "
                f"{self.env.real_cfg.control_freq_hz}."
            )
        real_control_period_s = 1.0 / real_control_freq_hz

        with torch.no_grad():
            print("\rReal Eval Rollout 1/1", end="", flush=True)

            obs, _ = self.env.reset(seed=seed)
            obs = self._policy_obs(obs)
            if self.history_len > 0:
                self.reset_history_buffer()

            video_path = eval_dir / "rollout.mp4"
            traj_path = eval_dir / "rollout.pt"
            meta_path = eval_dir / "rollout.json"
            step_dump_dir = eval_dir / "rollout_step_dumps"
            step_dump_dir.mkdir(parents=True, exist_ok=True)
            print(f"[real-eval] step dump directory: {step_dump_dir}")
            recording_started = False
            step_dump_paths: list[str] = []

            if not real_eval_cfg.initialization_only:
                print(f"[real-eval] starting video recording -> {video_path}")
                self.env.physics_manager.start_recording()
                self.env.physics_manager.video_recording_hook(0)
                recording_started = True

            rollout_obs = [obs.detach().cpu().clone()]
            rollout_named_privileged_obs = [
                _clone_named_obs(self.env.current_named_privileged_obs)
            ]
            rollout_actions = []
            rollout_rewards = []
            rollout_terminated = []
            rollout_truncated = []
            rollout_dones = []
            rollout_infos = []

            steps_completed = 0
            ended_early = False
            rollout_limit = (
                command_substeps
                if real_eval_cfg.single_step_only
                else max(
                    real_eval_cfg.rollout_length * command_substeps,
                    real_eval_cfg.max_rollout_steps,
                )
            )

            if real_eval_cfg.initialization_only:
                rollout_metadata = {
                    "rollout_seed": int(seed),
                    "num_steps": 0,
                    "ended_early": False,
                    "checkpoint_timestep": int(ckpt_timestep),
                    "policy_checkpoint": str(policy_ckpt),
                    "video_path": None,
                    "trajectory_path": str(traj_path),
                    "mode": "initialization_only",
                    "eval_config": asdict(real_eval_cfg),
                }
                rollout_data = {
                    "metadata": rollout_metadata,
                    "observations": torch.stack(rollout_obs, dim=0),
                    "actions": torch.empty(0),
                    "rewards": torch.empty(0),
                    "terminated": torch.empty(0, dtype=torch.bool),
                    "truncated": torch.empty(0, dtype=torch.bool),
                    "dones": torch.empty(0, dtype=torch.bool),
                    "named_observations": rollout_named_privileged_obs,
                    "infos": rollout_infos,
                    "eval_config": asdict(real_eval_cfg),
                }
                torch.save(rollout_data, traj_path)
                meta_path.write_text(json.dumps(rollout_metadata, indent=2))
                session_metadata["rollout"] = rollout_metadata
                print("\nCompleted real evaluation initialization only.")
                session_path = eval_dir / "session_metadata.json"
                session_path.write_text(json.dumps(session_metadata, indent=2))
                self.env.command_timestep_substeps = previous_command_timestep_substeps
                self.env.set_mode(previous_mode)
                return session_metadata

            while steps_completed < rollout_limit:
                rollout_step_start = perf_counter()
                if real_eval_cfg.print_command_timing:
                    saved_pause_on_visualization_update = (
                        self.env.real_cfg.pause_on_visualization_update
                    )
                    self.env.real_cfg.pause_on_visualization_update = False
                    try:
                        obs_refresh_start = perf_counter()
                        current_obs = self.env.refresh_current_observations(
                            visualize_event=None
                        )
                        obs = self._policy_obs(current_obs)
                        obs_refresh_ms = (perf_counter() - obs_refresh_start) * 1000.0
                    finally:
                        self.env.real_cfg.pause_on_visualization_update = (
                            saved_pause_on_visualization_update
                        )

                history = None
                if self.history_len > 0:
                    history_obs, _, history_dones = (
                        self.history.get_previous_timesteps_data(
                            "obs", "actions", "dones", device=self.device
                        )
                    )
                    history = self._get_history_inputs(
                        history_obs=history_obs,
                        history_dones=history_dones,
                        obs_processor=self.policy_obs_processor,
                    )

                policy_forward_start = perf_counter()
                proc_obs = self.policy_obs_processor(obs, update_stats=False)
                action = self.eval_policy_action(
                    obs,
                    history=history,
                    greedy_actions=True,
                )
                policy_forward_ms = (perf_counter() - policy_forward_start) * 1000.0

                if real_eval_cfg.print_command_timing:
                    saved_pause_on_visualization_update = (
                        self.env.real_cfg.pause_on_visualization_update
                    )
                    self.env.real_cfg.pause_on_visualization_update = False
                    try:
                        action_post_start = perf_counter()
                        self.env._apply_actions_and_advance(action)
                        action_post_ms = (perf_counter() - action_post_start) * 1000.0
                    finally:
                        self.env.real_cfg.pause_on_visualization_update = (
                            saved_pause_on_visualization_update
                        )
                    total_command_ms = (
                        obs_refresh_ms + policy_forward_ms + action_post_ms
                    )
                    print(
                        "[real-command-timing] "
                        f"step={steps_completed} "
                        f"obs_refresh_ms={obs_refresh_ms:.1f} "
                        f"policy_forward_ms={policy_forward_ms:.1f} "
                        f"action_post_ms={action_post_ms:.1f} "
                        f"total_ms={total_command_ms:.1f} "
                        f"freq_hz={1000.0 / total_command_ms:.2f}"
                    )
                    next_obs, reward, terminated, truncated, info = (
                        self.env._finish_step_after_actions(action)
                    )
                else:
                    next_obs, reward, terminated, truncated, info = self.env.step(
                        action
                    )

                next_obs = self._policy_obs(next_obs)
                dones = terminated | truncated

                if self.env.current_action_probe is None:
                    raise ValueError(
                        "env.current_action_probe is not populated after env.step() during real eval."
                    )
                if self.env.current_named_policy_obs is None:
                    raise ValueError(
                        "env.current_named_policy_obs is not populated after env.step() during real eval."
                    )

                action_probe = _clone_probe_value_for_env(
                    self.env.current_action_probe, env_idx=0
                )
                named_policy_obs = {
                    key: value[0].detach().cpu().clone()
                    for key, value in self.env.current_named_policy_obs.items()
                    if isinstance(value, torch.Tensor)
                }

                contact_probe = None
                if self.env.current_contact_probe is not None:
                    contact_probe = {
                        key: (
                            value[0].detach().cpu().clone()
                            if isinstance(value, torch.Tensor)
                            else value
                        )
                        for key, value in self.env.current_contact_probe.items()
                    }

                step_dump_path = (
                    step_dump_dir / f"rollout_step_{steps_completed:04d}.pkl"
                )
                step_dump_payload = {
                    "step": int(steps_completed),
                    "policy_obs": obs[0].detach().cpu().clone(),
                    "processed_policy_obs": proc_obs[0].detach().cpu().clone(),
                    "named_policy_obs": named_policy_obs,
                    "action": action[0].detach().cpu().clone(),
                    "robot_joint_values": action_probe["achieved_qpos"],
                    "target_joint_values": action_probe["target_qpos"],
                    "joint_names": list(action_probe["joint_names"]),
                    "action_probe": action_probe,
                    "contact_probe": contact_probe,
                }
                with step_dump_path.open("wb") as handle:
                    pickle.dump(step_dump_payload, handle)
                step_dump_paths.append(str(step_dump_path))
                print(
                    f"[real-eval] dumped rollout step {steps_completed} -> {step_dump_path}"
                )

                rollout_actions.append(action.detach().cpu().clone())
                rollout_rewards.append(reward.detach().cpu().clone())
                rollout_terminated.append(terminated.detach().cpu().clone())
                rollout_truncated.append(truncated.detach().cpu().clone())
                rollout_dones.append(dones.detach().cpu().clone())
                rollout_infos.append(
                    {
                        key: value.detach().cpu().clone()
                        for key, value in info.items()
                        if isinstance(value, torch.Tensor)
                    }
                )
                rollout_obs.append(next_obs.detach().cpu().clone())
                rollout_named_privileged_obs.append(
                    _clone_named_obs(self.env.current_named_privileged_obs)
                )

                if self.history_len > 0:
                    self.history.add_samples(
                        obs=obs, actions=action, dones=dones.unsqueeze(-1)
                    )

                steps_completed += 1
                self.env.physics_manager.video_recording_hook(0)

                if bool(torch.all(dones).item()):
                    ended_early = True
                    break

                obs = next_obs
                remaining_period_s = real_control_period_s - (
                    perf_counter() - rollout_step_start
                )
                if remaining_period_s > 0.0:
                    sleep(remaining_period_s)

            if recording_started:
                print(f"[real-eval] stopping video recording -> {video_path}")
                self.env.physics_manager.stop_recording(
                    save_to_filename=str(video_path), fps=30
                )

            rollout_data = {
                "metadata": {
                    "rollout_seed": int(seed),
                    "num_steps": steps_completed,
                    "ended_early": ended_early,
                    "checkpoint_timestep": int(ckpt_timestep),
                    "policy_checkpoint": str(policy_ckpt),
                    "mode": (
                        "single_step_only"
                        if real_eval_cfg.single_step_only
                        else "full_rollout"
                    ),
                },
                "observations": torch.stack(rollout_obs, dim=0),
                "actions": (
                    torch.stack(rollout_actions, dim=0)
                    if rollout_actions
                    else torch.empty(0)
                ),
                "rewards": (
                    torch.stack(rollout_rewards, dim=0)
                    if rollout_rewards
                    else torch.empty(0)
                ),
                "terminated": (
                    torch.stack(rollout_terminated, dim=0)
                    if rollout_terminated
                    else torch.empty(0, dtype=torch.bool)
                ),
                "truncated": (
                    torch.stack(rollout_truncated, dim=0)
                    if rollout_truncated
                    else torch.empty(0, dtype=torch.bool)
                ),
                "dones": (
                    torch.stack(rollout_dones, dim=0)
                    if rollout_dones
                    else torch.empty(0, dtype=torch.bool)
                ),
                "named_observations": rollout_named_privileged_obs,
                "infos": rollout_infos,
                "eval_config": asdict(real_eval_cfg),
            }
            torch.save(rollout_data, traj_path)
            print(f"[real-eval] saved rollout tensor archive -> {traj_path}")

            step_files = sorted(step_dump_dir.glob("rollout_step_*.pkl"))
            if len(step_files) != steps_completed:
                raise ValueError(
                    "Mismatch between completed rollout steps and dumped step files: "
                    f"steps_completed={steps_completed}, dumped_files={len(step_files)}."
                )

            print(
                "[real-eval] loading step dump files for end-of-episode plots "
                f"from {step_dump_dir}"
            )
            loaded_action_traces: list[dict] = []
            loaded_contact_traces: list[dict] = []
            loaded_contact_xyz: list[torch.Tensor] = []
            loaded_finger_names: list[str] | None = None
            for expected_step, step_file in enumerate(step_files):
                with step_file.open("rb") as handle:
                    step_payload = pickle.load(handle)
                payload_step = int(step_payload["step"])
                if payload_step != expected_step:
                    raise ValueError(
                        "Unexpected step index in rollout step dump: "
                        f"expected {expected_step}, got {payload_step} in {step_file}."
                    )

                action_probe = step_payload.get("action_probe")
                if not isinstance(action_probe, dict):
                    raise ValueError(
                        f"Missing action_probe dictionary in step dump file {step_file}."
                    )
                loaded_action_traces.append(action_probe)

                contact_probe = step_payload.get("contact_probe")
                if contact_probe is None:
                    continue
                if not isinstance(contact_probe, dict):
                    raise ValueError(
                        f"Expected contact_probe dict in {step_file}, got {type(contact_probe).__name__}."
                    )
                if "contact" not in contact_probe:
                    raise ValueError(
                        f"Missing contact tensor in contact_probe for step dump {step_file}."
                    )
                if "finger_names" not in contact_probe:
                    raise ValueError(
                        f"Missing finger_names in contact_probe for step dump {step_file}."
                    )

                finger_names = list(contact_probe["finger_names"])
                contact_tensor = contact_probe["contact"]
                if not isinstance(contact_tensor, torch.Tensor):
                    raise ValueError(
                        "Expected contact_probe['contact'] to be a torch.Tensor in "
                        f"{step_file}, got {type(contact_tensor).__name__}."
                    )
                if contact_tensor.ndim != 1:
                    raise ValueError(
                        "Expected flattened contact tensor with shape (num_fingers*3,), "
                        f"got shape {tuple(contact_tensor.shape)} in {step_file}."
                    )
                expected_contact_dim = len(finger_names) * 3
                if contact_tensor.numel() != expected_contact_dim:
                    raise ValueError(
                        "Contact tensor size mismatch in step dump: expected "
                        f"{expected_contact_dim}, got {contact_tensor.numel()} in {step_file}."
                    )
                if loaded_finger_names is None:
                    loaded_finger_names = finger_names
                elif loaded_finger_names != finger_names:
                    raise ValueError(
                        "Inconsistent finger_names across rollout step dumps: "
                        f"{loaded_finger_names} vs {finger_names} in {step_file}."
                    )

                loaded_contact_traces.append(contact_probe)
                loaded_contact_xyz.append(contact_tensor.reshape(len(finger_names), 3))

            analysis_dir = eval_dir / "rollout_analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            tracking_summaries = _finalize_action_tracking_by_side(
                episode_idx=1,
                episode_dir=analysis_dir,
                episode_action_traces=loaded_action_traces,
            )
            tracking_summary = next(iter(tracking_summaries.values()))
            print(
                "[real-eval] saved target-vs-achieved joint plot -> "
                f"{tracking_summary['plot_path']}"
            )

            contact_summary = None
            contact_axis_plot_path = None
            if len(loaded_contact_traces) > 0:
                contact_summary = finalize_contact_tracking_episode(
                    episode_idx=1,
                    episode_dir=analysis_dir,
                    episode_contact_traces=loaded_contact_traces,
                )
                if loaded_finger_names is None:
                    raise ValueError(
                        "Contact traces were loaded but finger names are unavailable."
                    )
                if len(loaded_contact_xyz) != len(step_files):
                    raise ValueError(
                        "Expected contact probe data at every rollout step, but loaded "
                        f"{len(loaded_contact_xyz)} contact entries for {len(step_files)} steps."
                    )
                contact_axis_plot_path = (
                    analysis_dir / "contact_signals_per_finger_axis_xyz.png"
                )
                save_contact_axis_tracking_plot(
                    episode_idx=1,
                    finger_names=loaded_finger_names,
                    per_finger_contact_xyz=torch.stack(loaded_contact_xyz, dim=0),
                    plot_path=contact_axis_plot_path,
                )
                print(
                    "[real-eval] saved tactile xyz contact plot -> "
                    f"{contact_axis_plot_path}"
                )

            rollout_metadata = {
                "rollout_seed": int(seed),
                "num_steps": steps_completed,
                "ended_early": ended_early,
                "checkpoint_timestep": int(ckpt_timestep),
                "policy_checkpoint": str(policy_ckpt),
                "video_path": str(video_path) if recording_started else None,
                "trajectory_path": str(traj_path),
                "command_substeps": command_substeps,
                "step_dump_dir": str(step_dump_dir),
                "step_dump_files": step_dump_paths,
                "analysis_dir": str(analysis_dir),
                "joint_plot_path": str(tracking_summary["plot_path"]),
                "wrist_plot_path": str(tracking_summary["wrist_plot_path"]),
                "contact_plot_path": (
                    str(contact_summary["plot_path"])
                    if contact_summary is not None
                    else None
                ),
                "contact_axis_xyz_plot_path": (
                    str(contact_axis_plot_path)
                    if contact_axis_plot_path is not None
                    else None
                ),
                "mode": (
                    "single_step_only"
                    if real_eval_cfg.single_step_only
                    else "full_rollout"
                ),
                "eval_config": asdict(real_eval_cfg),
            }
            meta_path.write_text(json.dumps(rollout_metadata, indent=2))
            print(f"[real-eval] saved rollout metadata -> {meta_path}")
            session_metadata["rollout"] = rollout_metadata

        print("\nCompleted real evaluation.")
        session_path = eval_dir / "session_metadata.json"
        session_path.write_text(json.dumps(session_metadata, indent=2))
        self.env.command_timestep_substeps = previous_command_timestep_substeps
        self.env.set_mode(previous_mode)
        return session_metadata

    def _print_real_startup_command_timing(self, *, num_loops: int) -> None:
        if self.env.cfg.backend != "real":
            return
        if num_loops <= 0:
            raise ValueError(f"Expected num_loops > 0, got {num_loops}.")

        physics = self.env.physics_manager
        if self.history_len > 0:
            self.reset_history_buffer()

        saved_previous_position_targets_by_side = {
            side: value.detach().clone()
            for side, value in self.env.previous_position_targets_by_side.items()
        }
        saved_timesteps = self.env.timesteps.detach().clone()
        saved_episode_length = self.env.episode_length.detach().clone()
        saved_command_time_step = self.env.command_time_step.detach().clone()
        saved_start_timestep = self.env.start_timestep.detach().clone()
        saved_env_step_counter = int(self.env.env_step_counter)
        saved_current_obs = self.env.current_obs
        saved_current_policy_obs = self.env.current_policy_obs.detach().clone()
        saved_current_privileged_obs = self.env.current_privileged_obs.detach().clone()
        saved_current_wrist_pos_by_side = {
            side: value.detach().clone()
            for side, value in self.env.current_wrist_pos_by_side.items()
        }
        saved_current_wrist_quat_by_side = {
            side: value.detach().clone()
            for side, value in self.env.current_wrist_quat_by_side.items()
        }
        saved_current_wrist_control_jpos_by_side = {
            side: value.detach().clone()
            for side, value in self.env.current_wrist_control_jpos_by_side.items()
        }
        saved_current_named_policy_obs = self.env.current_named_policy_obs
        saved_current_named_privileged_obs = self.env.current_named_privileged_obs
        saved_current_action_probe = self.env.current_action_probe

        saved_pause_on_visualization_update = (
            self.env.real_cfg.pause_on_visualization_update
        )
        saved_visualize_policy_pcd = bool(self.env.visualize_policy_pcd)
        saved_bypass_low_level_control = bool(
            self.env.real_cfg.bypass_low_level_control
        )

        try:
            self.env.real_cfg.pause_on_visualization_update = False
            self.env.visualize_policy_pcd = False
            self.env.real_cfg.bypass_low_level_control = True
            physics._capture_tactile_reference()
            physics.step()
            self.env.previous_position_targets_by_side = {
                side: physics.get_hand_dof_positions(spec.hand, spec.hand.dof_map)
                .detach()
                .clone()
                for side, spec in self.env.action_manager.hand_specs.items()
            }
            self.env.timesteps.zero_()
            self.env.episode_length.zero_()
            self.env.start_timestep.zero_()
            self.env.update_command_time_step()
            self.env.env_step_counter = 0

            total_obs_refresh_ms = 0.0
            total_policy_forward_ms = 0.0
            total_action_post_ms = 0.0

            with torch.no_grad():
                for _ in range(num_loops):
                    obs_refresh_start = perf_counter()
                    current_obs = self.env.refresh_current_observations(
                        visualize_event=None
                    )
                    obs = self._policy_obs(current_obs)
                    obs_refresh_ms = (perf_counter() - obs_refresh_start) * 1000.0

                    history = None
                    if self.history_len > 0:
                        history_obs, _, history_dones = (
                            self.history.get_previous_timesteps_data(
                                "obs", "actions", "dones", device=self.device
                            )
                        )
                        history = self._get_history_inputs(
                            history_obs=history_obs,
                            history_dones=history_dones,
                            obs_processor=self.policy_obs_processor,
                        )

                    policy_forward_start = perf_counter()
                    if history is None:
                        action, _ = self.policy_net.act_greedy(obs)
                    else:
                        proc_obs = self.policy_obs_processor(obs, update_stats=False)
                        history_obs, history_dones = history
                        action, _ = self.policy_net.net.act_greedy(
                            proc_obs,
                            command_timesteps=(
                                self.policy_obs_processor.command_timesteps(obs)
                            ),
                            history_obs=history_obs,
                            history_dones=history_dones,
                        )
                    policy_forward_ms = (perf_counter() - policy_forward_start) * 1000.0

                    action_post_start = perf_counter()
                    self.env._apply_actions_and_advance(action)
                    action_post_ms = (perf_counter() - action_post_start) * 1000.0

                    total_obs_refresh_ms += obs_refresh_ms
                    total_policy_forward_ms += policy_forward_ms
                    total_action_post_ms += action_post_ms

                    if self.history_len > 0:
                        self.history.add_samples(
                            obs=obs,
                            actions=action,
                            dones=torch.zeros(
                                (self.env.num_envs, 1),
                                dtype=torch.bool,
                                device=self.device,
                            ),
                        )

            avg_obs_refresh_ms = total_obs_refresh_ms / num_loops
            avg_policy_forward_ms = total_policy_forward_ms / num_loops
            avg_action_post_ms = total_action_post_ms / num_loops
            avg_total_ms = (
                avg_obs_refresh_ms + avg_policy_forward_ms + avg_action_post_ms
            )
            print(
                "[real-startup-command-timing] "
                f"loops={num_loops} "
                f"avg_obs_refresh_ms={avg_obs_refresh_ms:.1f} "
                f"avg_policy_forward_ms={avg_policy_forward_ms:.1f} "
                f"avg_action_post_ms={avg_action_post_ms:.1f} "
                f"avg_total_ms={avg_total_ms:.1f} "
                f"avg_freq_hz={1000.0 / avg_total_ms:.2f}"
            )
        finally:
            self.env.real_cfg.pause_on_visualization_update = (
                saved_pause_on_visualization_update
            )
            self.env.visualize_policy_pcd = saved_visualize_policy_pcd
            self.env.real_cfg.bypass_low_level_control = saved_bypass_low_level_control
            self.env.previous_position_targets_by_side = (
                saved_previous_position_targets_by_side
            )
            self.env.timesteps.copy_(saved_timesteps)
            self.env.episode_length.copy_(saved_episode_length)
            self.env.command_time_step.copy_(saved_command_time_step)
            self.env.start_timestep.copy_(saved_start_timestep)
            self.env.env_step_counter = saved_env_step_counter
            self.env.current_obs = saved_current_obs
            self.env.current_policy_obs = saved_current_policy_obs
            self.env.current_privileged_obs = saved_current_privileged_obs
            self.env.current_wrist_pos_by_side = saved_current_wrist_pos_by_side
            self.env.current_wrist_quat_by_side = saved_current_wrist_quat_by_side
            self.env.current_wrist_control_jpos_by_side = (
                saved_current_wrist_control_jpos_by_side
            )
            self.env.current_named_policy_obs = saved_current_named_policy_obs
            self.env.current_named_privileged_obs = saved_current_named_privileged_obs
            self.env.current_action_probe = saved_current_action_probe

    def eval_policy_sim(
        self,
        policy_ckpt: str | Path,
        seed: int,
        compute_gt: bool = False,
        greedy_actions: bool = True,
    ) -> dict[str, int | str]:
        if self.env.num_envs != 1:
            raise ValueError(
                f"eval_policy_sim expects exactly one environment, got {self.env.num_envs}."
            )

        policy_ckpt = Path(policy_ckpt)
        if not policy_ckpt.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_ckpt}")
        if policy_ckpt.name != "policy.pt":
            raise ValueError(
                "eval_policy_sim expects a checkpoint file named 'policy.pt', "
                f"got {policy_ckpt}."
            )

        self.load(policy_ckpt)
        self.policy_net.eval()
        self.policy_obs_processor.eval()
        self.value_obs_processor.eval()
        self.value_net.eval()
        if self.use_discriminator:
            self.discriminator.eval()

        torch.manual_seed(seed)
        np.random.seed(seed)

        previous_mode = getattr(self.env, "mode", "train")
        self.env.set_mode("eval")
        obs, _ = self.env.reset(seed=seed, options={"gt_info": compute_gt})
        obs = self._policy_obs(obs)
        if self.history_len > 0:
            self.reset_history_buffer()

        print(
            f"Starting continuous sim evaluation from checkpoint {policy_ckpt}. "
            "Press Ctrl+C to stop."
        )

        episodes_completed = 0
        steps_completed = 0
        episode_return = 0.0
        episode_length = 0
        eval_dir = self.logger.experiment_dir / "sim_eval_action_tracking"
        eval_dir.mkdir(parents=True, exist_ok=True)
        episode_action_traces: list[dict] = []
        contact_probe_enabled = (
            "contact" in self.env.observation_manager.policy_obs_terms
        )
        episode_contact_traces: list[dict] = []

        try:
            with torch.no_grad():
                while True:
                    history = None
                    if self.history_len > 0:
                        history_obs, _, history_dones = (
                            self.history.get_previous_timesteps_data(
                                "obs", "actions", "dones", device=self.device
                            )
                        )
                        history = self._get_history_inputs(
                            history_obs=history_obs,
                            history_dones=history_dones,
                            obs_processor=self.policy_obs_processor,
                        )

                    action = self.eval_policy_action(
                        obs,
                        history=history,
                        greedy_actions=greedy_actions,
                    )

                    next_obs, reward, terminated, truncated, _ = self.env.step(
                        action, compute_gt=compute_gt
                    )
                    next_obs = self._policy_obs(next_obs)
                    done = terminated | truncated
                    reward_value = float(reward[0].item())
                    done_value = bool(done[0].item())

                    episode_return += reward_value
                    episode_length += 1
                    steps_completed += 1
                    if self.env.current_action_probe is None:
                        raise ValueError(
                            "env.current_action_probe is not populated after env.step(). "
                            "Action probe instrumentation is expected during sim evaluation."
                        )
                    episode_action_traces.append(
                        _clone_probe_value_for_env(
                            self.env.current_action_probe, env_idx=0
                        )
                    )
                    if contact_probe_enabled:
                        if self.env.current_contact_probe is None:
                            raise ValueError(
                                "env.current_contact_probe is not populated after env.step(). "
                                "Contact probe instrumentation is expected when contact observations are enabled."
                            )
                        episode_contact_traces.append(
                            {
                                key: (
                                    value[0].detach().cpu().clone()
                                    if isinstance(value, torch.Tensor)
                                    else value
                                )
                                for key, value in self.env.current_contact_probe.items()
                            }
                        )

                    if self.history_len > 0:
                        if done_value:
                            self.reset_history_buffer()
                        else:
                            self.history.add_samples(
                                obs=obs,
                                actions=action,
                                dones=done.unsqueeze(-1),
                            )

                    if done_value:
                        episodes_completed += 1
                        episode_dir = eval_dir / f"episode_{episodes_completed:04d}"
                        tracking_summaries = _finalize_action_tracking_by_side(
                            episode_idx=episodes_completed,
                            episode_dir=episode_dir,
                            episode_action_traces=episode_action_traces,
                        )
                        contact_summary = None
                        if contact_probe_enabled:
                            contact_summary = finalize_contact_tracking_episode(
                                episode_idx=episodes_completed,
                                episode_dir=episode_dir,
                                episode_contact_traces=episode_contact_traces,
                            )
                        if bool(terminated[0].item()):
                            done_reason = "terminated"
                        elif bool(truncated[0].item()):
                            done_reason = "truncated"
                        else:
                            done_reason = "done"
                        contact_summary_text = ""
                        if contact_summary is not None:
                            contact_summary_text = (
                                ", mean_contact_fraction="
                                f"{contact_summary['mean_contact_fraction']:.4f}"
                            )
                        print(
                            f"Episode {episodes_completed}: steps={episode_length}, "
                            f"return={episode_return:.4f}, reason={done_reason}, "
                            f"{_format_tracking_summary_by_side(tracking_summaries)}"
                            f"{contact_summary_text}"
                        )
                        episode_return = 0.0
                        episode_length = 0
                        episode_action_traces = []
                        episode_contact_traces = []

                    obs = next_obs
        finally:
            self.env.set_mode(previous_mode)

        return {
            "policy_checkpoint": str(policy_ckpt),
            "episodes_completed": episodes_completed,
            "steps_completed": steps_completed,
        }

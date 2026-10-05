import numpy as np
import torch

# from dexlink.src.rl.config.base import AgentConfig


class Memory:
    def __init__(self, n_rollout_steps, n_envs: int, seq_len: int = 1, name="memory"):
        self.rollouts = n_rollout_steps
        self.name = name
        self.num_envs = n_envs
        self.seq_len = seq_len
        self.memory_info = {}  # dict of tensor_name: shape
        self._tensors = {}  # dict of tensor_name: tensor, contain flattened values
        self.tensors = {}  # dict of tensor_name: tensor, contain original views
        self._sampling_view = {}
        self.current_memory_idx = {}  # pointer to current index for adding samples
        self.filled = False

        if self.seq_len > 1:
            assert (
                self.rollouts >= self.seq_len
            ), "Number of rollouts must be >= seq_len for sequence sampling"
            # create all sequence indices for sampling
            self.total_seqs = (self.rollouts - self.seq_len + 1) * self.num_envs

            # sampling view has (step1-env1, step1-env2, ..., step2-env1, step2-env2, ...)
            # we want to map the arange(total_seqs) to the corresponding traj indices in the above view
            self._seq_indices = np.zeros(
                (self.total_seqs, self.seq_len), dtype=np.int64
            )
            # for each env, slide through the rollout window

            print("Creating sequence indices for memory sampling... ", end="")
            for env_idx in range(self.num_envs):
                # the full seq in memory for env_idx is [env_idx, env_idx + num_envs, ..., env_idx + (rollouts - 1) * num_envs]
                for traj_idx in range(0, self.rollouts - self.seq_len + 1):
                    # env_idx + start_step * num_envs  is the starting index
                    # env_idx + (start_step + seq_len - 1) * num_envs is the ending index
                    seq = np.arange(
                        traj_idx, traj_idx + self.seq_len
                    )  # these are rollout indices
                    # need to map seq to the flattened view
                    mapped_seq = env_idx + seq * self.num_envs
                    self._seq_indices[
                        env_idx * (self.rollouts - self.seq_len + 1) + traj_idx, :
                    ] = mapped_seq

            print("Done.")

    def create_tensors(self, tensor_info_dict):
        # tensor_info_dict is {names: shape}
        self.memory_info.update(tensor_info_dict)
        self.current_memory_idx.update({name: 0 for name in tensor_info_dict.keys()})

        self.tensors = {}
        for name, shape in tensor_info_dict.items():
            # shape is without num_envs, and rollouts
            flattened_shape = np.prod(shape)
            full_shape = (self.rollouts, self.num_envs, flattened_shape)
            self._tensors[name] = torch.zeros(full_shape, device="cpu")
            # also store original views of the tensors
            original_shape = (self.rollouts, self.num_envs) + shape
            self.tensors[name] = self._tensors[name].view(original_shape)
            self._sampling_view[name] = self._tensors[name].view(
                self.rollouts * self.num_envs, flattened_shape
            )

    def add_samples(self, **kwargs):

        # kwargs are tensor_name=tensor of shape (num_envs, *shape)
        for name, value in kwargs.items():
            mem_idx = self.current_memory_idx[name]
            assert name in self.memory_info, f"Tensor name {name} not in memory info"
            assert (
                value.shape[0] == self.num_envs
            ), f"Value for {name} has incorrect batch size {value.shape}, expected {self.num_envs} at 0"
            assert (
                value.shape[1:] == self.memory_info[name]
            ), f"Value for {name} has incorrect shape {value.shape}, expected {self.memory_info[name]} at 1:"

            self._tensors[name][mem_idx].copy_(
                value.view(self.num_envs, -1).detach().cpu()
            )
            self.current_memory_idx[name] = (mem_idx + 1) % self.rollouts

        # print(f"current_memory_idx: {self.current_memory_idx}. Need to be {self.rollouts} to fill.")
        # Check if memory is filled
        if (not self.filled) and all(x == 0 for x in self.current_memory_idx.values()):
            self.filled = True
            print(f"{self.name} buffer filled.")

    def sample(self, batch_size, device="cpu"):

        all_mem_idx = list(self.current_memory_idx.values())
        # before sampling make sure all tensors have same current index
        assert all(
            x == all_mem_idx[0] for x in all_mem_idx
        ), f"Memory tensors have different current indices, {self.current_memory_idx}"

        # yield batches of samples of size batch_size, randomly shuffled
        total_samples = self.rollouts * self.num_envs
        indices = torch.randperm(total_samples)

        for start_idx in range(0, total_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            batch = {}
            for name in self.memory_info.keys():
                batch[name] = (
                    self._sampling_view[name][batch_indices]
                    .view(-1, *self.memory_info[name])
                    .detach()
                    .to(device)
                )
            yield batch

    def sample_indices(self, batch_size):
        # return a list of all sample indices each of shape (batch_size,)
        all_mem_idx = list(self.current_memory_idx.values())
        # before sampling make sure all tensors have same current index
        assert all(
            x == all_mem_idx[0] for x in all_mem_idx
        ), f"Memory tensors have different current indices, {self.current_memory_idx}"
        total_samples = self.rollouts * self.num_envs
        # indices = torch.randperm(total_samples)
        indices = np.random.permutation(total_samples)
        batch_indices = [
            indices[start_idx : start_idx + batch_size]
            for start_idx in range(0, total_samples, batch_size)
        ]
        return batch_indices  # this way we are not preparing all the batches at once

    def query_indices(self, indices: np.ndarray, device="cpu"):
        batch = {}
        for name in self.memory_info.keys():
            batch[name] = (
                self._sampling_view[name][indices]
                .view(-1, *self.memory_info[name])
                .detach()
                .to(device)
            )
        return batch

    def sample_sequence_indices(self, batch_size):
        all_mem_idx = list(self.current_memory_idx.values())
        assert all(
            x == all_mem_idx[0] for x in all_mem_idx
        ), f"Memory tensors have different current indices, {self.current_memory_idx}"
        assert (
            all_mem_idx[0] == 0
        ), f"Memory not full yet, current indices: {self.current_memory_idx}"

        # total sequences
        # env dim: num_envs
        # rollout_dim = n_rollouts - seq_len + 1

        # indices = torch.randperm(self.total_seqs)
        indices = np.random.permutation(self.total_seqs)
        batch_indices = [
            indices[start_idx : start_idx + batch_size]
            for start_idx in range(0, self.total_seqs, batch_size)
        ]
        seq_indices = [self._seq_indices[batch_idx, :] for batch_idx in batch_indices]
        return seq_indices  # list of (batch_size, seq_len)

    def query_sequence_indices(self, seq_indices, seq_first=False, device="cpu"):
        batch_size = seq_indices.shape[0]
        flattened_seq_indices = seq_indices.reshape(-1)  # (batch_size * seq_len
        batch = {}
        for name in self.memory_info.keys():
            tensor_view = self._sampling_view[
                name
            ]  # (rollouts * num_envs, flattened_shape)
            batch[name] = tensor_view[flattened_seq_indices].view(
                batch_size, self.seq_len, *self.memory_info[name]
            )
            if seq_first:
                batch[name] = torch.transpose(
                    batch[name], 0, 1
                )  # (seq_len, batch_size, *shape)
            batch[name] = batch[name].detach().to(device)
        return batch

    def sample_sequence(self, batch_size, seq_first=False, device="cpu"):
        # yields randomly shuffled batches of (batch_size, seq_len, *shape)
        all_mem_idx = list(self.current_memory_idx.values())
        # before sampling make sure all tensors have same current index
        assert all(
            x == all_mem_idx[0] for x in all_mem_idx
        ), f"Memory tensors have different current indices, {self.current_memory_idx}"
        # also make sure that memory is full (otherwise mismatch will occur)
        assert (
            all_mem_idx[0] == 0
        ), f"Memory not full yet, current indices: {self.current_memory_idx}"

        # total sequences
        # env dim: num_envs
        # rollout_dim = n_rollouts - seq_len + 1

        # indices = torch.randperm(self.total_seqs)
        indices = np.random.permutation(self.total_seqs)

        for start_idx in range(0, self.total_seqs, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            seq_indices = self._seq_indices[batch_indices, :]  # (batch_size, seq_len)

            # it is possible that last batch is smaller than batch_size
            # assert seq_indices.shape == (batch_size, self.seq_len), f"Seq indices shape mismatch: {seq_indices.shape}, expected {(batch_size, self.seq_len)}"
            actual_batch_size = seq_indices.shape[0]

            flattened_seq_indices = seq_indices.reshape(-1)  # (batch_size * seq_len,)

            batch = {}
            for name in self.memory_info.keys():
                tensor_view = self._sampling_view[
                    name
                ]  # (rollouts * num_envs, flattened_shape)
                batch[name] = tensor_view[flattened_seq_indices].view(
                    actual_batch_size, self.seq_len, *self.memory_info[name]
                )
                if seq_first:
                    batch[name] = torch.transpose(
                        batch[name], 0, 1
                    )  # (seq_len, batch_size, *shape)
                batch[name] = batch[name].detach().to(device)
            yield batch

    def get_previous_timesteps_data(self, *tensor_names: str, device="cpu"):

        # used to get all the past available data arranged in order of steps

        # return a copy of the past timesteps min(seq_len, current_idx)
        # return in order of tensor names requested

        prev_data = []
        for key in tensor_names:
            cur_idx = self.current_memory_idx[key]
            # cur_idx means 0, cur_idx-1 are filled, that is cur_idx values are available

            if cur_idx == 0:
                value = self.tensors[key].clone()
            else:
                part1 = self.tensors[key][:cur_idx].clone()
                part2 = self.tensors[key][cur_idx:].clone()
                value = torch.cat([part2, part1], dim=0)
            prev_data.append(value.detach().to(device))
        return tuple(prev_data)

    def reset(self):
        self.current_memory_idx = {name: 0 for name in self.memory_info.keys()}
        self.filled = False

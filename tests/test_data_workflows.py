"""Check reference formats and the episode boundaries consumed by data users."""
from pathlib import Path

import numpy as np
import pytest

from dexlink.workflows.data import index_rollout_chunks
from dexlink.visualizations.rl.rollout_io import load_reference_motion


@pytest.mark.parametrize('suffix', ['.npy','.npz'])
def test_render_reference_formats_preserve_scene_arrays(tmp_path: Path, suffix: str) -> None:
    source = {'scene_object_pos':np.arange(18,dtype=float).reshape(2,3,3)}
    path = tmp_path / ('motion'+suffix)
    if suffix=='.npy':
        np.save(path,source)
    else:
        np.savez(path,**source)
    np.testing.assert_array_equal(load_reference_motion(path)['scene_object_pos'],source['scene_object_pos'])


def test_render_rejects_a_non_dictionary_npy(tmp_path: Path) -> None:
    path = tmp_path / 'invalid.npy'
    np.save(path,np.ones((2,3)))
    with pytest.raises(ValueError,match='scalar-dictionary'):
        load_reference_motion(path)


def write_chunk(tmp_path: Path, *, complete: bool=True) -> None:
    folder = tmp_path/'rollouts'
    folder.mkdir()
    done = np.array([[False,False],[True,False],[False,False],[False,complete]])
    observation = np.zeros((4,2,3))
    observation[2:,0] = np.nan  # This padding follows the first episode's terminal transition.
    np.savez(folder/'chunk_0000.npz',done=done,policy_observation=observation,policy_action=np.zeros((4,2,2)))


def test_dataset_index_excludes_post_reset_padding(tmp_path: Path) -> None:
    write_chunk(tmp_path)
    index = index_rollout_chunks(tmp_path,2)
    assert index['valid_transitions']==6
    assert index['chunks'][0]['episode_lengths']==[2,4]


def test_dataset_index_rejects_incomplete_episodes(tmp_path: Path) -> None:
    write_chunk(tmp_path,complete=False)
    with pytest.raises(ValueError,match='terminal transition'):
        index_rollout_chunks(tmp_path,2)


def test_dataset_index_checks_episode_count(tmp_path: Path) -> None:
    write_chunk(tmp_path)
    with pytest.raises(ValueError,match='Expected 3 episodes'):
        index_rollout_chunks(tmp_path,3)

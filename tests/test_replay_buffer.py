import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from replay_buffer import ReplayBuffer


class TestReplayBuffer:

    def test_initialization(self):
        capacity = 100
        buffer = ReplayBuffer(cap=capacity)

        assert buffer.capacity == capacity
        assert buffer.head == 0
        assert buffer.size == 0
        assert buffer.state.shape == (capacity, ReplayBuffer.STATE_DIM)
        assert buffer.action.shape == (capacity,)
        assert buffer.reward.shape == (capacity,)
        assert buffer.next_state.shape == (capacity, ReplayBuffer.STATE_DIM)
        assert buffer.done.shape == (capacity,)

    def test_single_transition_store(self):
        buffer = ReplayBuffer(cap=10)
        state = np.ones(ReplayBuffer.STATE_DIM)
        action = 2
        reward = 1.0
        next_state = np.ones(ReplayBuffer.STATE_DIM) * 2
        done = False

        buffer.store_transition(state, action, reward, next_state, done)

        assert buffer.size == 1
        assert buffer.head == 1
        np.testing.assert_array_equal(buffer.state[0], state)
        assert buffer.action[0] == action
        assert buffer.reward[0] == reward
        np.testing.assert_array_equal(buffer.next_state[0], next_state)
        assert buffer.done[0] == done

    def test_filling_to_capacity(self):
        capacity = 5
        buffer = ReplayBuffer(cap=capacity)

        for i in range(capacity):
            buffer.store_transition(
                state=np.full(ReplayBuffer.STATE_DIM, i),
                action=i,
                reward=float(i),
                next_state=np.full(ReplayBuffer.STATE_DIM, i + 1),
                done=(i == capacity - 1),
            )

        assert buffer.size == capacity
        assert buffer.head == 0

    def test_circular_overwriting(self):
        capacity = 3
        buffer = ReplayBuffer(cap=capacity)

        for i in range(capacity):
            buffer.store_transition(
                np.full(ReplayBuffer.STATE_DIM, i), i, float(i), np.full(ReplayBuffer.STATE_DIM, i), False
            )

        assert buffer.size == capacity
        assert buffer.head == 0

        new_state = np.full(ReplayBuffer.STATE_DIM, 99)
        buffer.store_transition(new_state, 99, 99.0, new_state, True)

        assert buffer.size == capacity
        assert buffer.head == 1
        np.testing.assert_array_equal(buffer.state[0], new_state)
        assert buffer.action[0] == 99

    def test_sample_output_shapes(self):
        buffer = ReplayBuffer(cap=20)
        for i in range(10):
            buffer.store_transition(
                np.full(ReplayBuffer.STATE_DIM, i), i, float(i), np.full(ReplayBuffer.STATE_DIM, i + 1), False
            )

        batch_size = 4
        batch = buffer.sample(batch_size)

        assert isinstance(batch, tuple)
        assert len(batch) == 5

        states, actions, rewards, next_states, dones = batch

        assert states.shape == (batch_size, ReplayBuffer.STATE_DIM)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, ReplayBuffer.STATE_DIM)
        assert dones.shape == (batch_size,)

    def test_sample_contains_valid_stored_data(self):
        buffer = ReplayBuffer(cap=10)

        for i in range(5):
            buffer.store_transition(
                np.full(ReplayBuffer.STATE_DIM, i), i, float(i), np.full(ReplayBuffer.STATE_DIM, i + 10), False
            )

        states, actions, rewards, next_states, dones = buffer.sample(n=20)

        assert np.all((actions >= 0) & (actions < 5))
        assert np.all((rewards >= 0.0) & (rewards < 5.0))

        for idx in range(len(actions)):
            act = int(actions[idx])
            np.testing.assert_array_equal(states[idx], np.full(ReplayBuffer.STATE_DIM, act))
            np.testing.assert_array_equal(next_states[idx], np.full(ReplayBuffer.STATE_DIM, act + 10))
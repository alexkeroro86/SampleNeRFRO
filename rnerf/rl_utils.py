import numpy as np

from jax.lax import stop_gradient
import jax.numpy as jnp

from rnerf import math_utils


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, total_episode):
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.buffer_counter = 0
        self.batch_indices = None
        self.is_exceed_buffer_size = False
        self.episode = 0
        self.total_episode = total_episode
        
        self.ray_position_buffer = np.zeros((self.buffer_size, 3), dtype=np.float32)
        self.ray_distance_buffer = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.index_data_buffer = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.index_grad_buffer = np.zeros((self.buffer_size, 3), dtype=np.float32)
        # PER
        self.priority_buffer = np.zeros((self.buffer_size, 1), dtype=np.float32)

    # Take (ray_position, ray_distance, index_data, index_grad) tuple as input
    def add(self, experience, experience_size):
        for i in range(experience_size):
            if not self.is_exceed_buffer_size and self.buffer_counter == self.buffer_size:
                self.is_exceed_buffer_size = True
            self.buffer_counter = self.buffer_counter % self.buffer_size

            self.ray_position_buffer[self.buffer_counter] = experience[0][i]
            self.ray_distance_buffer[self.buffer_counter] = experience[1][i]
            self.index_data_buffer[self.buffer_counter] = experience[2][i]
            self.index_grad_buffer[self.buffer_counter] = experience[3][i]

            self.priority_buffer[self.buffer_counter] = np.abs(experience[4][i]) + 1e-4
            self.buffer_counter += 1
    
    # Return 
    def sample(self):
        proba = self.priority_buffer[:, 0]**0.6  # alpha=0.6
        proba = proba / np.sum(proba)

        if self.is_exceed_buffer_size:
            batch_indices = np.random.choice(self.buffer_size, self.batch_size, p=proba)
        else:
            batch_indices = np.random.choice(self.buffer_counter, self.batch_size, p=proba[:self.buffer_counter], replace=True)

        ray_position_batch = jnp.array(self.ray_position_buffer[batch_indices])
        ray_distance_batch = jnp.array(self.ray_distance_buffer[batch_indices])
        index_data_batch = jnp.array(self.index_data_buffer[batch_indices])
        index_grad_batch = jnp.array(self.index_grad_buffer[batch_indices])
        weight_batch = jnp.array(
            (1.0 / (self.buffer_size * self.priority_buffer[batch_indices]))**(0.4 + self.episode / self.total_episode * 0.6))  # beta=0.4->1

        weight_batch = weight_batch / weight_batch.max()
        self.batch_indices = batch_indices
        return (
            stop_gradient(ray_position_batch),
            stop_gradient(ray_distance_batch),
            stop_gradient(index_data_batch),
            stop_gradient(index_grad_batch),
            stop_gradient(weight_batch))

    # Prioritized Experience Replay
    def peek(self):
        ray_position_batch = jnp.array(self.ray_position_buffer[self.batch_indices])
        ray_distance_batch = jnp.array(self.ray_distance_buffer[self.batch_indices])
        index_data_batch = jnp.array(self.index_data_buffer[self.batch_indices])
        index_grad_batch = jnp.array(self.index_grad_buffer[self.batch_indices])

        return (
            stop_gradient(ray_position_batch),
            stop_gradient(ray_distance_batch),
            stop_gradient(index_data_batch),
            stop_gradient(index_grad_batch))

    def update(self, td_error):
        self.priority_buffer[self.batch_indices] = np.abs(td_error) + 1e-4

def square_to_hemisphere(r1, r2, exp=0.0):
  """
  Args:
    - exp: float, 0 for cosine, 1 for uniform distribution
  """
  cos_phi = jnp.cos(2.0 * jnp.pi * r1)
  sin_phi = jnp.sin(2.0 * jnp.pi * r1)
  cos_theta = (1.0 - r2)**(1.0 / (exp + 1.0))
  sin_theta = jnp.sqrt(1.0 - cos_theta * cos_theta)
  return jnp.concatenate([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], axis=-1)

def compute_action_space(square_size, shrink=0.0):
  X, Y = jnp.meshgrid(jnp.linspace(0, 1, square_size + 1),
                      jnp.linspace(0, 1 - shrink, square_size + 1))
  r = jnp.stack([X, Y], axis=-1)
  r = 0.5 * (r[1:, 1:] + r[:-1, :-1])
  r = r.reshape(-1, 2)
  actions = square_to_hemisphere(r[:, 0:1], r[:, 1:2], exp=1.0)
  return actions

def local_axis(from_here, to_there, dataset="blender", eps=1e-6):
  """
  Args:
    - from_here: jnp.ndarray, [square*square, 3]
    - to_there: jnp.ndarray, [batch, sample, 3]
    - dataset: string, up vector [0, 0, 1] for 'blender', [0, 1, 0] for 'opencv'
  Returns:
    - jnp.ndarray, [batch, sample, square*square, 3]
  """
  w = math_utils.safe_l2_normalize(to_there)[:, :, None]
  # Avoid parallel to up vector
  if dataset == "blender":
    up = jnp.array([0, eps, 1])[None]
  elif dataset == "opencv":
    up = jnp.array([0, 1, eps])[None]
  v = math_utils.safe_l2_normalize(jnp.cross(w, up))
  u = math_utils.safe_l2_normalize(jnp.cross(w, v))
  return stop_gradient(from_here[None, None, :, 0:1] * u + from_here[None, None, :, 1:2] * v + from_here[None, None, :, 2:3] * w)

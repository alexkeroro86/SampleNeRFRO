import numpy as np
import jax
import jax.numpy as jnp


def safe_l2_normalize(x, eps=1e-6):
  return x / safe_l2_norm(x, eps)


def safe_l2_norm(x, eps=1e-6):
  """jnp.norm output nan gradient if 0"""
  return jnp.sqrt(jnp.maximum(jnp.sum(x**2, axis=-1, keepdims=True), eps))


def safe_divide(a, b, eps=1e-6):
  return a / (b + eps)


def safe_log(x, eps=1e-6):
  return jnp.log(jnp.maximum(x, eps))


def matmul(a, b):
  """jnp.matmul defaults to bfloat16, but this helper function doesn't."""
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def safe_trig_helper(x, fn, t=100 * jnp.pi):
  return fn(jnp.where(jnp.abs(x) < t, x, x % t))


def safe_cos(x):
  """jnp.cos() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.cos)


def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.sin)


trans_t = lambda t : np.array([
  [1,0,0,0],
  [0,1,0,0],
  [0,0,1,t],
  [0,0,0,1]], dtype=np.float32)

rot_phi = lambda phi : np.array([
  [1,0,0,0],
  [0,np.cos(phi),-np.sin(phi),0],
  [0,np.sin(phi), np.cos(phi),0],
  [0,0,0,1]], dtype=np.float32)

rot_theta = lambda th : np.array([
  [np.cos(th),0,-np.sin(th),0],
  [0,1,0,0],
  [np.sin(th),0, np.cos(th),0],
  [0,0,0,1]], dtype=np.float32)


def pose_spherical(theta, phi, radius):
  c2w = trans_t(radius)
  c2w = rot_phi(phi/180.*np.pi) @ c2w
  c2w = rot_theta(theta/180.*np.pi) @ c2w
  c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.float32) @ c2w
  return c2w

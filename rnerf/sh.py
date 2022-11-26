#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import math
import jax.numpy as jnp
import jax.scipy as jsp

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,]

def eval_sh(deg, sh, dirs):
  """
  Evaluate spherical harmonics at unit directions
  using hardcoded SH polynomials.
  Works with torch/np/jnp.
  ... Can be 0 or more batch dimensions.

  Args:
      deg: int SH deg. Currently, 0-3 supported
      sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
      dirs: jnp.ndarray unit directions [..., 3]

  Returns:
      [..., C]
  """
  assert deg <= 4 and deg >= 0
  assert (deg + 1) ** 2 == sh.shape[-1]
  C = sh.shape[-2]

  result = C0 * sh[..., 0]
  if deg > 0:
    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    result = (result -
              C1 * y * sh[..., 1] +
              C1 * z * sh[..., 2] -
              C1 * x * sh[..., 3])
    if deg > 1:
      xx, yy, zz = x * x, y * y, z * z
      xy, yz, xz = x * y, y * z, x * z
      result = (result +
                C2[0] * xy * sh[..., 4] +
                C2[1] * yz * sh[..., 5] +
                C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                C2[3] * xz * sh[..., 7] +
                C2[4] * (xx - yy) * sh[..., 8])

      if deg > 2:
        result = (result +
                  C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                  C3[1] * xy * z * sh[..., 10] +
                  C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                  C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                  C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                  C3[5] * z * (xx - yy) * sh[..., 14] +
                  C3[6] * x * (xx - 3 * yy) * sh[..., 15])
        if deg > 3:
          result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                    C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                    C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                    C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                    C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                    C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                    C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                    C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                    C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
  return result

def dir_enc(data_in, sh_degree):
  """https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/encodings/spherical_harmonics.h
  
  Args:
      - data_in: jnp.ndarray, viewdir, [B, 3]
      - sh_degree: int, level of sh basis, (1, 8)
  
  Returns:
      - out: jnp.ndarray, sh projection of viewdir, [B, 1 + 3 + ...]
  """
  x = data_in[..., 0]
  y = data_in[..., 1]
  z = data_in[..., 2]

  xy, xz, yz, x2, y2, z2 = x * y, x * z, y * z, x * x, y * y, z * z
  xyz = xy * z
  x4, y4, z4, = x2 * x2, y2 * y2, z2 * z2
  x6, y6, z6 = x4 * x2, y4 * y2, z4 * z2

  # SH polynomials
  out = []
  out.append(0.28209479177387814 * jnp.ones_like(x))
  if sh_degree <= 1: return jnp.stack(out, axis=-1)
  out.append(-0.48860251190291987 * y)
  out.append(0.48860251190291987 * z)
  out.append(-0.48860251190291987 * x)
  if sh_degree <= 2: return jnp.stack(out, axis=-1)
  out.append(1.0925484305920792 * xy)
  out.append(-1.0925484305920792 * yz)
  out.append(0.94617469575755997 * z2 - 0.31539156525251999)
  out.append(-1.0925484305920792 * xz)
  out.append(0.54627421529603959 * x2 - 0.54627421529603959 * y2)
  if sh_degree <= 3: return jnp.stack(out, axis=-1)
  out.append(0.59004358992664352 * y * (-3.0 * x2 + y2))
  out.append(2.8906114426405538 * xyz)
  out.append(0.45704579946446572 * y * (1.0 - 5.0 * z2))
  out.append(0.3731763325901154 * z * (5.0 * z2 - 3.0))
  out.append(0.45704579946446572 * x * (1.0 - 5.0 * z2))
  out.append(1.4453057213202769 * z * (x2 - y2))
  out.append(0.59004358992664352 * x * (-x2 + 3.0 * y2))
  if sh_degree <= 4: return jnp.stack(out, axis=-1)
  out.append(2.5033429417967046 * xy * (x2 - y2))
  out.append(1.7701307697799304 * yz * (-3.0 * x2 + y2))
  out.append(0.94617469575756008 * xy * (7.0 * z2 - 1.0))
  out.append(0.66904654355728921 * yz * (3.0 - 7.0 * z2))
  out.append(-3.1735664074561294 * z2 + 3.7024941420321507 * z4 + 0.31735664074561293)
  out.append(0.66904654355728921 * xz * (3.0 - 7.0 * z2))
  out.append(0.47308734787878004 * (x2 - y2) * (7.0 * z2 - 1.0))
  out.append(1.7701307697799304 * xz * (-x2 + 3.0 * y2))
  out.append(-3.7550144126950569 * x2 * y2 + 0.62583573544917614 * x4 + 0.62583573544917614 * y4)
  if sh_degree <= 5: return jnp.stack(out, axis=-1)
  out.append(0.65638205684017015 * y * (10.0 * x2 * y2 - 5.0 * x4 - y4))
  out.append(8.3026492595241645 * xyz * (x2 - y2))
  out.append(-0.48923829943525038 * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0))
  out.append(4.7935367849733241 * xyz * (3.0 * z2 - 1.0))
  out.append(0.45294665119569694 * y * (14.0 * z2 - 21.0 * z4 - 1.0))
  out.append(0.1169503224534236 * z * (-70.0 * z2 + 63.0 * z4 + 15.0))
  out.append(0.45294665119569694 * x * (14.0 * z2 - 21.0 * z4 - 1.0))
  out.append(2.3967683924866621 * z * (x2 - y2) * (3.0 * z2 - 1.0))
  out.append(-0.48923829943525038 * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0))
  out.append(2.0756623148810411 * z * (-6.0 * x2 * y2 + x4 + y4))
  out.append(0.65638205684017015 * x * (10.0 * x2 * y2 - x4 - 5.0 * y4))
  if sh_degree <= 6: return jnp.stack(out, axis=-1)
  out.append(1.3663682103838286 * xy * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4))
  out.append(2.3666191622317521 * yz * (10.0 * x2 * y2 - 5.0 * x4 - y4))
  out.append(2.0182596029148963 * xy * (x2 - y2) * (11.0 * z2 - 1.0))
  out.append(-0.92120525951492349 * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0))
  out.append(0.92120525951492349 * xy * (-18.0 * z2 + 33.0 * z4 + 1.0))
  out.append(0.58262136251873131 * yz * (30.0 * z2 - 33.0 * z4 - 5.0))
  out.append(6.6747662381009842 * z2 - 20.024298714302954 * z4 + 14.684485723822165 * z6 - 0.31784601133814211)
  out.append(0.58262136251873131 * xz * (30.0 * z2 - 33.0 * z4 - 5.0))
  out.append(0.46060262975746175 * (x2 - y2) * (11.0 * z2 * (3.0 * z2 - 1.0) - 7.0 * z2 + 1.0))
  out.append(-0.92120525951492349 * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0))
  out.append(0.50456490072872406 * (11.0 * z2 - 1.0) * (-6.0 * x2 * y2 + x4 + y4))
  out.append(2.3666191622317521 * xz * (10.0 * x2 * y2 - x4 - 5.0 * y4))
  out.append(10.247761577878714 * x2 * y4 - 10.247761577878714 * x4 * y2 + 0.6831841051919143 * x6 - 0.6831841051919143 * y6)
  if sh_degree <= 7: return jnp.stack(out, axis=-1)
  out.append(0.70716273252459627 * y * (-21.0 * x2 * y4 + 35.0 * x4 * y2 - 7.0 * x6 + y6))
  out.append(5.2919213236038001 * xyz * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4))
  out.append(-0.51891557872026028 * y * (13.0 * z2 - 1.0) * (-10.0 * x2 * y2 + 5.0 * x4 + y4))
  out.append(4.1513246297620823 * xyz * (x2 - y2) * (13.0 * z2 - 3.0))
  out.append(-0.15645893386229404 * y * (3.0 * x2 - y2) * (13.0 * z2 * (11.0 * z2 - 3.0) - 27.0 * z2 + 3.0))
  out.append(0.44253269244498261 * xyz * (-110.0 * z2 + 143.0 * z4 + 15.0))
  out.append(0.090331607582517306 * y * (-135.0 * z2 + 495.0 * z4 - 429.0 * z6 + 5.0))
  out.append(0.068284276912004949 * z * (315.0 * z2 - 693.0 * z4 + 429.0 * z6 - 35.0))
  out.append(0.090331607582517306 * x * (-135.0 * z2 + 495.0 * z4 - 429.0 * z6 + 5.0))
  out.append(0.07375544874083044 * z * (x2 - y2) * (143.0 * z2 * (3.0 * z2 - 1.0) - 187.0 * z2 + 45.0))
  out.append(-0.15645893386229404 * x * (x2 - 3.0 * y2) * (13.0 * z2 * (11.0 * z2 - 3.0) - 27.0 * z2 + 3.0))
  out.append(1.0378311574405206 * z * (13.0 * z2 - 3.0) * (-6.0 * x2 * y2 + x4 + y4))
  out.append(-0.51891557872026028 * x * (13.0 * z2 - 1.0) * (-10.0 * x2 * y2 + x4 + 5.0 * y4))
  out.append(2.6459606618019 * z * (15.0 * x2 * y4 - 15.0 * x4 * y2 + x6 - y6))
  out.append(0.70716273252459627 * x * (-35.0 * x2 * y4 + 21.0 * x4 * y2 - x6 + 7.0 * y6))
  return jnp.stack(out, axis=-1)

def cosine_easing_factor(band, alpha):
  x = jnp.clip(alpha - band, 0.0, 1.0)
  return 0.5 * (1 + jnp.cos(jnp.pi * x + jnp.pi))

def annealed_dir_enc(data_in, sh_degree, alpha):
  x = data_in[..., 0]
  y = data_in[..., 1]
  z = data_in[..., 2]

  xy, xz, yz, x2, y2, z2 = x * y, x * z, y * z, x * x, y * y, z * z
  xyz = xy * z
  x4, y4, z4, = x2 * x2, y2 * y2, z2 * z2
  x6, y6, z6 = x4 * x2, y4 * y2, z4 * z2

  # SH polynomials
  out = []
  factor = cosine_easing_factor(0, alpha)
  out.append(factor * 0.28209479177387814 * jnp.ones_like(x))
  if sh_degree <= 1: return jnp.stack(out, axis=-1)
  factor = cosine_easing_factor(1, alpha)
  out.append(factor * -0.48860251190291987 * y)
  out.append(factor * 0.48860251190291987 * z)
  out.append(factor * -0.48860251190291987 * x)
  if sh_degree <= 2: return jnp.stack(out, axis=-1)
  factor = cosine_easing_factor(2, alpha)
  out.append(factor * 1.0925484305920792 * xy)
  out.append(factor * -1.0925484305920792 * yz)
  out.append(factor * 0.94617469575755997 * z2 - 0.31539156525251999)
  out.append(factor * -1.0925484305920792 * xz)
  out.append(factor * 0.54627421529603959 * x2 - 0.54627421529603959 * y2)
  if sh_degree <= 3: return jnp.stack(out, axis=-1)
  factor = cosine_easing_factor(3, alpha)
  out.append(factor * 0.59004358992664352 * y * (-3.0 * x2 + y2))
  out.append(factor * 2.8906114426405538 * xyz)
  out.append(factor * 0.45704579946446572 * y * (1.0 - 5.0 * z2))
  out.append(factor * 0.3731763325901154 * z * (5.0 * z2 - 3.0))
  out.append(factor * 0.45704579946446572 * x * (1.0 - 5.0 * z2))
  out.append(factor * 1.4453057213202769 * z * (x2 - y2))
  out.append(factor * 0.59004358992664352 * x * (-x2 + 3.0 * y2))
  if sh_degree <= 4: return jnp.stack(out, axis=-1)
  factor = cosine_easing_factor(4, alpha)
  out.append(factor * 2.5033429417967046 * xy * (x2 - y2))
  out.append(factor * 1.7701307697799304 * yz * (-3.0 * x2 + y2))
  out.append(factor * 0.94617469575756008 * xy * (7.0 * z2 - 1.0))
  out.append(factor * 0.66904654355728921 * yz * (3.0 - 7.0 * z2))
  out.append(factor * -3.1735664074561294 * z2 + 3.7024941420321507 * z4 + 0.31735664074561293)
  out.append(factor * 0.66904654355728921 * xz * (3.0 - 7.0 * z2))
  out.append(factor * 0.47308734787878004 * (x2 - y2) * (7.0 * z2 - 1.0))
  out.append(factor * 1.7701307697799304 * xz * (-x2 + 3.0 * y2))
  out.append(factor * -3.7550144126950569 * x2 * y2 + 0.62583573544917614 * x4 + 0.62583573544917614 * y4)
  if sh_degree <= 5: return jnp.stack(out, axis=-1)
  factor = cosine_easing_factor(5, alpha)
  out.append(factor * 0.65638205684017015 * y * (10.0 * x2 * y2 - 5.0 * x4 - y4))
  out.append(factor * 8.3026492595241645 * xyz * (x2 - y2))
  out.append(factor * -0.48923829943525038 * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0))
  out.append(factor * 4.7935367849733241 * xyz * (3.0 * z2 - 1.0))
  out.append(factor * 0.45294665119569694 * y * (14.0 * z2 - 21.0 * z4 - 1.0))
  out.append(factor * 0.1169503224534236 * z * (-70.0 * z2 + 63.0 * z4 + 15.0))
  out.append(factor * 0.45294665119569694 * x * (14.0 * z2 - 21.0 * z4 - 1.0))
  out.append(factor * 2.3967683924866621 * z * (x2 - y2) * (3.0 * z2 - 1.0))
  out.append(factor * -0.48923829943525038 * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0))
  out.append(factor * 2.0756623148810411 * z * (-6.0 * x2 * y2 + x4 + y4))
  out.append(factor * 0.65638205684017015 * x * (10.0 * x2 * y2 - x4 - 5.0 * y4))
  if sh_degree <= 6: return jnp.stack(out, axis=-1)
  factor = cosine_easing_factor(6, alpha)
  out.append(factor * 1.3663682103838286 * xy * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4))
  out.append(factor * 2.3666191622317521 * yz * (10.0 * x2 * y2 - 5.0 * x4 - y4))
  out.append(factor * 2.0182596029148963 * xy * (x2 - y2) * (11.0 * z2 - 1.0))
  out.append(factor * -0.92120525951492349 * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0))
  out.append(factor * 0.92120525951492349 * xy * (-18.0 * z2 + 33.0 * z4 + 1.0))
  out.append(factor * 0.58262136251873131 * yz * (30.0 * z2 - 33.0 * z4 - 5.0))
  out.append(factor * 6.6747662381009842 * z2 - 20.024298714302954 * z4 + 14.684485723822165 * z6 - 0.31784601133814211)
  out.append(factor * 0.58262136251873131 * xz * (30.0 * z2 - 33.0 * z4 - 5.0))
  out.append(factor * 0.46060262975746175 * (x2 - y2) * (11.0 * z2 * (3.0 * z2 - 1.0) - 7.0 * z2 + 1.0))
  out.append(factor * -0.92120525951492349 * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0))
  out.append(factor * 0.50456490072872406 * (11.0 * z2 - 1.0) * (-6.0 * x2 * y2 + x4 + y4))
  out.append(factor * 2.3666191622317521 * xz * (10.0 * x2 * y2 - x4 - 5.0 * y4))
  out.append(factor * 10.247761577878714 * x2 * y4 - 10.247761577878714 * x4 * y2 + 0.6831841051919143 * x6 - 0.6831841051919143 * y6)
  if sh_degree <= 7: return jnp.stack(out, axis=-1)
  factor = cosine_easing_factor(7, alpha)
  out.append(factor * 0.70716273252459627 * y * (-21.0 * x2 * y4 + 35.0 * x4 * y2 - 7.0 * x6 + y6))
  out.append(factor * 5.2919213236038001 * xyz * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4))
  out.append(factor * -0.51891557872026028 * y * (13.0 * z2 - 1.0) * (-10.0 * x2 * y2 + 5.0 * x4 + y4))
  out.append(factor * 4.1513246297620823 * xyz * (x2 - y2) * (13.0 * z2 - 3.0))
  out.append(factor * -0.15645893386229404 * y * (3.0 * x2 - y2) * (13.0 * z2 * (11.0 * z2 - 3.0) - 27.0 * z2 + 3.0))
  out.append(factor * 0.44253269244498261 * xyz * (-110.0 * z2 + 143.0 * z4 + 15.0))
  out.append(factor * 0.090331607582517306 * y * (-135.0 * z2 + 495.0 * z4 - 429.0 * z6 + 5.0))
  out.append(factor * 0.068284276912004949 * z * (315.0 * z2 - 693.0 * z4 + 429.0 * z6 - 35.0))
  out.append(factor * 0.090331607582517306 * x * (-135.0 * z2 + 495.0 * z4 - 429.0 * z6 + 5.0))
  out.append(factor * 0.07375544874083044 * z * (x2 - y2) * (143.0 * z2 * (3.0 * z2 - 1.0) - 187.0 * z2 + 45.0))
  out.append(factor * -0.15645893386229404 * x * (x2 - 3.0 * y2) * (13.0 * z2 * (11.0 * z2 - 3.0) - 27.0 * z2 + 3.0))
  out.append(factor * 1.0378311574405206 * z * (13.0 * z2 - 3.0) * (-6.0 * x2 * y2 + x4 + y4))
  out.append(factor * -0.51891557872026028 * x * (13.0 * z2 - 1.0) * (-10.0 * x2 * y2 + x4 + 5.0 * y4))
  out.append(factor * 2.6459606618019 * z * (15.0 * x2 * y4 - 15.0 * x4 * y2 + x6 - y6))
  out.append(factor * 0.70716273252459627 * x * (-35.0 * x2 * y4 + 21.0 * x4 * y2 - x6 + 7.0 * y6))
  return jnp.stack(out, axis=-1)

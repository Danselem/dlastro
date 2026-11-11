import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (10000, 10000))
y = x @ x.T

y.block_until_ready()

print("Device:", y.device)
print("Result shape:", y.shape)
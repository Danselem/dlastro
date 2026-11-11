import os
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import arviz as az
import flax.nnx as nnx
from functools import partial
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

os.makedirs("./plots", exist_ok=True)


x = jnp.array([1.0, 2.0, 3.0])
print(x)

x = jnp.array([1.0, 2.0, 3.0])
# x[0] = 10.0  # Uncommenting this would raise an error — JAX arrays are immutable

x_new = x.at[0].set(10.0)
print(f"x = {x}, x_new = {x_new}")

# Define a simple function to inspect its JAX expression (jaxpr)
def f(x):
    return jnp.sin(x) + x**2

print(jax.make_jaxpr(f)(2.0))

# Compute the derivative of f
dfdx = jax.grad(f)

print("∂f(2.0)/∂x  =", dfdx(2.0))

# Define a multi-variable function
def g(a, b):
    return a**2 + 3*b + a*b

# Compute gradients with respect to specific arguments
dg_da = jax.grad(g, argnums=0)
dg_db = jax.grad(g, argnums=1)

a, b = 2.0, 5.0
print("∂g/∂a =", dg_da(a, b))
print("∂g/∂b =", dg_db(a, b))

# Compute gradients with respect to multiple arguments simultaneously
dg_dab = jax.grad(g, argnums=(0, 1))
print("∂g/∂(a,b) =", dg_dab(a, b))

# Define a mean squared error loss function
def mse_loss(theta, x, y):
    yhat = theta[0] * x + theta[1]
    return jnp.mean((yhat - y)**2)

# Create synthetic data
x = jnp.linspace(0.0, 1.0, 100)
y = 2.0 * x + 0.1

# Create an array of parameters (slope and intercept)
theta = jnp.array([1.0, 0.0])

# Get both loss value and gradient simultaneously
loss_and_grad = jax.value_and_grad(mse_loss)

# Evaluate at theta given x and y
val, grad = loss_and_grad(theta, x, y)
print("loss =", val)
print("grad =", grad)

# Define a vector-valued function
def f_vec(x):
    return jnp.array([x**2, jnp.sin(x), jnp.exp(x)])

# Compute forward-mode and reverse-mode Jacobians
print("jacfwd at 2.0:", jax.jacfwd(f_vec)(2.0))
print("jacrev at 2.0:", jax.jacrev(f_vec)(2.0))

# Define a function mapping R^2 to R^3
def F_vec(theta):
    a, b = theta
    return jnp.array([a + b, a*b, a**2 + jnp.sin(b)])

theta = jnp.array([2.0, 0.5])

J_fwd = jax.jacfwd(F_vec)(theta)
J_rev = jax.jacrev(F_vec)(theta)

print("J_fwd:\n", J_fwd)
print("J_rev:\n", J_rev)

# Redefine data and parameters for clarity
x = jnp.linspace(0.0, 1.0, 100)
y = 2.0 * x + 0.1
theta = jnp.array([1.0, 0.0])

# Compute the Hessian of the loss function directly
H = jax.hessian(mse_loss)(theta, x, y)

print("Hessian shape:", H.shape)
print(H)

# Compute the Hessian via forward-over-reverse mode
Hp = jax.jacfwd(jax.grad(mse_loss))(theta, x, y)
print("Hessian via jacfwd/grad shape:", Hp.shape)
print(Hp)

# Compute the Hessian via reverse-over-forward mode
Hpp = jax.jacrev(jax.jacfwd(mse_loss))(theta, x, y)
print("Hessian via jacfwd/jacfwd shape:", Hpp.shape)
print(Hpp)

# Just in time compilation (JIT) for performance optimization

from datetime import datetime

# Define a toy function that performs repeated nonlinear operations
def f_heavy(x):
    for _ in range(50):
        x = jnp.sin(x) + 0.1 * x**2 - 0.05 * jnp.cos(2.0 * x)
    return jnp.mean(x)

# Define a large input array
x = jnp.linspace(-3.0, 3.0, 2000000)

# Time eager execution
t0 = datetime.now()
y = f_heavy(x)
t1 = datetime.now()
print("eager result:", y)
print("eager time  :", t1 - t0, "s")

# JIT-compile the function
f_heavy_jit = jax.jit(f_heavy)

# First call includes compilation time
t0 = datetime.now()
y = f_heavy_jit(x)
t1 = datetime.now()
print("jit (compile+run) result:", y)
print("jit (compile+run) time  :", t1 - t0, "s")

# Second call is just execution time
t0 = datetime.now()
y = f_heavy_jit(x)
t1 = datetime.now()
print("jit (run) result:", y)
print("jit (run) time  :", t1 - t0, "s")

# Alternatively, use the @jax.jit decorator
@jax.jit
def f_heavy_jitdec(x):
    for _ in range(50):
        x = jnp.sin(x) + 0.1 * x**2 - 0.05 * jnp.cos(2.0 * x)
    return jnp.mean(x)

x = jnp.linspace(-3.0, 3.0, 2000000)

# First call includes compilation time
t0 = datetime.now()
y = f_heavy_jitdec(x)
t1 = datetime.now()
print("jit_dec (compile+run) result:", y)
print("jit_dec (compile+run) time  :", t1 - t0, "s")

# Second call is just execution time
t0 = datetime.now()
y = f_heavy_jitdec(x)
t1 = datetime.now()
print("jit_dec (run) result:", y)
print("jit_dec (run) time  :", t1 - t0, "s")

# Create larger synthetic data
x_in = jnp.linspace(0.0, 1.0, 50000)
y_in = 2.0 * x_in + 0.1
theta = jnp.array([1.0, 0.0])

# Eager evaluation of loss and gradient
t0 = datetime.now()
val, grad = jax.value_and_grad(mse_loss)(theta, x_in, y_in)
t1 = datetime.now()
print("eager loss  :", val)
print("eager grad  :", grad)
print("eager time  :", t1 - t0, "s")

# JIT-compiled evaluation of loss and gradient
loss_and_grad_jit = jax.jit(jax.value_and_grad(mse_loss))

# First call includes compilation time
val_jit, grad_jit = loss_and_grad_jit(theta, x_in, y_in)

# Second call is just execution time
t0 = datetime.now()
val_jit, grad_jit = loss_and_grad_jit(theta, x_in, y_in)
t1 = datetime.now()
print("jit loss    :", val_jit)
print("jit grad    :", grad_jit)
print("jit time    :", t1 - t0, "s")

# Define a simple function
def f(x):
    jnp.sum(jnp.sin(x))

# Define inputs with different shapes
x1 = jnp.ones((1000,))
x2 = jnp.ones((2000,))  # Different shape will trigger recompilation

# JIT the function
f_jit = jax.jit(f)

# Call with different shapes
out1 = f_jit(x1)
out2 = f_jit(x2)
print("Ran with two different shapes (may compile twice).")


# Define a function with a Python control flow loop
def f_static(x, n_steps):
    for _ in range(n_steps):
        x = jnp.sin(x) + 0.1 * x**2 - 0.05 * jnp.cos(2.0 * x)
    return jnp.mean(x)

x = jnp.linspace(-3.0, 3.0, 200000)

# JIT the function without marking n_steps as static
f_static_jit_broken = jax.jit(f_static)

# Calling it fails: the Python `range(n_steps)` loop needs n_steps as a concrete
# value at trace time, but jit traces it as an abstract value
try:
    y = f_static_jit_broken(x, n_steps=50)
    print("result:", y)
except Exception as e:
    print(f"As expected, this fails:\n{type(e).__name__}: {e}")


# JIT the function with static_argnames defined
f_static_jit = jax.jit(f_static, static_argnames=("n_steps",))

# Call the function, passing n_steps as a regular argument
y = f_static_jit(x, n_steps=50)

print("result:", y)

from jax import debug

# Use jax.debug.print to print values during JIT execution
@jax.jit
def f_with_print(x):
    debug.print("Inside jit: mean(x) = {}", jnp.mean(x))
    return jnp.sum(jnp.sin(x))

_ = f_with_print(jnp.ones((10,))).block_until_ready()


# Define a function that normalizes a single vector
def fsing(x):
    return x / (jnp.linalg.norm(x) + 1e-8)

# Create a batch of 4 vectors in R^30
X = jnp.arange(120.0).reshape(4, 30)
print("X shape:", X.shape)

# Apply function using a Python for-loop
t0 = datetime.now()
outloop = []
for i in range(X.shape[0]):
    outloop.append(fsing(X[i]))
outloop = jnp.stack(outloop)

print("outloop shape:", outloop.shape)
print("Runtime: ", datetime.now() - t0, "s")

# Apply function using a list comprehension
t0 = datetime.now()
outlist = jnp.stack([fsing(X[i]) for i in range(X.shape[0])])

print("outlist shape:", outlist.shape)
print("Runtime: ", datetime.now() - t0, "s")


# Apply function using JAX vmap
t0 = datetime.now()
fvmap = jax.vmap(fsing, in_axes=0, out_axes=0)
outvmap = fvmap(X)

print("outvmap shape:", outvmap.shape)
print("Runtime: ", datetime.now() - t0, "s")

# Apply function using JAX vmap combined with JIT
fjit = jax.jit(fvmap)

t0 = datetime.now()
outjit = fjit(X)

print("outjit shape:", outjit.shape)
print("Runtime: ", datetime.now() - t0, "s")

# Compute gradients through JIT-compiled vmapped functions
fjitgrad = jax.jit(jax.vmap(jax.jacfwd(fsing), in_axes=0, out_axes=0))

t0 = datetime.now()
outgradjit = fjitgrad(X)

print("outgradjit shape:", outgradjit.shape)
print("Runtime: ", datetime.now() - t0, "s")


# Define a single-example loss function
def loss_single(theta, x, y):
    yhat = theta[0] * x + theta[1]
    return (yhat - y)**2

# Define synthetic data
theta = jnp.array([2.0, -1.0])
xarr = jnp.linspace(-1.0, 1.0, 8)
yarr = 3.0 * xarr + 0.2

# Transform loss_single into a batched version using vmap
# in_axes specifies which axes to map over for each input argument
loss_batch = jax.vmap(loss_single, in_axes=(None, 0, 0))
losses = loss_batch(theta, xarr, yarr)



# Generate synthetic data
key = jax.random.PRNGKey(0)

# Ground-truth parabola parameters
true_params = {"a": 1.5, "b": -0.8, "c": 0.2}

# Sample x values and noisy y
N = 200
x = jnp.linspace(-2.0, 2.0, N)

key, subkey = jax.random.split(key)
noise_sigma = 0.1
y = (true_params["a"] * x**2 + true_params["b"] * x + true_params["c"] 
     + noise_sigma * jax.random.normal(subkey, shape=(N,)))
yerr = noise_sigma * jnp.ones_like(y)


# Define the forward model
def model(theta, x):
    a, b, c = theta
    return a * x**2 + b * x + c

# Define mean squared error loss
def mse_loss(theta, x, y):
    yhat = model(theta, x)
    return jnp.mean((yhat - y) ** 2)

# Define a single JIT-compiled training step
@jax.jit
def train_step(theta, opt_state, x, y):
    loss_val, grads = jax.value_and_grad(mse_loss)(theta, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params=theta)
    theta = optax.apply_updates(theta, updates)
    return theta, opt_state, loss_val

# Initialize parameters
theta0 = jnp.array([0.0, 0.0, 0.0])

# Setup SGD optimizer
lr = 1E-3
optimizer = optax.sgd(learning_rate=lr)
opt_state = optimizer.init(theta0)

# Training loop for 5000 steps
num_steps = 5000
loss_history = []

for step in range(num_steps):
    if step == 0:
        theta = theta0
    
    theta, opt_state, loss_val = train_step(theta, opt_state, x, y)
    loss_history.append(loss_val)

    if step % 500 == 0:
        print(f"step {step:4d} | loss = {loss_val:.4f} | theta = {theta}")
        
theta_final = theta
print("Final fitted parameters:", theta_final)



# Generate best-fit curve and plot results
y_fit = model(theta_final, x)

fig, axlist = plt.subplots(1, 2, figsize=(12, 5))
axlist[0].scatter(x, y, label="data", s=10)
axlist[0].plot(x, y_fit, color='orange', label="fit", linewidth=2)
axlist[0].set_xlabel("x")
axlist[0].set_ylabel("y")
axlist[0].legend()

axlist[1].plot(loss_history)
axlist[1].set_yscale('log')
axlist[1].set_xlabel("Training step")
axlist[1].set_ylabel("Loss (MSE)")

# Define negative log-likelihood function assuming Gaussian errors
def nll(theta, x, y, yerr):
    r = y - model(theta, x)
    return 0.5 * jnp.sum(r**2 / yerr**2)

# Compute Hessian of NLL at best-fit parameters
H_nll = jax.hessian(nll)(theta_final, x, y, yerr)

# Invert Hessian to obtain the covariance matrix
cov = jnp.linalg.inv(H_nll)

# 1-sigma standard errors are the square root of the diagonal
se = jnp.sqrt(jnp.diag(cov))

param_names = ["a", "b", "c"]
for name, val, err in zip(param_names, theta_final, se):
    print(f"{name} = {val:.6f} ± {err:.6f}")


# Define our probabilistic model
def model(x, y=None, yerr=None):
    # Prior distributions on parameters
    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    c = numpyro.sample("c", dist.Normal(0.0, 5.0))

    # Define the forward model
    mu = a * x**2 + b * x + c
    
    # Likelihood of observations
    numpyro.sample("y", dist.Normal(mu, yerr), obs=y)
    
# Generate synthetic data for MCMC
key = jax.random.PRNGKey(0)
true_params = {"a": 1.5, "b": -0.8, "c": 0.2}

N = 200
x = jnp.linspace(-2.0, 2.0, N)

key, subkey = jax.random.split(key)
noise_sigma = 0.1


# Complete synthetic data generation
y = (true_params["a"] * x**2 + true_params["b"] * x + true_params["c"] 
     + noise_sigma * jax.random.normal(subkey, shape=(N,)))
yerr = noise_sigma * jnp.ones_like(y)

# Set up NUTS MCMC
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=1, progress_bar=True)

# Run MCMC to sample from posterior
rng_key = jax.random.PRNGKey(1)
mcmc.run(rng_key, x=x, y=y, yerr=yerr)

# Print summary of MCMC results
mcmc.print_summary()

# Extract posterior samples
samples = mcmc.get_samples()
{k: v.shape for k, v in samples.items()}



# Convert NumPyro MCMC object to ArviZ InferenceData
idata = az.from_numpyro(mcmc)

# Generate a corner plot of the posterior
var_names = ["a", "b", "c"]
az.plot_pair(
    idata,
    var_names=var_names,
    marginal_kind="kde",
    marginal=True,
    backend="matplotlib",
    figure_kwargs={"figsize": (10, 10)},
)
plt.suptitle("Posterior corner plot (NumPyro NUTS)", y=1.02)
plt.savefig("./plots/corner.png", dpi=300, bbox_inches="tight")
plt.close()




# Define a simple Multi-Layer Perceptron using Flax NNX
class MLP(nnx.Module):
    def __init__(self, *, hidden_dim=64, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(1, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.out    = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.dense1(x))
        x = nnx.relu(self.dense2(x))
        return self.out(x)

# Generate synthetic data
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
x = jnp.linspace(-2.0, 2.0, 200)[:, None]
y = 1.5 * x**2 - 0.8 * x + 0.2 + jax.random.normal(subkey, x.shape) * 0.1

# Initialize model, then split into static graph definition and parameter arrays
mlp = MLP(hidden_dim=64, rngs=nnx.Rngs(0))
model, params = nnx.split(mlp)


# Initialize optimizer state from parameters
optimizer = optax.adam(learning_rate=1E-3)
opt_state = optimizer.init(params)

# Define loss in terms of parameters and static model
@partial(jax.jit, static_argnames=("model",))
def loss_fn(params, x, y, *, model):
    m = nnx.merge(model, params)
    yhat = m(x)
    return jnp.mean((yhat - y) ** 2)

# Define a single training step to update parameter arrays
@partial(jax.jit, static_argnames=("model",))
def train_step(params, opt_state, x, y, *, model):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y, model=model)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
loss_hist = []
for step in range(5000):
    params, opt_state, loss = train_step(params, opt_state, x, y, model=model)
    loss_hist.append(loss)
    if step % 500 == 0:
        print(f"step {step:4d} | loss = {loss:.6f}")

# Rebuild trained model for prediction
trained_model = nnx.merge(model, params)
y_pred = trained_model(x)

# Plot the final MLP fit
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.scatter(x, y, s=15, label="true")
ax.plot(x, y_pred, linewidth=2, label="MLP fit")
ax.legend()
ax.set_title("Flax NNX MLP fit (appendix)")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.savefig("./plots/flaxmlp.png", dpi=300, bbox_inches="tight")
plt.close()
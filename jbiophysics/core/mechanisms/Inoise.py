import jax
import jax.numpy as jnp
import jaxley as jx
import optax

class Inoise(jx.channels.Channel):
    """Stochastic Ornstein-Uhlenbeck noise channel."""
    def __init__(self, name: str = None, initial_seed: Optional[int] = None, initial_amp_noise: Optional[float] = None, initial_tau: Optional[float] = None, initial_mean: Optional[float] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {"amp_noise": 0.01, "mean": 0.00, "tau": 20.0}
        if initial_seed is None:
            self.channel_params["seed"] = float(np.random.randint(0, 2**16 - 1))
        else:
            self.channel_params["seed"] = float(initial_seed)
        self.channel_params["amp_noise"] = float(initial_amp_noise) if initial_amp_noise is not None else 0.01
        self.channel_params["tau"] = float(initial_tau) if initial_tau is not None else 20.0
        self.channel_params["mean"] = float(initial_mean) if initial_mean is not None else 0.00
        self.channel_states = {"n": 0.00, "step": 0.0}
        self.current_name = "i_noise"

    def update_states(self, states, dt, v, params):
        n, step, seeds_int = states["n"], states["step"], params["seed"].astype(int)
        if seeds_int.ndim == 0:
            base_key = jax.random.PRNGKey(seeds_int)
        else:
            base_key = jax.vmap(jax.random.PRNGKey)(seeds_int)
        if step.ndim == 0:
            step_key = jax.random.fold_in(base_key, step.astype(int))
        else:
            step_key = jax.vmap(jax.random.fold_in)(base_key, step.astype(int))
        xi = jax.random.normal(step_key) if step_key.ndim == 1 else jax.vmap(jax.random.normal)(step_key)
        mu, sigma, tau = params["mean"], params["amp_noise"], params["tau"]
        drift = (mu - n) / tau * dt
        diffusion = sigma * jnp.sqrt(2.0 / tau) * xi * jnp.sqrt(dt)
        new_n = n + drift + diffusion
        new_n = jnp.where(jnp.isnan(new_n) | jnp.isinf(new_n), n, new_n)
        return {"n": new_n, "step": step + 1.0}

    def compute_current(self, states, v, params): return -states["n"]
    def init_state(self, states, v, params, delta_t): return {"n": params["mean"], "step": 0.0}
import jax
import jax.numpy as jnp
import jaxley as jx
import optax

def calculate_axial_current(traces_soma, traces_dend, ra=100.0):
    """Calculates the axial current (MEG dipole approximation)."""
    return (traces_dend - traces_soma) / ra

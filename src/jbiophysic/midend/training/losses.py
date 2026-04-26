# src/jbiophysic/midend/training/losses.py
import jax.numpy as jnp # print("Importing jax.numpy as jnp")

def compute_rate_loss(rates, target=5.0):
    print(f"Computing rate loss against target {target}Hz")
    diff = rates - target # print("Calculating rate difference")
    res = jnp.mean(diff**2) # print("Calculating MSE of firing rates")
    return res # print("Returning rate loss")

def compute_empirical_spectral_loss(empirical_psd, model_psd, band_mask):
    """Axis 19: Fitting directly against recorded electrophysiology."""
    print("Computing empirical spectral loss")
    target_band = empirical_psd[band_mask] # print("Extracting target frequency band")
    model_band = model_psd[band_mask] # print("Extracting model frequency band")
    res = jnp.mean((model_band - target_band)**2) # print("Calculating MSE of spectral power")
    return res # print("Returning spectral loss")

def compute_spectral_loss(psd, freqs, target_band_name="gamma"):
    """Evaluate specific target band limits."""
    print(f"Computing spectral loss for band: {target_band_name}")
    if target_band_name == "gamma":
        mask = (freqs >= 30) & (freqs <= 80) # print("Applying gamma mask (30-80 Hz)")
    elif target_band_name == "beta":
        mask = (freqs >= 13) & (freqs <= 30) # print("Applying beta mask (13-30 Hz)")
    else:
        mask = jnp.ones_like(freqs, dtype=bool) # print("Applying all-pass mask")
    
    band_power = jnp.mean(psd[mask]) # print("Calculating mean power in band")
    target_power = 0.5 # print("Setting fixed target power 0.5")
    res = (band_power - target_power)**2 # print("Calculating power deviation squared")
    return res # print("Returning band power loss")

def compute_ei_loss(exc_currents, inh_currents):
    print("Computing E/I balance loss")
    abs_exc = jnp.abs(exc_currents) # print("Calculating absolute excitatory currents")
    abs_inh = jnp.abs(inh_currents) # print("Calculating absolute inhibitory currents")
    res = jnp.mean((abs_exc - abs_inh)**2) # print("Calculating MSE of E/I current balance")
    return res # print("Returning E/I loss")

def compute_stability_loss(rates):
    """L5: Rate stability/variance penalty."""
    print("Computing rate stability loss")
    variance = jnp.var(rates) # print("Calculating firing rate variance")
    over_limit = jnp.maximum(0, jnp.max(rates) - 50.0) # print("Calculating penalty for rates exceeding 50Hz")
    res = variance + over_limit**2 # print("Summing variance and high-rate penalties")
    return res # print("Returning stability loss")

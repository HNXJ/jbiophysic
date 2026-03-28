# Adding a Custom Channel/Synapse

1. Subclass `jx.channels.Channel` or `jx.synapses.Synapse`
2. Define `channel_params`/`synapse_params`, `channel_states`/`synapse_states`
3. Implement `update_states()`, `compute_current()`, `init_state()`
4. Add NaN/Inf barrier: `jnp.where(jnp.isnan(new_s), s, new_s)`
5. Register in `jbiophysics/core/mechanisms/__init__.py`
6. Add to `SYNAPSE_TYPES` in `compose.py` for NetBuilder support

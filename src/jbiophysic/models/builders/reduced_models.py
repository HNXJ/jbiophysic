# src/jbiophysic/models/builders/reduced_models.py
import jax
import jax.numpy as jnp
import equinox as eqx

class Izhikevich(eqx.Module):
    """
    Izhikevich Point Neuron Model (Axis 5).
    DynaSim parity: 'Izh.pop'.
    
    ### Equations: 
    - dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    - du/dt = a*(b*v - u)
    
    ### Reset Logic: 
    - if v >= 30: v = c, u = u + d
    
    ### Useful Notes
    - **Canonical Params**:
        - Regular Spiking (RS): a=0.02, b=0.2, c=-65, d=8
        - Chattering (CH): a=0.02, b=0.2, c=-50, d=2
        - Fast Spiking (FS): a=0.1, b=0.2, c=-65, d=2
    - **Differentiability**: Uses `jax.lax.select` to allow gradient-based 
      optimization of (a, b, c, d) parameters.
    """
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 6.0
    
    def __call__(self, v, u, I, dt):
        dv = 0.04 * v**2 + 5 * v + 140 - u + I
        du = self.a * (self.b * v - u)
        
        v_next = v + dv * dt
        u_next = u + du * dt
        
        # Reset logic (Differentiable via jax.lax.select)
        is_spike = v_next >= 30.0
        v_final = jax.lax.select(is_spike, self.c, v_next)
        u_final = jax.lax.select(is_spike, u_next + self.d, u_next)
        
        return v_final, u_final, is_spike

class FitzHughNagumo(eqx.Module):
    """
    FitzHugh-Nagumo (FHN) Model.
    DynaSim parity: 'FHN.pop'.
    
    ### Equations:
    - dv/dt = v - v^3/3 - w + I
    - dw/dt = (v + a - b*w) / tau
    
    ### Useful Notes
    - **Phase Plane**: This model is excellent for teaching oscillatory 
      dynamics. The cubic nullcline for 'v' creates the excitability threshold.
    - **Optimization**: Often used as a low-dimensional surrogate to 
      understand the bifurcation structure of the full HH model.
    """
    a: float = 0.7
    b: float = 0.8
    tau: float = 12.5
    
    def __call__(self, v, w, I, dt):
        dv = v - (v**3) / 3.0 - w + I
        dw = (v + self.a - self.b * w) / self.tau
        
        return v + dv * dt, w + dw * dt

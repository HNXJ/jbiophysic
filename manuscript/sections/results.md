# 3. Results

## 3.1. Mathematical Validation of Cortical Oscillations
To ensure that the hierarchical network produced biophysically faithful oscillations rather than numerical artifacts, we subjected the multi-population dynamics (E, PV, SST, VIP) to a rigorous mathematical validation framework. First, mapping the system's steady-states into a Jacobian matrix revealed non-zero imaginary components in the eigenvalues ($\lambda = a \pm i\omega$), mathematically confirming the presence of stable limit cycle attractors.

Furthermore, Hilbert transformation of the Pyramidal (E) and Parvalbumin (PV) time series demonstrated a persistent geometric phase lag converging at $\approx 90^\circ$ ($\pi/2$), a canonical signature of true E/I feedback loops. The addition of biological noise ($\sigma=0.05$) to the simulation did not collapse the trajectories; rather, the system exhibited robust noise-driven resonance with a broad PSD peak, distinct from the sharp unphysiological tracking spikes typical of linear-clipped models. These results prove the network is operating via realistic oscillatory attractors suitable for predictive coding evaluation.

## 3.2. Mapping the Inhibitory Control Landscape
We sought to systematically evaluate how interneuron subtypes control the transition between sensory-driven and prediction-driven dynamics. By scanning the perisomatic feedback parameter space ($G_{PV}$) against the dendritic gating parameter space ($G_{SST}$), we observed a distinct temporal bifurcation. 

As predicted by microcircuit canonical theories (Bastos et al., 2012):
1. **Gamma-Dominance (Feedforward Mode)** emerges exclusively in high-$PV$, low-$SST$ states. PV interneurons track and restrict the temporal window for local Pyramidal firing, amplifying high-frequency precision (~45 Hz).
2. **Beta-Dominance (Feedback Mode)** emerges when $SST$ scaling is elevated. SST networks suppress superficial inputs and disinhibit slow, deep-layer predictive synchronization (~20 Hz).

## 3.3. VIP Disinhibition and the Balanced "Omission" Regime
Real cortical environments require a temporal "Balance Regime", avoiding pathological runaway excitation while allowing task-dependent context switching. In our phase landscape, this Balanced Regime emerged distinctly at moderate levels of PV and SST coupling ($\alpha_{PV} = 1.0, \alpha_{SST} = 1.0$). 

Crucially, when challenged with the Omission Predictability Task (Sequence: S1 $\rightarrow$ S2 $\rightarrow$ $\emptyset$), only networks operating at or near the Balanced Regime reproduced the empirical $LFP$ signature. As sequence predictions violated external sensory drive, the network correctly resolved the omission via VIP-mediated disinhibition of SST interneurons. This specific microcircuit switch increased the $\Delta \beta_{omission}$ power and attenuated top-down prediction errors. 
Through this, we mathematically trace sequence-predictive behavioral tracking directly to the precise tuning of the cortical inhibitory subtypes.

"""Smoke run for JTFNE correct vs inverse spectrolaminar scaffold."""
from pathlib import Path
from jbiophysic import jtfne

out = Path("outputs/jtfne_smoke_compare")
out.mkdir(parents=True, exist_ok=True)
rows = []
for mode in ("correct", "inverse"):
    cfg = jtfne.default_cfg(mode, smoke=True)
    cfg = type(cfg)(init=cfg.init, sim=cfg.sim, vis=type(cfg.vis)(output_dir=str(out / mode), write_html=True, write_json=True, show=False), opt=cfg.opt)
    model = jtfne.construct(cfg.init)
    signals = jtfne.simulate(model, cfg.sim)
    ev = jtfne.evaluate(signals, cfg.opt)
    jtfne.visualize(signals, cfg.vis)
    rows.append({"mode": mode, "mean_similarity": ev.mean_similarity, "min_similarity": ev.min_similarity})
print(rows)
if not rows[0]["mean_similarity"] > rows[1]["mean_similarity"]:
    raise SystemExit("correct mode did not score above inverse mode")

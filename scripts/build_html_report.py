import os
import json
import base64

def build_html_report():
    os.makedirs("results", exist_ok=True)
    
    # Load metrics
    with open("results/data/metrics.json", "r") as f:
        metrics = json.load(f)
        
    contexts = ["ff_only", "spontaneous", "attended", "omission"]
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Two-Column Cortical Omission Paradigm - Research Report</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/reveal.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/theme/black.min.css">
    <style>
        :root {{
            --gold: #CFB87C;
            --violet: #9400D3;
            --cyan: #00FFFF;
            --bg: #0D0D0F;
        }}
        .reveal {{ background: var(--bg); color: #E8E8E8; }}
        .reveal h1, .reveal h2, .reveal h3 {{ color: var(--gold); text-transform: none; }}
        .reveal .controls {{ color: var(--gold); }}
        .reveal .progress {{ color: var(--gold); }}
        
        .metric-card {{
            background: #1A1A1C;
            border-left: 4px solid var(--gold);
            padding: 15px;
            margin: 10px;
            display: inline-block;
            min-width: 200px;
            text-align: left;
        }}
        .metric-value {{ font-size: 1.2em; color: var(--cyan); font-weight: bold; }}
        .metric-label {{ font-size: 0.7em; color: #888; text-transform: uppercase; }}
        
        .fig-containerimg {{
            max-width: 90%;
            border: 1px solid #333;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="reveal">
        <div class="slides">
            <!-- Intro -->
            <section>
                <h1>Biophysical Omission Simulation</h1>
                <h3>Two-Column Cortical Architecture</h3>
                <p style="color: var(--violet)">Bastos/Markov Laminar Logic | V1 + HO</p>
                <div style="font-size: 0.6em; margin-top: 50px;">
                    <p><b>Investigator:</b> Hamed Nejat</p>
                    <p><b>Affiliation:</b> Bastoslab, Vanderbilt University</p>
                </div>
            </section>

            <!-- Architecture -->
            <section>
                <h2>Network Architecture</h2>
                <div style="display: flex; align-items: center; justify-content: center;">
                    <div style="text-align: left; font-size: 0.7em;">
                        <ul>
                            <li><b>V1 Column:</b> 200 neurons (HH Mechanisms)</li>
                            <li><b>HO Column:</b> 100 neurons (HH Mechanisms)</li>
                            <li><b>Feedforward:</b> V1-L2/3 &rarr; HO-L4 (AMPA)</li>
                            <li><b>Feedback:</b> HO-L5/6 &rarr; V1-L2/3 (AMPA+NMDA)</li>
                        </ul>
                    </div>
                </div>
            </section>
"""

    for ctx, m in zip(contexts, metrics):
        html += f"""
            <section>
                <section>
                    <h2>Simulation Context: {m['context']}</h2>
                    <div class="metric-card">
                        <div class="metric-label">V1 MFR</div>
                        <div class="metric-value">{m['mfr_v1']:.1f} Hz</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">HO MFR</div>
                        <div class="metric-value">{m['mfr_ho']:.1f} Hz</div>
                    </div>
                    <div>
                        <img src="data/{ctx}_raster.png" style="max-height: 400px; width: auto;">
                    </div>
                </section>
                <section>
                    <h3>LFP & Rhythms: {m['context']}</h3>
                    <img src="data/{ctx}_lfp.png" style="max-height: 400px; width: auto;">
                </section>
                <section>
                    <h3>Spectral Profile: {m['context']}</h3>
                    <img src="data/{ctx}_tfr.png" style="max_height: 400px; width: auto;">
                </section>
            </section>
"""

    html += """
            <!-- Conclusion -->
            <section>
                <h2>Summary of Findings</h2>
                <div style="text-align: left; font-size: 0.7em;">
                    <ul>
                        <li><b style="color:var(--cyan)">HO Prediction:</b> Higher activity in Omission context suggests stable prediction signals.</li>
                        <li><b style="color:var(--gold)">Spectral Signatures:</b> Beta-band dominance in feedback-driven states.</li>
                        <li><b style="color:var(--violet)">Laminar Motif:</b> FF drive correlates with Gamma; FB drive with Alpha/Beta.</li>
                    </ul>
                </div>
            </section>
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/reveal.min.js"></script>
    <script>
        Reveal.initialize({
            hash: true,
            center: true,
            transition: 'slide'
        });
    </script>
</body>
</html>
"""

    with open("results/index.html", "w") as f:
        f.write(html)
        
    print("✅ HTML Interactive Report Generated at 'results/index.html'")

if __name__ == "__main__":
    build_html_report()

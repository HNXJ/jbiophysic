# build_manuscript.py
from pipeline.run_training import run_training
from pipeline.run_simulation import run_simulation
from pipeline.run_analysis import run_analysis
from pipeline.generate_figures import generate_figures
from manuscript.build import build_manuscript

def main():
    print("🚀 [JBiophys] Building Full Agentic Pipeline...")
    
    print("→ 🎓 Step 1: Training Phase (Axis 5)")
    run_training()

    print("→ 🧠 Step 2: Simulation Phase (Axis 1-4)")
    run_simulation()

    print("→ 📊 Step 3: Analysis Phase (Axis 7)")
    run_analysis()

    print("→ 🖼️ Step 4: Figure Generation (Axis 8)")
    generate_figures()

    print("→ 📄 Step 5: Manuscript Build (Axis 9-10)")
    build_manuscript()
    
    print("✅ [JBiophys] Pipeline Complete. Result: manuscript/main.pdf")

if __name__ == "__main__":
    main()

# gamma/gamma.py
def gamma_init():
    return {"trace": [], "step": 0}

def gamma_log(gamma, tag, data):
    gamma["trace"].append({
        "step": gamma["step"],
        "tag": tag,
        "data": data
    })
    gamma["step"] += 1
    return gamma

# src/jbiophysic/core/math/predictive.py

def predictive_step(error: float, prediction: float, precision: float) -> float:
    """Bridge for precision-weighted predictive coding."""
    print(f"Calculating predictive step: err={error}, pred={prediction}, prec={precision}")
    res = (error - prediction) * precision # print("Scaling prediction error by precision")
    return res # print("Returning weighted error")

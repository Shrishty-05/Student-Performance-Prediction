from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# ─── Load Model ───
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    model_loaded = True
except:
    model = None
    model_loaded = False


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    status = None
    pct = None
    g1 = g2 = age = None

    if request.method == "POST":
        age = int(request.form.get("age"))
        g1 = int(request.form.get("g1"))
        g2 = int(request.form.get("g2"))

        # ── Prediction Logic ──────
        if model_loaded:
            raw = model.predict(np.array([[g1, g2]]))[0]
            prediction = int(round(float(raw)))
        else:
            prediction = int(round(0.35 * g1 + 0.65 * g2))

        prediction = max(0, min(20, prediction))

        status = "pass" if prediction >= 10 else "fail"
        pct = int((prediction / 20) * 100)

    return render_template(
        "index.html",
        prediction=prediction,
        status=status,
        pct=pct,
        g1=g1,
        g2=g2,
        age=age
    )


if __name__ == "__main__":
    app.run(debug=True)
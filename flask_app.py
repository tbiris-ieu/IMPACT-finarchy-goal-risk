"""
Finarchy: goal abandonment risk predictor.

Loads the trained logistic regression model (fintech.joblib) and serves it in
two ways:

  GET  /              HTML form for users
  POST /predict       HTML form handler, renders the prediction on the page
  GET  /api/predict   JSON API, inputs as query-string parameters
  POST /api/predict   JSON API, inputs as a JSON body (or form fields)

Both the page and the API use the same two helpers: parse_inputs() validates
the three input variables and predict_goal() applies the model.
"""
import json
import math
import os
import sys

import pandas as pd
from flask import Flask, jsonify, render_template, request
from joblib import load
from werkzeug.middleware.proxy_fix import ProxyFix

import fintech  # the training script; only used if the saved model cannot be loaded

# Resolve the model file relative to this script so the app also works when
# it is started from another working directory (for example on PythonAnywhere).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fintech.joblib")

# The model's input variables, in the order used for training (see fintech.py).
FEATURES = [
    "months_since_goal_created",
    "avg_monthly_contribution",
    "missed_contributions_last_6_months",
]

# Range of each variable in the training data. Only used to warn the caller
# when a prediction is an extrapolation.
TRAINING_RANGE = {
    "months_since_goal_created": (1, 36),
    "avg_monthly_contribution": (10, 500),
    "missed_contributions_last_6_months": (0, 6),
}

# Probability thresholds for the risk bands. "high" always matches a
# prediction of 1 (the model's own 0.5 cut-off).
MEDIUM_RISK_FROM = 0.30

RECOMMENDED_ACTION = {
    "low": "No action needed. Keep the standard monthly reminder.",
    "medium": "Send a nudge before the next contribution date and show the saver "
              "how close the goal already is.",
    "high": "Flag for intervention: offer a smaller monthly amount or a later goal "
            "date before the saver gives up on the goal.",
}

app = Flask(__name__)
app.json.sort_keys = False  # keep the response keys in a readable order

# Behind a hosting proxy (for example PythonAnywhere) this makes request.url_root
# report the public https:// address, so the examples shown on the page are right.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)


def load_model():
    """
    Load the trained model from fintech.joblib and check that it works.

    The file was saved with scikit-learn 1.8.0. A server with a different
    scikit-learn version may load it but fail when predicting (for example
    1.6 raises "no attribute 'multi_class'"). In that case the model is retrained
    with fintech.py, which is seeded and produces the same model, and the new
    file replaces the old one so the next start loads it directly.
    """
    try:
        model = load(MODEL_PATH)
        model.predict_proba(pd.DataFrame([[12, 150.0, 3]], columns=FEATURES))
        return model
    except Exception as exc:
        print(f"Could not use {MODEL_PATH} ({exc!r}). Retraining it with fintech.py.",
              file=sys.stderr)
        return fintech.train_and_save(write_data=False, verbose=False)


model = load_model()


@app.after_request
def allow_cross_origin_api_calls(response):
    """Let browser-based applications on other domains call the API."""
    if request.path.startswith("/api/"):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


class InputError(ValueError):
    """Raised when one of the input variables is missing or invalid."""


def _number(source, name, whole=False, minimum=None, maximum=None):
    """Read one numeric input from a dict-like source and validate it."""
    raw = source.get(name)
    if raw is None or str(raw).strip() == "":
        raise InputError(f"Missing input: {name}.")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a number.")
    if not math.isfinite(value):
        raise InputError(f"{name} must be a number.")
    if whole:
        if value != int(value):
            raise InputError(f"{name} must be a whole number.")
        value = int(value)
    if minimum is not None and value < minimum:
        raise InputError(f"{name} must be {minimum} or more.")
    if maximum is not None and value > maximum:
        raise InputError(f"{name} must be between {minimum} and {maximum}.")
    return value


def parse_inputs(source):
    """
    Validate the three model inputs.

    `source` can be request.form, request.args or a parsed JSON body: anything
    with a .get(name) method. Returns a dict with the cleaned values or raises
    InputError with a message that explains what to fix.
    """
    return {
        "months_since_goal_created": _number(
            source, "months_since_goal_created", whole=True, minimum=0),
        "avg_monthly_contribution": _number(
            source, "avg_monthly_contribution", minimum=0),
        "missed_contributions_last_6_months": _number(
            source, "missed_contributions_last_6_months", whole=True, minimum=0, maximum=6),
    }


def predict_goal(inputs):
    """Apply the trained model to one goal and describe the result."""
    X = pd.DataFrame([inputs], columns=FEATURES)
    probability_abandoned = float(model.predict_proba(X)[0, 1])
    prediction = int(model.predict(X)[0])  # 1 = abandoned before the goal date

    if prediction == 1:
        risk_level = "high"
    elif probability_abandoned >= MEDIUM_RISK_FROM:
        risk_level = "medium"
    else:
        risk_level = "low"

    warnings = []
    for name, (low, high) in TRAINING_RANGE.items():
        if not low <= inputs[name] <= high:
            warnings.append(
                f"{name} = {inputs[name]} is outside the range seen in training "
                f"({low} to {high}), so treat this prediction with care.")

    return {
        "inputs": inputs,
        "prediction": prediction,
        "prediction_label": "goal_abandoned" if prediction == 1 else "on_track",
        "probability_abandoned": round(probability_abandoned, 4),
        "probability_on_track": round(1 - probability_abandoned, 4),
        "risk_level": risk_level,
        "recommended_action": RECOMMENDED_ACTION[risk_level],
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# HTML front end
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Show the form with example values and no prediction yet."""
    return render_template("index.html", form={}, result=None, error=None)


@app.route("/predict", methods=["POST"])
def predict_form():
    """Handle the HTML form and render the prediction on the same page."""
    try:
        result = predict_goal(parse_inputs(request.form))
        error = None
    except InputError as exc:
        result, error = None, str(exc)
    return render_template(
        "index.html",
        form=request.form,
        result=result,
        error=error,
        result_json=json.dumps(result, indent=2) if result else None,
    )


# ---------------------------------------------------------------------------
# JSON API for other applications
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["GET", "POST"])
def api_predict():
    """
    Return the prediction as JSON.

    POST: send the three inputs as a JSON object (Content-Type: application/json)
          or as form fields.
    GET:  send them as query-string parameters, for example
          /api/predict?months_since_goal_created=12&avg_monthly_contribution=150
          &missed_contributions_last_6_months=3
    """
    if request.method == "POST":
        source = request.get_json(silent=True)
        if not isinstance(source, dict):
            source = request.form
    else:
        source = request.args

    try:
        return jsonify(predict_goal(parse_inputs(source)))
    except InputError as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True)

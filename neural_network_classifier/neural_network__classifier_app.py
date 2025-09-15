from flask import Flask, request, jsonify
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# adjust these paths if you saved artifacts elsewhere
BASE_DIR = os.path.dirname(__file__)
PREPROC_PATH = os.path.join(BASE_DIR, "expansion_preprocessor.joblib")
FEATS_PATH = os.path.join(BASE_DIR, "expansion_features.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "expansion_mlp.pt")


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=[256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_artifacts():
    pre = joblib.load(PREPROC_PATH)
    feat_dict = joblib.load(FEATS_PATH)
    num_feats = feat_dict["num_feats"]
    cat_feats = feat_dict["cat_feats"]
    return pre, num_feats, cat_feats


preprocessor, NUM_FEATS, CAT_FEATS = load_artifacts()


def prepare_dataframe(payload):
    """
    Accepts either a dict (single row) or list of dicts (multiple rows).
    Ensures columns for num+cat features exist and have sensible defaults.
    """
    single = False
    if isinstance(payload, dict):
        payload = [payload]
        single = True

    # Build DataFrame with all expected columns
    rows = []
    for item in payload:
        row = {}
        # numeric features default to 0.0
        for c in NUM_FEATS:
            v = item.get(c, None)
            if v is None or v == "":
                row[c] = 0.0
            else:
                try:
                    row[c] = float(v)
                except Exception:
                    row[c] = 0.0
        # categorical features default to "missing"
        for c in CAT_FEATS:
            v = item.get(c, None)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                row[c] = "missing"
            else:
                row[c] = str(v)
        rows.append(row)

    df = pd.DataFrame(rows, columns=(NUM_FEATS + CAT_FEATS))
    return df, single


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON API:
    - POST a single object: {"mrr": 10.5, "prev_mrr": 5.0, "plan": "free", ...}
    - or POST a list of objects for batch scoring.
    Response: JSON with expansion_score (0-1) or list of scores.
    """
    payload = request.get_json(force=True)
    df, single = prepare_dataframe(payload)

    # transform
    X = preprocessor.transform(df)

    # build model with correct input dim and load weights
    input_dim = X.shape[1]
    model = MLP(input_dim=input_dim)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        t = torch.tensor(X.astype("float32"))
        logits = model(t)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    if single:
        return jsonify({"expansion_score": float(np.round(probs[0], 4))})
    else:
        return jsonify({"expansion_scores": [float(np.round(float(p), 4)) for p in probs]})


@app.route("/", methods=["GET"])
def index():
    return (
        "Neural network expansion scorer. POST JSON to /predict with keys: "
        + ", ".join(NUM_FEATS + CAT_FEATS)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
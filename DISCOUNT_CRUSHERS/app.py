from flask import Flask, render_template, request

app = Flask(__name__)

# === Your fitted logistic regression (statsmodels output) ===
# Logit(upgrade=1) = const + b_disc*discount_applied + b_seats*seats
#                     + b_tickets*support_tickets + b_total*total
COEF = {
    "const": 0.5190,
    "discount_applied": 0.7029,
    "seats": 0.0041,
    "support_tickets": -0.2085,
    "total": -0.2462,
}

def sigmoid(z):
    return 1.0 / (1.0 + (2.718281828459045 ** (-z)))  # fast exp alternative is fine here

def predict_upgrade_prob(seats, support_tickets, total, discount_flag):
    z = (
        COEF["const"]
        + COEF["discount_applied"] * discount_flag
        + COEF["seats"] * seats
        + COEF["support_tickets"] * support_tickets
        + COEF["total"] * total
    )
    return float(sigmoid(z))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            seats = int(request.form.get("seats", "0").strip())
            support_tickets = int(request.form.get("support_tickets", "0").strip())
            total = int(request.form.get("total", "0").strip())

            if seats < 0 or support_tickets < 0 or total < 0:
                raise ValueError("Inputs must be non‑negative.")

            # Predict WITHOUT discount (d=0) and WITH discount (d=1)
            p_no_disc = predict_upgrade_prob(seats, support_tickets, total, discount_flag=0)
            p_with_disc = predict_upgrade_prob(seats, support_tickets, total, discount_flag=1)
            delta = p_with_disc - p_no_disc

            # Decision rule (your spec):
            # - If d=1 pushes prob to >= 0.5 and d=0 < 0.5  -> Offer discount
            # - If d=0 already >= 0.5                        -> No discount needed
            # - Else                                         -> Discount not sufficient
            if p_no_disc >= 0.5:
                decision = "no_discount_needed"
                decision_text = "No Discount Needed!"
                tone = "good"
            elif p_with_disc >= 0.5:
                decision = "offer_discount"
                decision_text = "Offer Discount!"
                tone = "ok"
            else:
                decision = "not_enough"
                decision_text = "Discount Not Sufficient"
                tone = "bad"

            result = {
                "p_no_disc": p_no_disc,
                "p_with_disc": p_with_disc,
                "delta": delta,
                "decision": decision,
                "decision_text": decision_text,
                "tone": tone,
                "inputs": {
                    "seats": seats,
                    "support_tickets": support_tickets,
                    "total": total,
                },
            }

        except Exception as e:
            error = f"{type(e).__name__}: {e}"

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
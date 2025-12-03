from fastapi import FastAPI
# Import the routers from the individual model files
from backend.models import logistic_regression
from backend.models import naive_bayes

# --- FastAPI Initialization ---
app = FastAPI(
    title="Toxicity Classification API (Modular)",
    description="Endpoint for real-time text toxicity analysis across 10 categories, featuring multiple model backends."
)

# --- Include Routers ---
# This adds the endpoints from the model files to the main application.
# Endpoints will be accessible at:
# /logistic_regression/predict
# /naive_bayes/predict
app.include_router(logistic_regression.router)
app.include_router(naive_bayes.router)
# add app.incude_router(mi esquema) bunty

# Optional: Add a root health check endpoint
@app.get("/health")
def read_health():
    return {"status": "ok"}
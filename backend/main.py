from fastapi import FastAPI

# Import the routers from the individual model files
from backend.models import logistic_regression
from backend.models import naive_bayes
from backend.models import svm_optimized

# --- FastAPI Initialization ---
app = FastAPI(
    title="Toxicity Classification API (Modular)",
    description="Endpoint for real-time text toxicity analysis across 10 categories, featuring multiple model backends.",
)

# --- Include Routers ---
# This adds the endpoints from the model files to the main application.
# Endpoints will be accessible at:
# /logistic_regression/predict
# /naive_bayes/predict
# /svm_optimized/predict
app.include_router(logistic_regression.router)
app.include_router(naive_bayes.router)
app.include_router(svm_optimized.router)


# Optional: Add a root health check endpoint
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "API is running. Use /logistic_regression/predict, /naive_bayes/predict or /svm_optimized/predict for predictions.",
    }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import joblib
from src.simulate_data import generate_class
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL LOAD
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

model = joblib.load(MODEL_PATH)


# -----------------------------
# INPUT SCHEMA (IMPORTANT FIX)
# -----------------------------
class Student(BaseModel):
    name: str = "Student"
    attendance: float
    study_hours: float
    quiz_avg: float
    assignment_avg: float
    past_score: float


# -----------------------------
# SINGLE PREDICTION
# -----------------------------
@app.post("/predict")
def predict(data: Student):

    features = np.array([[ 
        data.attendance,
        data.study_hours,
        data.quiz_avg,
        data.assignment_avg,
        data.past_score
    ]])

    prob = model.predict_proba(features)[0][1]

    if prob < 0.4:
        risk = "LOW RISK 🟢"
    elif prob < 0.7:
        risk = "MEDIUM RISK 🟡"
    else:
        risk = "HIGH RISK 🔴"

    return {
        "name": data.name,
        "pass_probability": float(prob),
        "risk_level": risk,
        "explanations": explain_student(data)
    }


# -----------------------------
# EXPLANATION ENGINE
# -----------------------------
def explain_student(data):

    reasons = []

    if data.attendance < 60:
        reasons.append("Low attendance is affecting performance")

    if data.study_hours < 3:
        reasons.append("Insufficient study hours")

    if data.quiz_avg < 50:
        reasons.append("Weak quiz performance")

    if data.assignment_avg < 50:
        reasons.append("Assignments are weak")

    if data.past_score < 50:
        reasons.append("Poor academic history")

    if not reasons:
        reasons.append("Student is performing consistently well")

    return reasons


# -----------------------------
# BATCH ANALYTICS (CLASS VIEW)
# -----------------------------
@app.post("/batch_predict")
def batch_predict(students: List[dict]):

    results = []
    low = 0
    medium = 0
    high = 0

    for data in students:

        features = np.array([[
            data["attendance"],
            data["study_hours"],
            data["quiz_avg"],
            data["assignment_avg"],
            data["past_score"]
        ]])

        prob = model.predict_proba(features)[0][1]

        if prob < 0.4:
            risk = "LOW"
            low += 1
        elif prob < 0.7:
            risk = "MEDIUM"
            medium += 1
        else:
            risk = "HIGH"
            high += 1

        results.append({
            "name": data.get("name", "Student"),
            "probability": float(prob),
            "risk": risk,

            # ✅ IMPORTANT FIX FOR SCATTER CHART
            "attendance": data["attendance"],
            "study_hours": data["study_hours"],
            "quiz_avg": data["quiz_avg"],
            "assignment_avg": data["assignment_avg"],
            "past_score": data["past_score"]
        })

    return {
        "students": results,
        "summary": {
            "low": low,
            "medium": medium,
            "high": high
        }
    }
@app.get("/simulate_class")
def simulate_class():
    df = generate_class(50)

    return {
        "students": df.to_dict(orient="records")
    }
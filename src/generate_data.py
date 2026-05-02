import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "attendance": np.random.randint(40, 100, n),
    "study_hours": np.random.randint(0, 10, n),
    "quiz_avg": np.random.randint(20, 100, n),
    "assignment_avg": np.random.randint(20, 100, n),
    "past_score": np.random.randint(20, 100, n),
})

# 🔥 FIXED SCORING (balanced distribution)
df["score"] = (
    df["attendance"] * 0.25 +
    df["study_hours"] * 6 +
    df["quiz_avg"] * 0.25 +
    df["assignment_avg"] * 0.25 +
    df["past_score"] * 0.25
)

# 🔥 IMPORTANT FIX: lower threshold
df["pass"] = (df["score"] > 120).astype(int)

print("Class distribution:")
print(df["pass"].value_counts())

df.to_csv("data/students.csv", index=False)
print("Dataset created successfully!")
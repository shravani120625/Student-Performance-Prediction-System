import numpy as np
import pandas as pd

def generate_class(n=100):
    data = pd.DataFrame({
        "attendance": np.random.randint(40, 100, n),
        "study_hours": np.random.randint(1, 10, n),
        "quiz_avg": np.random.randint(30, 100, n),
        "assignment_avg": np.random.randint(30, 100, n),
        "past_score": np.random.randint(30, 100, n),
    })
    return data


# test run
if __name__ == "__main__":
    df = generate_class(10)
    print(df.head())
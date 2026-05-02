import pandas as pd

df = pd.read_csv("data/students.csv")

print("\n📊 Dataset Insights")
print("----------------------")

print("Total Students:", len(df))
print("Pass %:", df["pass"].mean() * 100)

print("\nFeature Averages:")
print(df.mean(numeric_only=True))
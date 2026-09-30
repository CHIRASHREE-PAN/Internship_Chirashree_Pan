import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 1. Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Basic information
print("\nDataset Information:")
print(df.info())

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())


# 2. Handle missing data

# Fill Age missing values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked missing values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill Fare missing values with median
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# 3. Encode Sex using LabelEncoder

label_encoder = LabelEncoder()

df["Sex"] = label_encoder.fit_transform(df["Sex"])

print("\nAfter Label Encoding Sex:")
print(df[["Sex"]].head())


# 4. Encode Embarked using OneHotEncoder

one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

embarked_encoded = one_hot_encoder.fit_transform(df[["Embarked"]])

embarked_columns = one_hot_encoder.get_feature_names_out(["Embarked"])

embarked_df = pd.DataFrame(
    embarked_encoded,
    columns=embarked_columns,
    index=df.index
)

# Add encoded columns
df = pd.concat([df, embarked_df], axis=1)

# Remove original Embarked column
df.drop("Embarked", axis=1, inplace=True)


# 5. Visualize Age Distribution

plt.figure(figsize=(8, 5))

sns.histplot(df["Age"], bins=30, kde=True)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.show()


# 6. Save cleaned dataset

df.to_csv("Titanic_Cleaned.csv", index=False)

print("\nCleaned dataset saved as Titanic_Cleaned.csv")

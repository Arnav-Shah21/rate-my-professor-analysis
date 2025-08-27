#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 01:10:00 2024

@author: arnavshah
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.decomposition import PCA
from scipy import stats
from scipy import stats
from scipy.stats import levene
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

# Seeding the random number generator
n_number = 19615707
np.random.seed(n_number)

# Define column names for the dataset
column_names = [
    "Average Rating", "Average Difficulty", "Number of Ratings", 
    "Received a Pepper", "Proportion of Students Retaking", 
    "Number of Online Ratings", "Male", "Female"
]

# Load the dataset and assign column names
data = pd.read_csv(
    "/Users/arnavshah/Desktop/rmpCapstoneNum.csv", 
    header=None,  # Indicate no header row in the CSV
    names=column_names  # Assign custom column names
)

# Step 1: Inspecting Data
print("Dataset Shape:", data.shape)
print("\nData Types:\n", data.dtypes)
print("\nMissing Values per Column:\n", data.isnull().sum())

# Step 2: Handling Missing Values
# Replace NaN in the "Proportion of Students Retaking" column with its median
column_name = "Proportion of Students Retaking"
if column_name in data.columns and data[column_name].isnull().sum() > 0:
    data[column_name].fillna(data[column_name].median(), inplace=True)

# Drop rows with missing values in all other columns
columns_to_check = [
    "Average Rating", "Average Difficulty", "Number of Ratings", 
    "Received a Pepper", "Number of Online Ratings", "Male", "Female"
]
data.dropna(subset=columns_to_check, inplace=True)

print("\nDataset Shape After Handling Missing Values:", data.shape)

# Step 3: Calculate 20% Quartile and Filter Dataset
# Calculate the 20th percentile for "Number of Ratings"

data = data[data["Number of Ratings"] >= 5]

print("\nDataset Shape After Filtering Data by 'Number of Ratings':", data.shape)

# Step 4: Summary Statistics
print("\nSummary Statistics for Numerical Columns:\n", data.describe())

# Step 5: Data Visualization
# Numerical columns
num_cols = [
    "Average Rating", "Average Difficulty", "Number of Ratings", 
    "Proportion of Students Retaking"
]

# Categorical columns
cat_cols = [
    "Received a Pepper", "Male", "Female"
]

# Univariate Plots: Histograms for numerical columns
for col in num_cols:
    plt.figure()
    sns.histplot(data[col], kde=True, bins=20, color='blue')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Bar charts for categorical variables
for col in cat_cols:
    plt.figure()
    sns.countplot(x=data[col])
    plt.title(f"Bar Chart of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# Step 6: Distribution Plots for Normality
for col in num_cols:
    plt.figure(figsize=(12, 6))
    
    # Histogram with KDE
    plt.subplot(1, 2, 1)
    sns.histplot(data[col], kde=True, bins=20, color='blue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    

# Step 7: Placeholder for Significance Testing
# Use alpha = 0.005 as the threshold for statistical significance
# Insert appropriate statistical tests here based on the questions being answered

# QUESTION 1: Gender Bias Analysis with Random Sampling

# Separate data into male and female groups
male_ratings = data[data["Male"] == 1]["Average Rating"]
female_ratings = data[data["Female"] == 1]["Average Rating"]

# Randomly sample 2500 data points from each group
male_ratings_sample = male_ratings.sample(n=2500, random_state=42)
female_ratings_sample = female_ratings.sample(n=2500, random_state=42)

# Perform the Mann-Whitney U Test
stat, p_value = mannwhitneyu(male_ratings_sample, female_ratings_sample, alternative='two-sided')

# Print the results
print(f"Mann-Whitney U Test Statistic: {stat:.4f}")
print(f"P-value: {p_value:.10f}")

# Interpretation of results
if p_value < 0.005:
    print("The difference in ratings between male and female professors is statistically significant.")
    if male_ratings_sample.median() > female_ratings_sample.median():
        print("Male professors have significantly higher ratings, suggesting possible pro-male bias.")
    else:
        print("Female professors have significantly higher ratings.")
else:
    print("There is no statistically significant difference in ratings between male and female professors.")

# Boxplot for visualization
plt.figure(figsize=(8, 6))
sns.boxplot(x=["Male"] * len(male_ratings_sample) + ["Female"] * len(female_ratings_sample),
            y=pd.concat([male_ratings_sample, female_ratings_sample]), 
            palette="pastel")
plt.title("Distribution of Average Ratings by Gender (Sampled Data)")
plt.xlabel("Gender")
plt.ylabel("Average Rating")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# QUESTION 2: Experience vs. Quality of Teaching

plt.figure(figsize=(8, 6))
sns.scatterplot(x=data["Number of Ratings"], y=data["Average Rating"])
plt.title("Scatterplot: Number of Ratings (Experience) vs. Average Rating (Quality)")
plt.xlabel("Number of Ratings (Experience)")
plt.ylabel("Average Rating (Quality)")
plt.show()

# Spearman's Correlation for Non-linear Data
spearman_corr, p_value = spearmanr(data["Number of Ratings"], data["Average Rating"])
print(f"Spearman's Correlation Coefficient: {spearman_corr:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.005:
    print("Statistically significant correlation between experience and quality.")
else:
    print("No statistically significant correlation between experience and quality.")


#QUESTION 3
plt.figure()
sns.scatterplot(x=data["Average Difficulty"], y=data["Average Rating"])
plt.title("Scatterplot: Average Difficulty vs. Average Rating")
plt.xlabel("Average Difficulty")
plt.ylabel("Average Rating")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Calculate Spearman's correlation coefficient
spearman_corr, p_value = spearmanr(data["Average Difficulty"], data["Average Rating"])

# Print results with high precision
print(f"Spearman's Correlation Coefficient: {spearman_corr:.10f}")
print(f"Exact P-value: {p_value:.10e}")  # Scientific notation for small p-values

# Interpret the result
if p_value < 0.005:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
    

#QUESTION 4

# Define online and non-online groups based on "Number of Online Ratings"
online_ratings = data[data["Number of Online Ratings"] > 0]["Average Rating"]
non_online_ratings = data[data["Number of Online Ratings"] == 0]["Average Rating"]

# Randomly sample 2500 observations from each group
online_sample = online_ratings.sample(n=2500, random_state=42)
non_online_sample = non_online_ratings.sample(n=2500, random_state=42)

# Visualize group sizes with a bar chart
group_counts = pd.Series([len(online_sample), len(non_online_sample)], index=["Online", "Non-Online"])
plt.figure(figsize=(8, 6))
sns.barplot(x=group_counts.index, y=group_counts.values, palette="pastel")
plt.title("Bar Chart of Online vs. Non-Online Professors (Sampled)", fontsize=14)
plt.xlabel("Group", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Check data distribution using histograms
plt.figure(figsize=(8, 6))
sns.histplot(online_sample, kde=True, color="blue", label="Online", bins=20, alpha=0.6)
sns.histplot(non_online_sample, kde=True, color="orange", label="Non-Online", bins=20, alpha=0.6)
plt.title("Distribution of Average Ratings for Online and Non-Online Professors (Sampled)", fontsize=14)
plt.xlabel("Average Rating", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.show()

# Perform Mann-Whitney U Test
stat, p_value_mw = mannwhitneyu(online_sample, non_online_sample, alternative="two-sided")

# Print Mann-Whitney U Test results
print(f"Mann-Whitney U Test Statistic: {stat:.4f}")
print(f"P-value: {p_value_mw:.10f}")

# Boxplot for visualization
plt.figure(figsize=(8, 6))
sns.boxplot(x=["Online"] * len(online_sample) + ["Non-Online"] * len(non_online_sample), 
            y=pd.concat([online_sample, non_online_sample]), 
            palette="pastel")
plt.title("Distribution of Average Ratings by Online Modality (Sampled)", fontsize=14)
plt.xlabel("Teaching Modality", fontsize=12)
plt.ylabel("Average Rating", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


# QUESTION 5: Scatterplot and Spearman Correlation

# Step 2: Scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=data["Proportion of Students Retaking"], 
    y=data["Average Rating"], 
    alpha=0.6
)
plt.title("Scatterplot: Proportion of Students Retaking vs. Average Rating", fontsize=14)
plt.xlabel("Proportion of Students Retaking (%)", fontsize=12)
plt.ylabel("Average Rating", fontsize=12)
plt.grid(axis="both", linestyle="--", alpha=0.7)
plt.show()

# Step 3: Spearman's Correlation
x = data["Proportion of Students Retaking"]
y = data["Average Rating"]

spearman_corr, p_value = spearmanr(x, y)

# Print Results
print(f"Spearman's Correlation Coefficient: {spearman_corr:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
if p_value < 0.005:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")

#QUESTION 6

# Step 1: Randomly sample 5000 data points
np.random.seed(42)  # For reproducibility
sampled_data = data.sample(n=5000, random_state=42)

# Step 2: Separate the sampled data into two groups based on "Received a Pepper"
pepper_ratings = sampled_data[sampled_data["Received a Pepper"] == 1]["Average Rating"]
no_pepper_ratings = sampled_data[sampled_data["Received a Pepper"] == 0]["Average Rating"]

# Step 3: Perform the Mann-Whitney U Test
stat, p_value = mannwhitneyu(pepper_ratings, no_pepper_ratings, alternative='two-sided')

# Step 4: Print the results
print(f"Mann-Whitney U Test Statistic: {stat:.4f}")
print(f"P-value: {p_value:.10f}")

# Step 5: Interpretation of results
if p_value < 0.005:
    print("The difference in ratings between professors who received a 'pepper' and those who didn't is statistically significant.")
else:
    print("There is no statistically significant difference in ratings between the two groups.")

# Step 6: Visualization with a Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=sampled_data, x="Received a Pepper", y="Average Rating", palette="pastel")
plt.title("Distribution of Average Ratings by Pepper Status (Sampled Data)", fontsize=14)
plt.xlabel("Received a Pepper", fontsize=12)
plt.ylabel("Average Rating", fontsize=12)
plt.xticks(ticks=[0, 1], labels=["No Pepper", "Pepper"], fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

### Question 7: Difficulty Only Model ###
# Step 3: Define feature (Average Difficulty) and target variable
X_difficulty = sm.add_constant(data[["Average Difficulty"]])  # Add constant for OLS
y_difficulty = data["Average Rating"]

# Step 4: Fit the OLS Model
difficulty_ols_model = sm.OLS(y_difficulty, X_difficulty).fit()

# Step 5: Summary and Evaluation
print("Question 7: Difficulty Only Model")
print(difficulty_ols_model.summary())

# Extract R2 and RMSE
r2_difficulty = difficulty_ols_model.rsquared
rmse_difficulty = (difficulty_ols_model.mse_resid ** 0.5)
slope_difficulty = difficulty_ols_model.params["Average Difficulty"]
intercept_difficulty = difficulty_ols_model.params["const"]

print(f"\nR-squared (R2): {r2_difficulty:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_difficulty:.4f}")
print(f"Slope (Coefficient): {slope_difficulty:.4f}")
print(f"Y-Intercept: {intercept_difficulty:.4f}")

# Step 6: Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data["Average Difficulty"], y=data["Average Rating"], label="Data")
sns.lineplot(x=data["Average Difficulty"], 
             y=difficulty_ols_model.predict(X_difficulty), color="red", label="Regression Line")
plt.title("Regression: Average Rating vs. Average Difficulty")
plt.xlabel("Average Difficulty")
plt.ylabel("Average Rating")
plt.legend()
plt.grid(axis="both", linestyle="--", alpha=0.7)
plt.show()


### Question 8: All Factors Model ###
# Step 3: Define features (All Factors) and target variable
X_all = sm.add_constant(data.drop(columns=["Average Rating"]))  # Add constant
y_all = data["Average Rating"]

# Step 4: Check for Multicollinearity (VIF)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_all.columns
vif_data["VIF"] = [variance_inflation_factor(X_all.values, i) for i in range(X_all.shape[1])]
print("\nVariance Inflation Factors (VIF) for All Factors Model:")
print(vif_data)

# Step 5: Fit the OLS Model
all_factors_ols_model = sm.OLS(y_all, X_all).fit()

# Step 6: Summary and Evaluation
print("\nQuestion 8: All Factors Model")
print(all_factors_ols_model.summary())

# Extract R2 and RMSE
r2_all = all_factors_ols_model.rsquared
rmse_all = (all_factors_ols_model.mse_resid ** 0.5)
intercept_all = all_factors_ols_model.params["const"]

print(f"\nR-squared (R2): {r2_all:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_all:.4f}")
print(f"Y-Intercept: {intercept_all:.4f}")

# Step 7: Coefficients
coefficients_all = all_factors_ols_model.params
print("\nRegression Coefficients for All Factors Model:")
print(coefficients_all)

# Step 8: Visualization of Predicted vs Actual
y_all_pred = all_factors_ols_model.predict(X_all)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_all, y=y_all_pred, alpha=0.6, label="Predicted vs Actual")
plt.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], 
         color="red", linestyle="--", label="Ideal Fit")
plt.title("Predicted vs Actual Average Rating (All Factors Model)")
plt.xlabel("Actual Average Rating")
plt.ylabel("Predicted Average Rating")
plt.legend()
plt.grid(axis="both", linestyle="--", alpha=0.7)
plt.show()

# Model Comparison
print("\nModel Comparison:")
print(f"Difficulty Only Model: R2 = {r2_difficulty:.4f}, RMSE = {rmse_difficulty:.4f}")
print(f"All Factors Model: R2 = {r2_all:.4f}, RMSE = {rmse_all:.4f}")

#QUESTION 9 

# Step 1: Prepare Data
X = data[["Average Rating"]].values  # Predictor: Average Rating
y = data["Received a Pepper"].values  # Target: Received a Pepper (binary)

# Step 2: Handle Class Imbalance with Custom Resampling
# Concatenate X and y for resampling
data_resampled = pd.concat([pd.DataFrame(X, columns=["Average Rating"]), pd.Series(y, name="Received a Pepper")], axis=1)

# Separate minority and majority classes
minority = data_resampled[data_resampled["Received a Pepper"] == 1]
majority = data_resampled[data_resampled["Received a Pepper"] == 0]

# Oversample minority class
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

# Combine majority class with upsampled minority class
data_balanced = pd.concat([majority, minority_upsampled])

# Separate X and y after resampling
X_resampled = data_balanced[["Average Rating"]].values
y_resampled = data_balanced["Received a Pepper"].values

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 4: Build and Train the Model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Step 6: Evaluate the Model
# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Step 7: Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label="Random Classifier")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend()
plt.grid()
plt.show()

# Step 8: Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Pepper", "Pepper"], yticklabels=["No Pepper", "Pepper"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

#QUESTION 10 

X_factors = data.drop(columns=["Received a Pepper"])  # Drop the target column
y_factors = data["Received a Pepper"]

# Address class imbalance by oversampling
data_factors_resampled = pd.concat([X_factors, y_factors], axis=1)
minority_factors = data_factors_resampled[data_factors_resampled["Received a Pepper"] == 1]
majority_factors = data_factors_resampled[data_factors_resampled["Received a Pepper"] == 0]

# Oversample the minority class
minority_factors_upsampled = resample(minority_factors, replace=True, n_samples=len(majority_factors), random_state=42)

# Combine oversampled minority class with majority class
data_factors_balanced = pd.concat([majority_factors, minority_factors_upsampled])

# Separate the predictors and target after resampling
X_factors_resampled = data_factors_balanced.drop(columns=["Received a Pepper"]).values
y_factors_resampled = data_factors_balanced["Received a Pepper"].values

# Split the resampled data into training and testing sets
X_factors_train, X_factors_test, y_factors_train, y_factors_test = train_test_split(
    X_factors_resampled, y_factors_resampled, test_size=0.2, random_state=42
)

# Build and train a Random Forest classifier
factors_model = RandomForestClassifier(random_state=42)
factors_model.fit(X_factors_train, y_factors_train)

# Make predictions
y_factors_pred = factors_model.predict(X_factors_test)
y_factors_pred_proba = factors_model.predict_proba(X_factors_test)[:, 1]

# Evaluate the model
# Classification report
print("Classification Report (All Factors Model):")
print(classification_report(y_factors_test, y_factors_pred))

# Confusion matrix
conf_matrix_factors = confusion_matrix(y_factors_test, y_factors_pred)
print("\nConfusion Matrix:")
print(conf_matrix_factors)

# ROC-AUC Score
roc_auc_factors = roc_auc_score(y_factors_test, y_factors_pred_proba)
print(f"\nROC-AUC Score (All Factors Model): {roc_auc_factors:.4f}")

# Plot ROC Curve
fpr_factors, tpr_factors, thresholds_factors = roc_curve(y_factors_test, y_factors_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr_factors, tpr_factors, color='blue', label=f"ROC Curve (AUC = {roc_auc_factors:.4f})")
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label="Random Classifier")
plt.title("ROC Curve (All Factors Model)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend()
plt.grid()
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_factors, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["No Pepper", "Pepper"], yticklabels=["No Pepper", "Pepper"])
plt.title("Confusion Matrix (All Factors Model)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Compare with "Average Rating Only" model
print("\nModel Comparison:")
print(f"Average Rating Only Model: ROC-AUC = {roc_auc:.4f}")
print(f"All Factors Model: ROC-AUC = {roc_auc_factors:.4f}")

#EXTRA CREDIT

# Load the qualitative data
qual_data = pd.read_csv("/Users/arnavshah/Desktop/rmpCapstoneQual.csv", header=None, names=["Field", "University", "State"])

# Merge the quantitative and qualitative datasets
data = pd.concat([data.reset_index(drop=True), qual_data.reset_index(drop=True)], axis=1)

# Step 9: Aggregate Fields for Better Visualization
field_mapping = {
    "Biology": "STEM",
    "Physics": "STEM",
    "Chemistry": "STEM",
    "Mathematics": "STEM",
    "Fine Arts": "Arts",
    "English": "Humanities",
    "History": "Humanities",
    "Political Science": "Social Sciences",
    "Economics": "Social Sciences",
    "Business": "Professional Studies",
    "Education": "Professional Studies",
    "Nursing": "Health Sciences",
    "Psychology": "Social Sciences",
    # Add more mappings based on your dataset as needed
}

data["Field Group"] = data["Field"].map(field_mapping).fillna("Other")

# Step 10: Plot Average Ratings by Aggregated Field Groups
plt.figure(figsize=(14, 8))
sns.boxplot(x="Field Group", y="Average Rating", data=data, palette="pastel")
plt.title("Average Rating by Academic Field Group", fontsize=16)
plt.xlabel("Academic Field Group", fontsize=12)
plt.ylabel("Average Rating", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Step 11: Plot Average Ratings for Top 10 Fields by Sample Size
top_fields = data["Field"].value_counts().nlargest(10).index
data_top_fields = data[data["Field"].isin(top_fields)]

plt.figure(figsize=(14, 8))
sns.boxplot(x="Field", y="Average Rating", data=data_top_fields, palette="pastel", order=top_fields)
plt.title("Average Rating by Top 10 Academic Fields", fontsize=16)
plt.xlabel("Academic Field", fontsize=12)
plt.ylabel("Average Rating", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Step 12: Plot All Fields Sorted by Median Rating
field_medians = data.groupby("Field")["Average Rating"].median().sort_values()
sorted_fields = field_medians.index

# Calculate Descriptive Statistics for Top 10 Academic Fields
top_fields_stats = data_top_fields.groupby("Field")["Average Rating"].describe()
print("Descriptive Statistics for Top 10 Academic Fields:")
print(top_fields_stats)

# Calculate Descriptive Statistics for Academic Field Groups
field_groups_stats = data.groupby("Field Group")["Average Rating"].describe()
print("\nDescriptive Statistics for Academic Field Groups:")
print(field_groups_stats)

# Calculate Median Ratings for Top 10 Fields and Field Groups
top_fields_median = data_top_fields.groupby("Field")["Average Rating"].median()
field_groups_median = data.groupby("Field Group")["Average Rating"].median()

print("\nMedian Ratings for Top 10 Fields:")
print(top_fields_median)

print("\nMedian Ratings for Academic Field Groups:")
print(field_groups_median)

# Visualization: Median Ratings
plt.figure(figsize=(10, 6))
top_fields_median.sort_values().plot(kind="bar", color="skyblue", alpha=0.8)
plt.title("Median Ratings for Top 10 Academic Fields", fontsize=14)
plt.xlabel("Academic Field")
plt.ylabel("Median Rating")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

plt.figure(figsize=(10, 6))
field_groups_median.sort_values().plot(kind="bar", color="salmon", alpha=0.8)
plt.title("Median Ratings for Academic Field Groups", fontsize=14)
plt.xlabel("Academic Field Group")
plt.ylabel("Median Rating")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


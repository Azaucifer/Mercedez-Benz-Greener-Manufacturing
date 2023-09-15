# Mercedes-Benz Greener Manufacturing

## Description

### Problem Statement Scenario

Mercedes-Benz, known for its legacy of automotive innovation, applies for nearly 2000 patents per year, making it a leader in the premium car industry. Offering a wide range of features and options, customers can personalize their dream Mercedes-Benz vehicle. To ensure the safety and reliability of each unique car configuration, the company has developed a robust testing system.

As a major manufacturer of premium cars, Mercedes-Benz places a strong emphasis on safety and efficiency in its production lines. Optimizing the testing system's speed for numerous feature combinations is a complex and time-consuming task without a powerful algorithmic approach.

The challenge is to reduce the time cars spend on the test bench, ultimately contributing to faster testing, lower carbon dioxide emissions, and maintaining Mercedes-Benz's high standards.

### Actions to Be Performed

To address this challenge, the following actions are to be performed:

1. Remove columns with zero variance.
2. Check for null and unique values in both test and train datasets.
3. Apply label encoding to handle categorical data.
4. Perform dimensionality reduction.
5. Predict test values using XGBoost.

### Steps Involved

Here's a breakdown of the steps involved in this project:

**Step 1: Import Required Libraries**
- Import necessary libraries for data processing and dimensionality reduction.

**Step 2: Read the Data from train.csv**
- Understand and inspect the data by printing a few rows.

**Step 3: Collect the Target Values (Y)**
- Separate the target values (Y) from the data for prediction.

**Step 4: Understand Data Types**
- Iterate through columns with "X" in the name to understand data types.

**Step 5: Count Data in Columns**

**Step 6: Read the test.csv Data**
- Remove unused columns (ID and Y).

**Step 7: Check for Null and Unique Values**

**Step 8: Remove Columns with Zero Variance**

**Step 9: Apply Label Encoder**
- Convert categorical data into numerical format.

**Step 10: Ensure Data is Numerical**
- Confirm that the data is now in numerical format.

**Step 11: Perform Dimensionality Reduction**
- Utilize Singular Value Decomposition (SVD) to project data into a lower-dimensional space.

**Step 12: Training Using XGBoost**
- Train the model using XGBoost.

**Step 13: Predict Test Values Using XGBoost**

### Conclusion

The analysis revealed a larger Root Mean Squared Error (RMSE) for the testing set, indicating that the initial model did not perform well. To improve model performance, it is recommended to use Cross Validation with XGBoost to identify features that yield a better training RMSE. The model with better performance on the training data can then be used for prediction.

This project aims to optimize Mercedes-Benz's testing system, ultimately contributing to greener manufacturing practices and maintaining the brand's high standards of safety and efficiency.

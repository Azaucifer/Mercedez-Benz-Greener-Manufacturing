# Mercedez-Benz-Greener-Manufacturing

DESCRIPTION

Reduce the time a Mercedes-Benz spends on the test bench.

Problem Statement Scenario:
Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. These include the passenger safety cell with a crumple zone, the airbag, and intelligent assistance systems. Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium carmakers. Mercedes-Benz is the leader in the premium car industry. With a huge selection of features and options, customers can choose the customized Mercedes-Benz of their dreams.

To ensure the safety and reliability of every unique car configuration before they hit the road, the company’s engineers have developed a robust testing system. As one of the world’s biggest manufacturers of premium cars, safety and efficiency are paramount on Mercedes-Benz’s production lines. However, optimizing the speed of their testing system for many possible feature combinations is complex and time-consuming without a powerful algorithmic approach.

You are required to reduce the time that cars spend on the test bench. Others will work with a dataset representing different permutations of features in a Mercedes-Benz car to predict the time it takes to pass testing. Optimal algorithms will contribute to faster testing, resulting in lower carbon dioxide emissions without reducing Mercedes-Benz’s standards.

Following actions should be performed:

If for any column(s), the variance is equal to zero, then you need to remove those variable(s).
Check for null and unique values for test and train sets.
Apply label encoder.
Perform dimensionality reduction.
Predict your test_df values using XGBoost.

Step1: Import the required libraries
1.1: linear algebra

1.2: data processing

1.3: for dimensionality reduction

Step2: Read the data from train.csv
2.1: let us understand the data

2.2: print few rows and see how the data looks like

Step3: Collect the Y values into an array
3.1: seperate the y from the data as we will use this to learn as the prediction output

Step4: Understand the data types we have
4.1:iterate through all the columns which has X in the name of the column

Step5: Count the data in each of the columns

Step6: Read the test.csv data
6.1: remove columns ID and Y from the data as they are not used for learning

Step7: Check for null and unique values for test and train sets

Step8: If for any column(s), the variance is equal to zero, then you need to remove those variable(s).
8.1: Apply label encoder

Step9: Make sure the data is now changed into numericals

Step10: Perform dimensionality reduction
10.1: Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.

Step11: Training using xgboost

Step12: Predict your test_df values using xgboost

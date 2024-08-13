import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define file paths
data_description_path = r'C:\Users\Yashraj\Downloads\house-prices-advanced-regression-techniques\data_description.txt'
sample_submission_path = r'C:\Users\Yashraj\Downloads\house-prices-advanced-regression-techniques\sample_submission.csv'
test_path = r'C:\Users\Yashraj\Downloads\house-prices-advanced-regression-techniques\test.csv'
train_path = r'C:\Users\Yashraj\Downloads\house-prices-advanced-regression-techniques\train.csv'

# Check if the train file exists
if os.path.exists(train_path):
    # Load the dataset from the CSV file
    data = pd.read_csv(train_path)

    # Preview the dataset
    print("Preview of the dataset:")
    print(data.head())

    # Print the column names to check available features
    print("Columns in the dataset:", data.columns.tolist())

    # Check for missing values
    print("Missing values in each column:")
    print(data.isnull().sum())

    # Fill or drop missing values if necessary
    data.fillna(0, inplace=True)  # Example: filling missing values with 0

    # Define features and target variable
    # Update these column names based on the printed column names from the dataset
    X = data[['LotArea', 'OverallQual', 'OverallCond']]  # Adjust these column names based on actual data
    y = data['SalePrice']  # Adjust this if the target variable has a different name

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    # Plotting predictions vs actual prices
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')  # Diagonal line
    plt.show()
else:
    print("File not found at:", train_path)

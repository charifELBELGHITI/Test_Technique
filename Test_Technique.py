#import libraries and modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import shutil

def load_data(file_path):
    """
    Load the dataset from the specified file path.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Perform data preprocessing including removing duplicates, unnecessary columns,
    and checking for null values.
    """
    # Remove 'Unnamed' column
    df.drop(columns=['Unnamed: 0'], inplace=True)

    # Check for null values
    null_values = df.isnull().sum()
    print("Null Values in Each Column:\n", null_values)

    # Removing duplicate rows
    df = df.drop_duplicates()

    return df


def summarize_data(df):
    """
    Compute summary statistics for numerical and categorical features.
    """
    print(df.describe())
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(df[col].value_counts())

def visualize_univariate_data(df):
    """
    Create histograms and boxplots for numerical features and bar plots for categorical features.
    """
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numerical_cols:
        plt.figure()
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f'Histogram of {col}')
        plt.show()

        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

    for col in categorical_cols:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=df[col])
        plt.title(f'Frequency of {col}')
        plt.xticks(rotation=90)
        plt.show()

def analyze_correlation(df, threshold=0.6):
    """
    Calculate and visualize the correlation matrix for numerical features,
    considering only correlations above a specified threshold.
    """
    # Select only numerical columns for correlation analysis
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numerical_cols].corr()

    # Apply threshold to filter out lower correlations
    # We only keep absolute correlations that are above the threshold
    filtered_corr = correlation_matrix[abs(correlation_matrix) >= threshold]
    
    # Create a mask to hide the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(filtered_corr, dtype=bool))

    # Visualize the filtered correlation matrix
    plt.figure(figsize=(20, 15))
    sns.heatmap(filtered_corr, mask=mask, cmap='coolwarm', vmax=1.0, vmin=-1.0, annot=True, fmt=".2f")
    plt.title(f'Filtered Correlation Matrix (Threshold: ±{threshold})')
    plt.show()



def encode_categorical_features(df, categorical_cols):
    """
    Apply One-Hot Encoding and Label Encoding to categorical features.
    """
    # Applying One-Hot Encoding to categorical columns
    one_hot_encoder = OneHotEncoder(drop='first')  # Removed the 'sparse' parameter
    one_hot_encoded = one_hot_encoder.fit_transform(df[categorical_cols])

    # Convert the one-hot encoded sparse matrix to a DataFrame
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded.toarray(), 
                                      columns=one_hot_encoder.get_feature_names_out(categorical_cols))

    # Applying Label Encoding to categorical columns
    label_encoded_df = df[categorical_cols].apply(LabelEncoder().fit_transform)

    # Concatenating the one-hot encoded and label encoded columns to the original DataFrame
    encoded_df = pd.concat([df.drop(categorical_cols, axis=1), 
                            one_hot_encoded_df, label_encoded_df], axis=1)

    return encoded_df

def save_and_move_file(df, destination_folder, filename='processed_iot_data.csv'):
    """
    Save the processed DataFrame to a CSV file and move it to a specified folder.
    """
    processed_file_path = df.to_csv(filename, index=False)
    shutil.move(processed_file_path, destination_folder + filename)
    print(f"File moved to {destination_folder}")

def main():
    file_path = "RT_IOT2022.csv"
    df = load_data(file_path)
    df = preprocess_data(df)
    summarize_data(df)
    visualize_univariate_data(df)
    analyze_correlation(df)
    encoded_df = encode_categorical_features(df, df.select_dtypes(include=['object']).columns)
    save_and_move_file(encoded_df, '/Documents/Python Scripts')

if __name__ == '__main__':
    main()
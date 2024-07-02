import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def condense_upper_triangular(corr_matrix):
    """function to condense the upper triangular part of a correlation matrix into a DataFrame"""
    rows, cols = corr_matrix.shape
    condensed_list = []
    
    for i in range(rows):
        for j in range(i+1, cols):
            if not np.isnan(corr_matrix.iloc[i, j]):
                condensed_list.append((corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    condensed_df = pd.DataFrame(condensed_list, columns=['Variable 1', 'Variable 2', 'Correlation'])
    return condensed_df

def find_highest_correlations(corr_matrix):
    """function to find the highest correlation in a correlation matrix, with respect to each column"""
    for column in corr_matrix.columns:
        # Exclude the diagonal element by replacing it with NaN
        corr_column = corr_matrix[column].copy()
        corr_column[column] = None
        max_corr = corr_column.max()
        min_corr = corr_column.min()
        if abs(max_corr) > abs(min_corr):
            highest_corr_value = max_corr
        else:
            highest_corr_value = min_corr

        highest_corr_variable = abs(corr_column).idxmax()
        print(f"Column '{column}' has highest correlation with '{highest_corr_variable}' : {highest_corr_value:.2f}")

def plot_decision_boundary(X, y, model, feature1, feature2):
    """Function to plot the decision boundary of a model."""
    plt.figure(figsize=(10, 6))

    # For title of the plot
    model_type = ""
    if model.__class__.__name__ == 'MultinomialResultsWrapper':
        model_type = "Multinomial Logistic Regression"
    elif model.__class__.__name__ == 'BinaryResultsWrapper':
        model_type = "Binary Logistic Regression"
    elif model.__class__.__name__ == 'LinearDiscriminantAnalysis':
        model_type = "Linear Discriminant Analysis"
    elif model.__class__.__name__ == 'GaussianNB':
        model_type = "Naive Bayes"
    elif model.__class__.__class__.__name__ == 'KNeighborsClassifier':
        model_type = "K-Nearest Neighbors"
    
    # Select two features for plotting
    X_selected = X[[feature1, feature2]]
    
    sns.scatterplot(x=X_selected[feature1], y=X_selected[feature2], hue=y, palette='viridis')
    
    x_min, x_max = X_selected[feature1].min() - 1, X_selected[feature1].max() + 1
    y_min, y_max = X_selected[feature2].min() - 1, X_selected[feature2].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Prepare the grid for prediction
    if (model.__class__.__name__ == 'MultinomialResultsWrapper'):
        grid_full = np.zeros((grid.shape[0], X.shape[1] + 1))  # +1 for the constant term
        grid_full[:, 1 + X.columns.get_loc(feature1)] = grid[:, 0]
        grid_full[:, 1 + X.columns.get_loc(feature2)] = grid[:, 1]
        grid_full[:, 0] = 1  # constant term

    else:
        grid_full = np.zeros((grid.shape[0], X.shape[1]))
        grid_full[:, X.columns.get_loc(feature1)] = grid[:, 0]
        grid_full[:, X.columns.get_loc(feature2)] = grid[:, 1]
    
    # Set other features to their mean values
    for col in X.columns:
        if col != feature1 and col != feature2:
            grid_full[:, X.columns.get_loc(col)] = X[col].mean()
    
    # If it's Logistic Regression, handle the prediction differently
    if model.__class__.__name__ == 'MultinomialResultsWrapper':
        # Predict class probabilities
        Z = model.predict(grid_full)
        # Find the class with the highest probability
        Z = np.argmax(Z, axis=1)
    else:
        # Convert grid_full back to DataFrame to silence warnings
        grid_full_df = pd.DataFrame(grid_full, columns=X.columns)
        Z = model.predict(grid_full_df)
    
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'Decision Boundary of {model_type}')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)


# -------------------------------------------------
# Question 1: Dataset generation and visualization
# -------------------------------------------------

def generate_clean_data(
    n_samples=500,
    noise=20,
    random_state=42
):
    """
    Generate a clean synthetic regression dataset.

    Return:
        X, y, true_coef

    Requirements:
    - n_samples = 500 by default
    - n_features = 1
    - n_informative = 1
    - noise = 20
    - coef = True
    - random_state = 42
    """
    X, y, coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=noise,
        coef=True,
        random_state=random_state
    )
    return X, y, coef


def add_outliers(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Add artificial outliers to the first n_outliers observations.

    Use:
        X[:n_outliers] = 10 + 0.75 * random_normal_values
        y[:n_outliers] = -15 + 20 * random_normal_values

    Return:
        X_out, y_out

    Important:
    Do not modify the original X and y directly.
    Make copies first.
    """
    X_out = X.copy()
    y_out = y.copy()

    np.random.seed(random_state)
    X_out[:n_outliers] = 10 + 0.75 * np.random.normal(size=(n_outliers, 1))
    y_out[:n_outliers] = -15 + 20 * np.random.normal(size=n_outliers)

    return X_out, y_out


def plot_dataset_with_outliers(
    X,
    y,
    n_outliers=25
):
    """
    Plot the dataset and highlight the first n_outliers observations.

    Return:
        matplotlib Figure object
      
    Requirements:
    - normal observations and artificial outliers should be visually different
    - include title
    - include x-label
    - include y-label
    - include legend
    """
    fig, ax = plt.subplots()

    # Normal observations (everything after n_outliers)
    ax.scatter(
        X[n_outliers:], y[n_outliers:],
        color='steelblue', label='Normal observations', alpha=0.7
    )

    # Artificial outliers (first n_outliers)
    ax.scatter(
        X[:n_outliers], y[:n_outliers],
        color='red', marker='x', s=80, label='Artificial outliers'
    )

    ax.set_title('Dataset with Artificial Outliers')
    ax.set_xlabel('Feature X')
    ax.set_ylabel('Target y')
    ax.legend()

    return fig


# -------------------------------------------------
# Question 2: Fit regression models
# -------------------------------------------------

def fit_linear_regression(X, y):
    """
    Fit ordinary Linear Regression.

    Return:
        fitted coefficient as a float
    """
    lr = LinearRegression().fit(X, y)
    return float(lr.coef_[0])


def fit_huber_regression(X, y):
    """
    Fit Huber Regression.

    Return:
        fitted coefficient as a float
    """
    huber = HuberRegressor().fit(X, y)
    return float(huber.coef_[0])


def fit_ransac_regression(X, y, random_state=42):
    """
    Fit RANSAC Regression.

    Return:
        fitted coefficient as a float

    Hint:
    RANSAC stores the final linear model in estimator_.
    """
    ransac = RANSACRegressor(random_state=random_state).fit(X, y)
    return float(ransac.estimator_.coef_[0])


def fit_theilsen_regression(X, y, random_state=42):
    """
    Fit Theil-Sen Regression.

    Return:
        fitted coefficient as a float
    """
    theilsen = TheilSenRegressor(random_state=random_state).fit(X, y)
    return float(theilsen.coef_[0])


def coefficient_errors(coef_dict, true_coef):
    """
    Given a dictionary of coefficients and the true coefficient,
    return a dictionary of absolute coefficient errors.

    Example input:
        {
            "linear_regression": 8.7,
            "huber_regression": 37.5,
            "ransac_regression": 62.8,
            "theilsen_regression": 59.4
        }

    Return:
        {
            "linear_regression": abs(...),
            ...
        }
    """
    return {model: abs(coef - true_coef) for model, coef in coef_dict.items()}


def best_robust_model(errors):
    """
    Return the name of the robust model with the smallest error.

    Only compare:
        huber_regression
        ransac_regression
        theilsen_regression
    """
    robust_models = {
        k: v for k, v in errors.items()
        if k in ('huber_regression', 'ransac_regression', 'theilsen_regression')
    }
    return min(robust_models, key=robust_models.get)


def ransac_outlier_summary(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Fit RANSAC and return:
        total_outliers_detected, added_outliers_detected
    """
    ransac = RANSACRegressor(random_state=random_state).fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = ~inlier_mask

    total_outliers_detected = int(sum(outlier_mask))
    added_outliers_detected = int(sum(outlier_mask[:n_outliers]))

    return total_outliers_detected, added_outliers_detected


# -------------------------------------------------
# Question 2: Visualization functions
# -------------------------------------------------

def plot_regression_fits(
    X,
    y,
    random_state=42
):
    """
    Plot fitted regression lines for all four models.

    Return:
        matplotlib Figure object
    """
    # Fit all models
    lr = LinearRegression().fit(X, y)
    huber = HuberRegressor().fit(X, y)
    ransac = RANSACRegressor(random_state=random_state).fit(X, y)
    theilsen = TheilSenRegressor(random_state=random_state).fit(X, y)

    # X range for plotting lines
    plotline_X = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    fig, ax = plt.subplots()

    # Scatter data
    ax.scatter(X, y, color='black', alpha=0.4, s=20, label='Data')

    # Fitted lines
    ax.plot(plotline_X, lr.predict(plotline_X),
            color='blue', linewidth=2, label='Linear Regression')
    ax.plot(plotline_X, huber.predict(plotline_X),
            color='orange', linewidth=2, label='Huber Regression')
    ax.plot(plotline_X, ransac.predict(plotline_X),
            color='green', linewidth=2, label='RANSAC Regression')
    ax.plot(plotline_X, theilsen.predict(plotline_X),
            color='red', linewidth=2, label='Theil-Sen Regression')

    ax.set_title('Regression Model Fits on Data with Outliers')
    ax.set_xlabel('Feature X')
    ax.set_ylabel('Target y')
    ax.legend()

    return fig


def plot_ransac_inliers_outliers(
    X,
    y,
    random_state=42
):
    """
    Fit RANSAC and visualize inliers vs outliers.

    Return:
        matplotlib Figure object
    """
    ransac = RANSACRegressor(random_state=random_state).fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = ~inlier_mask

    fig, ax = plt.subplots()

    ax.scatter(X[inlier_mask], y[inlier_mask],
               color='blue', alpha=0.6, label='Inliers')
    ax.scatter(X[outlier_mask], y[outlier_mask],
               color='red', marker='x', s=80, label='Outliers')

    ax.set_title('RANSAC: Inliers vs Outliers')
    ax.set_xlabel('Feature X')
    ax.set_ylabel('Target y')
    ax.legend()

    return fig


# svr_and_pcr.py
# Uses PCA pipeline from pca_model.py and replicates original plotting style

import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from gui.statistics.pca_model import MLDataProcessor

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


def load_pca_pipeline(pkl_path):
    """
    Load the StandardScaler and PCA objects saved in ml_data_processor.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['scaler'], data['pca']


def plot_regression_results(x_test, y_test, y_pred_test, title, filename):
    """
    Plot the regression results exactly as in the original script.
    """
    x_dummy = np.linspace(1, x_test.shape[0], x_test.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle(title, fontsize=20)
    plt.subplots_adjust(top=0.95, bottom=0.01, left=0.0, right=1.0, hspace=0.01, wspace=0.0)

    axes[0].scatter(x_dummy, (y_pred_test - y_test) / y_test * 100, marker="o", color="white", edgecolor="darkgreen")
    axes[0].set_ylabel(r"$percentage$ $error$ $[\%]$", fontsize=16)
    axes[0].set_xticks([])

    axes[1].scatter(x_dummy, y_test, color="darkblue", label="Real data")
    axes[1].scatter(x_dummy, y_pred_test, color="crimson", label="Estimated data")
    axes[1].set_xlabel(r"$Index$", fontsize=16)
    axes[1].set_ylabel(r"$IRCRA$", fontsize=16)
    axes[1].legend(fontsize=16)

    # Optional - save the figure
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 1) Fetch and merge data
    proc = MLDataProcessor(
        climber_db='/Users/annasebestikova/PycharmProjects/Diploma/db_testing/databases/climber_database.db',
        tests_db='/Users/annasebestikova/PycharmProjects/Diploma/db_testing/databases/tests_database.db',
        admin_id=1
    )
    df = proc.load_data(test_type='ao')
    X_raw, y = proc.prepare_features(df)

    # 2) Load PCA pipeline
    scaler, pca = load_pca_pipeline(os.path.join(MODEL_DIR, 'pca_model.pkl'))

    # 3) Transform features
    X_scaled = scaler.transform(X_raw)
    X_pcs = pca.transform(X_scaled)

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pcs, y, test_size=0.3, random_state=42
    )

    # 5) SVR
    svr = svm.SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)
    # Save SVR model
    with open(os.path.join(MODEL_DIR, 'svr_model.pkl'), 'wb') as f:
        pickle.dump(svr, f)

    # 6) Plot SVR results
    plot_regression_results(
        X_test, y_test, y_pred_svr,
        "Prediction of IRCRA level using SVR",
        "svr_prediction.png"
    )
    plt.scatter(y_test, y_pred_svr, alpha=0.7, color="darkblue")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'k--', lw=2
    )
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('IRCRA prediction with SVR')
    plt.savefig('svr_sel_scatter.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

    # 7) PCR (Linear Regression on PCs)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred_lin = linreg.predict(X_test)
    # Save PCR model
    with open(os.path.join(MODEL_DIR, 'pcr_model.pkl'), 'wb') as f:
        pickle.dump(linreg, f)

    # 8) Plot PCR results
    plot_regression_results(
        X_test, y_test, y_pred_lin,
        "Prediction of IRCRA level using PCR",
        "pcr_prediction.png"
    )
    plt.scatter(y_test, y_pred_lin, alpha=0.7, color="darkblue")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'k--', lw=2
    )
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('IRCRA prediction with PCR')
    plt.savefig('linreg_sel_scatter.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

    # 9) Print errors
    print(
        f"Linear regression error: {mean_squared_error(y_test, y_pred_lin):.3f}, "
        f"SVR error:           {mean_squared_error(y_test, y_pred_svr):.3f}"
    )


if __name__ == '__main__':
    main()

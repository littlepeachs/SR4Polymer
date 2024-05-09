from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np

class LCEN:
    def __init__(self, degree_range, alpha_range, l1_ratio_range, cutoff, max_iter=10000):
        self.degree_range = degree_range
        self.alpha_range = alpha_range
        self.l1_ratio_range = l1_ratio_range
        self.cutoff = cutoff
        self.max_iter = max_iter

    def fit(self, X, y):
        # LASSO step
        best_degree = None
        best_lasso_coefs = None
        best_lasso_score = float('inf')

        for degree in range(*self.degree_range):
            lasso_model = LassoCV(alphas=self.alpha_range, max_iter=self.max_iter, cv=5)
            lasso_pipe = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                                   ('lasso', lasso_model)])
            lasso_pipe.fit(X, y)
            lasso_score = -lasso_pipe.score(X, y)  # Negate score to find the minimum
            if lasso_score < best_lasso_score:
                best_degree = degree
                best_lasso_coefs = lasso_pipe.named_steps['lasso'].coef_
                best_lasso_score = lasso_score

        # Clip step
        best_lasso_coefs[np.abs(best_lasso_coefs) < self.cutoff] = 0
        import pdb; pdb.set_trace()
        print(best_lasso_coefs)
        # EN step
        en_model = ElasticNetCV(alphas=self.alpha_range, l1_ratio=self.l1_ratio_range,
                                 max_iter=self.max_iter, cv=5)
        en_pipe = Pipeline([('poly', PolynomialFeatures(degree=best_degree)),
                             ('en', en_model)])
        en_pipe.fit(X, y)
        en_coefs = en_pipe.named_steps['en'].coef_

        # Clip step II
        en_coefs[np.abs(en_coefs) < self.cutoff] = 0


        # Get the feature names from the PolynomialFeatures transformer
        poly_transformer = en_pipe.named_steps['poly']
        self.feature_names = poly_transformer.get_feature_names_out(input_features=[f"x{i}" for i in range(X.shape[1])])

        self.coef_ = en_coefs
        self.degree_ = best_degree
        formula="y="
        for coef, feature in zip(lcen.coef_, self.feature_names):
            if coef != 0:
                formula += f"{coef:.3f} * {feature} + "
        print(formula)
        return self

    def predict(self, X):
        poly = PolynomialFeatures(degree=self.degree_)
        X_poly = poly.fit_transform(X)
        return X_poly @ self.coef_

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(0)
X = np.random.uniform(2, 100, size=(100, 2))
y = 3 * X[:, 1] + 2 * X[:, 0] ** 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the LCEN model
lcen = LCEN(degree_range=(1, 3), alpha_range=np.logspace(-4, 1, 10),
            l1_ratio_range=[0.1, 0.5, 0.9], cutoff=0.01)
lcen.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lcen.predict(X_test)

# Calculate MSE and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R-squared (R²): {r2:.3f}")
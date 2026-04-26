"""Machine learning methods for statistical analysis.

Provides methods for:
- Regularized regression (LASSO, Ridge, Elastic Net)
- Random Forest variable importance
- Gradient Boosting (XGBoost-style)
- Cross-validation and model selection
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import warnings


@dataclass
class RegularizedRegressionResult:
    """Result of regularized regression analysis."""
    model_type: str
    n_observations: int
    n_features: int
    n_selected_features: int = 0

    # Regularization
    optimal_alpha: Optional[float] = None
    l1_ratio: Optional[float] = None  # For elastic net

    # Coefficients
    coefficients: Dict[str, float] = field(default_factory=dict)
    selected_features: List[str] = field(default_factory=list)

    # Model fit
    r_squared: Optional[float] = None
    mse: Optional[float] = None
    cv_score: Optional[float] = None
    cv_std: Optional[float] = None

    # For classification
    auc: Optional[float] = None
    accuracy: Optional[float] = None

    # Interpretation
    summary_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class RandomForestResult:
    """Result of Random Forest analysis."""
    model_type: str
    task_type: str  # "classification" or "regression"
    n_observations: int
    n_features: int
    n_trees: int

    # Variable importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    top_features: List[str] = field(default_factory=list)

    # Model performance
    oob_score: Optional[float] = None
    cv_score: Optional[float] = None
    cv_std: Optional[float] = None

    # Classification metrics
    auc: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None

    # Regression metrics
    r_squared: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None

    # Interpretation
    summary_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class GradientBoostingResult:
    """Result of Gradient Boosting analysis."""
    model_type: str
    task_type: str
    n_observations: int
    n_features: int
    n_estimators: int
    learning_rate: float
    max_depth: int

    # Variable importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    top_features: List[str] = field(default_factory=list)

    # Model performance
    cv_score: Optional[float] = None
    cv_std: Optional[float] = None

    # Classification metrics
    auc: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None

    # Regression metrics
    r_squared: Optional[float] = None
    mse: Optional[float] = None

    # Training history
    train_scores: Optional[List[float]] = None
    validation_scores: Optional[List[float]] = None

    # Interpretation
    summary_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class MachineLearning:
    """Machine learning methods for statistical analysis."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def lasso_regression(
        self,
        data: pd.DataFrame,
        outcome: str,
        predictors: List[str],
        cv_folds: int = 5,
        alphas: Optional[List[float]] = None
    ) -> RegularizedRegressionResult:
        """LASSO (L1) regularized regression for variable selection.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            predictors: List of predictor variable names
            cv_folds: Number of cross-validation folds
            alphas: List of alpha values to try (None for automatic)

        Returns:
            RegularizedRegressionResult object
        """
        # Prepare data
        all_vars = [outcome] + predictors
        clean_data = data[all_vars].dropna()

        if len(clean_data) < cv_folds * 2:
            return self._insufficient_data("LASSO Regression")

        X = clean_data[predictors].values
        y = clean_data[outcome].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            # Fit LASSO with cross-validation
            if alphas is None:
                alphas = np.logspace(-4, 2, 100)

            model = LassoCV(
                alphas=alphas,
                cv=cv_folds,
                random_state=self.random_state,
                max_iter=10000
            )
            model.fit(X_scaled, y)

            # Get coefficients (transform back to original scale)
            coefficients = {}
            for i, var in enumerate(predictors):
                coef = model.coef_[i] / scaler.scale_[i]  # Unstandardize
                coefficients[var] = float(coef)

            # Selected features (non-zero coefficients)
            selected = [p for p, c in coefficients.items() if abs(c) > 1e-10]

            # Model performance
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            # Cross-validation scores
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')

            # Generate summary
            summary = self._generate_lasso_summary(
                selected, model.alpha_, r2, cv_scores.mean()
            )

            # Educational notes
            edu_notes = {
                'what_is_lasso': 'LASSO adds an L1 penalty that shrinks some coefficients to exactly zero, performing automatic variable selection.',
                'interpreting_alpha': f'Optimal alpha={model.alpha_:.4f}. Larger alpha = more regularization = fewer selected variables.',
                'selected_variables': f'{len(selected)} of {len(predictors)} variables selected as important predictors.',
                'vs_ridge': 'Unlike Ridge regression, LASSO performs feature selection by setting some coefficients to exactly zero.',
                'when_to_use': 'Use LASSO when you have many predictors and want to identify the most important subset.',
                'limitations': 'LASSO tends to select only one variable from groups of correlated predictors.'
            }

            return RegularizedRegressionResult(
                model_type="LASSO Regression (L1)",
                n_observations=len(clean_data),
                n_features=len(predictors),
                n_selected_features=len(selected),
                optimal_alpha=float(model.alpha_),
                coefficients=coefficients,
                selected_features=selected,
                r_squared=float(r2),
                mse=float(mse),
                cv_score=float(cv_scores.mean()),
                cv_std=float(cv_scores.std()),
                summary_text=summary,
                educational_notes=edu_notes
            )

        except Exception as e:
            result = self._insufficient_data("LASSO Regression")
            result.warnings = [str(e)]
            return result

    def ridge_regression(
        self,
        data: pd.DataFrame,
        outcome: str,
        predictors: List[str],
        cv_folds: int = 5,
        alphas: Optional[List[float]] = None
    ) -> RegularizedRegressionResult:
        """Ridge (L2) regularized regression.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            predictors: List of predictor variable names
            cv_folds: Number of cross-validation folds
            alphas: List of alpha values to try

        Returns:
            RegularizedRegressionResult object
        """
        # Prepare data
        all_vars = [outcome] + predictors
        clean_data = data[all_vars].dropna()

        if len(clean_data) < cv_folds * 2:
            return self._insufficient_data("Ridge Regression")

        X = clean_data[predictors].values
        y = clean_data[outcome].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            if alphas is None:
                alphas = np.logspace(-4, 4, 100)

            model = RidgeCV(alphas=alphas, cv=cv_folds)
            model.fit(X_scaled, y)

            # Get coefficients
            coefficients = {}
            for i, var in enumerate(predictors):
                coef = model.coef_[i] / scaler.scale_[i]
                coefficients[var] = float(coef)

            # Sort by absolute importance
            sorted_vars = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = [v[0] for v in sorted_vars[:10]]

            # Model performance
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')

            # Educational notes
            edu_notes = {
                'what_is_ridge': 'Ridge adds an L2 penalty that shrinks coefficients toward zero but never exactly to zero.',
                'interpreting_alpha': f'Optimal alpha={model.alpha_:.4f}. Larger alpha = more shrinkage.',
                'vs_lasso': 'Unlike LASSO, Ridge keeps all variables but with reduced coefficients. Better when all predictors are relevant.',
                'multicollinearity': 'Ridge is particularly useful when predictors are highly correlated (multicollinearity).',
                'when_to_use': 'Use Ridge when you want to include all predictors but reduce overfitting and handle multicollinearity.'
            }

            return RegularizedRegressionResult(
                model_type="Ridge Regression (L2)",
                n_observations=len(clean_data),
                n_features=len(predictors),
                n_selected_features=len(predictors),  # Ridge keeps all
                optimal_alpha=float(model.alpha_),
                coefficients=coefficients,
                selected_features=top_features,
                r_squared=float(r2),
                mse=float(mse),
                cv_score=float(cv_scores.mean()),
                cv_std=float(cv_scores.std()),
                summary_text=f"Ridge regression with optimal alpha={model.alpha_:.4f}, R²={r2:.3f}",
                educational_notes=edu_notes
            )

        except Exception as e:
            result = self._insufficient_data("Ridge Regression")
            result.warnings = [str(e)]
            return result

    def elastic_net(
        self,
        data: pd.DataFrame,
        outcome: str,
        predictors: List[str],
        cv_folds: int = 5,
        l1_ratios: Optional[List[float]] = None
    ) -> RegularizedRegressionResult:
        """Elastic Net regression (combined L1 and L2).

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            predictors: List of predictor variable names
            cv_folds: Number of cross-validation folds
            l1_ratios: List of L1 ratios to try (0=Ridge, 1=LASSO)

        Returns:
            RegularizedRegressionResult object
        """
        all_vars = [outcome] + predictors
        clean_data = data[all_vars].dropna()

        if len(clean_data) < cv_folds * 2:
            return self._insufficient_data("Elastic Net")

        X = clean_data[predictors].values
        y = clean_data[outcome].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            if l1_ratios is None:
                l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

            model = ElasticNetCV(
                l1_ratio=l1_ratios,
                cv=cv_folds,
                random_state=self.random_state,
                max_iter=10000
            )
            model.fit(X_scaled, y)

            coefficients = {}
            for i, var in enumerate(predictors):
                coef = model.coef_[i] / scaler.scale_[i]
                coefficients[var] = float(coef)

            selected = [p for p, c in coefficients.items() if abs(c) > 1e-10]

            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')

            edu_notes = {
                'what_is_elastic_net': 'Elastic Net combines L1 (LASSO) and L2 (Ridge) penalties, balancing variable selection with coefficient shrinkage.',
                'l1_ratio': f'L1 ratio={model.l1_ratio_:.2f}. Closer to 1=more LASSO-like, closer to 0=more Ridge-like.',
                'when_to_use': 'Use when you want variable selection (like LASSO) but also want to handle correlated predictors (like Ridge).',
                'best_of_both': 'Elastic Net can select groups of correlated features together, unlike pure LASSO.'
            }

            return RegularizedRegressionResult(
                model_type="Elastic Net (L1+L2)",
                n_observations=len(clean_data),
                n_features=len(predictors),
                n_selected_features=len(selected),
                optimal_alpha=float(model.alpha_),
                l1_ratio=float(model.l1_ratio_),
                coefficients=coefficients,
                selected_features=selected,
                r_squared=float(r2),
                mse=float(mse),
                cv_score=float(cv_scores.mean()),
                cv_std=float(cv_scores.std()),
                summary_text=f"Elastic Net: alpha={model.alpha_:.4f}, L1_ratio={model.l1_ratio_:.2f}, {len(selected)} features selected",
                educational_notes=edu_notes
            )

        except Exception as e:
            result = self._insufficient_data("Elastic Net")
            result.warnings = [str(e)]
            return result

    def random_forest(
        self,
        data: pd.DataFrame,
        outcome: str,
        predictors: List[str],
        task_type: str = "auto",
        n_trees: int = 100,
        cv_folds: int = 5
    ) -> RandomForestResult:
        """Random Forest for variable importance and prediction.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            predictors: List of predictor variable names
            task_type: "classification", "regression", or "auto"
            n_trees: Number of trees in the forest
            cv_folds: Number of cross-validation folds

        Returns:
            RandomForestResult object
        """
        all_vars = [outcome] + predictors
        clean_data = data[all_vars].dropna()

        if len(clean_data) < cv_folds * 2:
            return self._insufficient_rf_data()

        X = clean_data[predictors].values
        y = clean_data[outcome].values

        # Determine task type
        if task_type == "auto":
            n_unique = len(np.unique(y))
            task_type = "classification" if n_unique <= 10 else "regression"

        try:
            if task_type == "classification":
                model = RandomForestClassifier(
                    n_estimators=n_trees,
                    random_state=self.random_state,
                    oob_score=True,
                    n_jobs=-1
                )
                scoring = 'roc_auc' if len(np.unique(y)) == 2 else 'accuracy'
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            else:
                model = RandomForestRegressor(
                    n_estimators=n_trees,
                    random_state=self.random_state,
                    oob_score=True,
                    n_jobs=-1
                )
                scoring = 'r2'
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

            # Fit model
            model.fit(X, y)

            # Feature importance
            importance = {}
            for i, var in enumerate(predictors):
                importance[var] = float(model.feature_importances_[i])

            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [v[0] for v in sorted_importance[:10]]

            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

            # Metrics
            y_pred = model.predict(X)

            result = RandomForestResult(
                model_type="Random Forest",
                task_type=task_type,
                n_observations=len(clean_data),
                n_features=len(predictors),
                n_trees=n_trees,
                feature_importance=importance,
                top_features=top_features,
                oob_score=float(model.oob_score_),
                cv_score=float(cv_scores.mean()),
                cv_std=float(cv_scores.std())
            )

            if task_type == "classification":
                result.accuracy = float(accuracy_score(y, y_pred))
                if len(np.unique(y)) == 2:
                    y_prob = model.predict_proba(X)[:, 1]
                    result.auc = float(roc_auc_score(y, y_prob))
                    result.precision = float(precision_score(y, y_pred))
                    result.recall = float(recall_score(y, y_pred))
                    result.f1 = float(f1_score(y, y_pred))
            else:
                result.r_squared = float(r2_score(y, y_pred))
                result.mse = float(mean_squared_error(y, y_pred))

            # Summary
            result.summary_text = self._generate_rf_summary(result, top_features[:5])

            # Educational notes
            result.educational_notes = {
                'what_is_rf': 'Random Forest is an ensemble of decision trees that provides robust predictions and variable importance rankings.',
                'interpreting_importance': 'Feature importance measures how much each variable contributes to reducing prediction error across all trees.',
                'oob_score': f'OOB score ({result.oob_score:.3f}) estimates performance on unseen data using out-of-bag samples.',
                'vs_linear': 'Unlike linear models, RF can capture non-linear relationships and interactions automatically.',
                'when_to_use': 'Use RF for prediction, variable importance screening, or when you suspect non-linear relationships.'
            }

            return result

        except Exception as e:
            result = self._insufficient_rf_data()
            result.warnings = [str(e)]
            return result

    def gradient_boosting(
        self,
        data: pd.DataFrame,
        outcome: str,
        predictors: List[str],
        task_type: str = "auto",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        cv_folds: int = 5
    ) -> GradientBoostingResult:
        """Gradient Boosting for prediction and variable importance.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            predictors: List of predictor variable names
            task_type: "classification", "regression", or "auto"
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinkage
            max_depth: Maximum depth of individual trees
            cv_folds: Number of cross-validation folds

        Returns:
            GradientBoostingResult object
        """
        all_vars = [outcome] + predictors
        clean_data = data[all_vars].dropna()

        if len(clean_data) < cv_folds * 2:
            return self._insufficient_gb_data()

        X = clean_data[predictors].values
        y = clean_data[outcome].values

        if task_type == "auto":
            n_unique = len(np.unique(y))
            task_type = "classification" if n_unique <= 10 else "regression"

        try:
            if task_type == "classification":
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=self.random_state
                )
                scoring = 'roc_auc' if len(np.unique(y)) == 2 else 'accuracy'
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            else:
                model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=self.random_state
                )
                scoring = 'r2'
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

            model.fit(X, y)

            # Feature importance
            importance = {}
            for i, var in enumerate(predictors):
                importance[var] = float(model.feature_importances_[i])

            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [v[0] for v in sorted_importance[:10]]

            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

            y_pred = model.predict(X)

            # Training history
            if hasattr(model, 'train_score_'):
                train_scores = [float(s) for s in model.train_score_]
            else:
                train_scores = None

            result = GradientBoostingResult(
                model_type="Gradient Boosting",
                task_type=task_type,
                n_observations=len(clean_data),
                n_features=len(predictors),
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                feature_importance=importance,
                top_features=top_features,
                cv_score=float(cv_scores.mean()),
                cv_std=float(cv_scores.std()),
                train_scores=train_scores
            )

            if task_type == "classification":
                result.accuracy = float(accuracy_score(y, y_pred))
                if len(np.unique(y)) == 2:
                    y_prob = model.predict_proba(X)[:, 1]
                    result.auc = float(roc_auc_score(y, y_prob))
                    result.precision = float(precision_score(y, y_pred))
                    result.recall = float(recall_score(y, y_pred))
                    result.f1 = float(f1_score(y, y_pred))
            else:
                result.r_squared = float(r2_score(y, y_pred))
                result.mse = float(mean_squared_error(y, y_pred))

            result.summary_text = f"Gradient Boosting ({task_type}): CV score = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"

            result.educational_notes = {
                'what_is_gb': 'Gradient Boosting builds trees sequentially, with each tree correcting errors of previous trees.',
                'hyperparameters': f'Using {n_estimators} trees, learning_rate={learning_rate}, max_depth={max_depth}.',
                'learning_rate': 'Lower learning rate requires more trees but often gives better generalization.',
                'vs_rf': 'GB often achieves higher accuracy than RF but is more prone to overfitting. RF is more robust to hyperparameters.',
                'when_to_use': 'Use GB when prediction accuracy is paramount and you can tune hyperparameters carefully.'
            }

            return result

        except Exception as e:
            result = self._insufficient_gb_data()
            result.warnings = [str(e)]
            return result

    def _generate_lasso_summary(
        self,
        selected: List[str],
        alpha: float,
        r2: float,
        cv_score: float
    ) -> str:
        """Generate LASSO summary."""
        lines = [
            f"LASSO Regression Results:",
            f"- Optimal regularization (alpha): {alpha:.4f}",
            f"- Selected {len(selected)} features",
            f"- R² = {r2:.3f}, CV R² = {cv_score:.3f}",
            f"- Selected variables: {', '.join(selected[:5])}{'...' if len(selected) > 5 else ''}"
        ]
        return '\n'.join(lines)

    def _generate_rf_summary(self, result: RandomForestResult, top_features: List[str]) -> str:
        """Generate Random Forest summary."""
        lines = [
            f"Random Forest ({result.task_type}):",
            f"- {result.n_trees} trees, {result.n_features} features",
            f"- OOB Score: {result.oob_score:.3f}",
            f"- CV Score: {result.cv_score:.3f} ± {result.cv_std:.3f}",
            f"- Top features: {', '.join(top_features)}"
        ]
        return '\n'.join(lines)

    def _insufficient_data(self, method: str) -> RegularizedRegressionResult:
        return RegularizedRegressionResult(
            model_type=method,
            n_observations=0,
            n_features=0,
            warnings=["Insufficient data for analysis"]
        )

    def _insufficient_rf_data(self) -> RandomForestResult:
        return RandomForestResult(
            model_type="Random Forest",
            task_type="",
            n_observations=0,
            n_features=0,
            n_trees=0,
            warnings=["Insufficient data for analysis"]
        )

    def _insufficient_gb_data(self) -> GradientBoostingResult:
        return GradientBoostingResult(
            model_type="Gradient Boosting",
            task_type="",
            n_observations=0,
            n_features=0,
            n_estimators=0,
            learning_rate=0,
            max_depth=0,
            warnings=["Insufficient data for analysis"]
        )

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class PredictionModels:
    """
    Class for performing various prediction tasks on the NBA dataset.
    """

    def __init__(self, data):
        """
        Initialize PredictionModels object with NBA dataset.

        Parameters:
        - data (DataFrame): NBA dataset
        """
        self.data = data
    
    def logistic_regression_classification(self):
        """
        Perform binary classification (Win/Loss Prediction) using Logistic Regression.

        Returns:
        - results (dict): Dictionary containing accuracy and classification report
        """
        X = self.data[['elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'importance']]
        y = np.where(self.data['score1'] > self.data['score2'], 1, 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logreg_model = LogisticRegression()
        logreg_model.fit(X_train, y_train)

        y_pred = logreg_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)

        results = {'accuracy': accuracy, 'classification_report': classification_rep}
        return results

    def random_forest_regression(self):
        """
        Perform regression (Score Prediction) using Random Forest Regressor.

        Returns:
        - results (dict): Dictionary containing MSE, MAE, and R-squared for both teams
        """
        X = self.data[['elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'importance']]
        y1 = self.data['score1']
        y2 = self.data['score2']

        X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.2, random_state=42)

        rf_model1 = RandomForestRegressor()
        rf_model1.fit(X_train, y1_train)
        y1_pred = rf_model1.predict(X_test)

        rf_model2 = RandomForestRegressor()
        rf_model2.fit(X_train, y2_train)
        y2_pred = rf_model2.predict(X_test)

        mse1 = mean_squared_error(y1_test, y1_pred)
        mae1 = mean_absolute_error(y1_test, y1_pred)
        r2_1 = r2_score(y1_test, y1_pred)

        mse2 = mean_squared_error(y2_test, y2_pred)
        mae2 = mean_absolute_error(y2_test, y2_pred)
        r2_2 = r2_score(y2_test, y2_pred)

        results = {'mse1': mse1, 'mae1': mae1, 'r2_1': r2_1, 'mse2': mse2, 'mae2': mae2, 'r2_2': r2_2}
        return results

    def gradient_boosting_classification(self):
        """
        Perform ranking prediction using Gradient Boosting Classifier.

        Returns:
        - results (dict): Dictionary containing accuracy and classification report
        """
        X = self.data[['elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'importance']]
        y = np.where(self.data['score1'] > self.data['score2'], 1, 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        gb_classifier = GradientBoostingClassifier()
        gb_classifier.fit(X_train, y_train)

        y_pred = gb_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)

        results = {'accuracy': accuracy, 'classification_report': classification_rep}
        return results

    def neural_network_classification(self):
        """
        Perform outcome probability prediction using Neural Networks.

        Returns:
        - results (dict): Dictionary containing accuracy and classification report
        """
        X = self.data[['elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'importance']]
        y = np.where(self.data['score1'] > self.data['score2'], 1, 0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
        mlp_classifier.fit(X_train, y_train)

        y_pred = mlp_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)

        results = {'accuracy': accuracy, 'classification_report': classification_rep}
        return results

    def decision_tree_classification(self):
        """
        Perform game importance prediction using Decision Trees

        Returns:
        - results (dict): Dictionary containing accuracy, precision, recall, and F1 score
        """
        X = self.data[['elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'importance']]
        y = np.where(self.data['importance'] > self.data['importance'].median(), 1, 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, y_train)

        y_pred = decision_tree.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        return results
    

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class GazeTracker:
    def __init__(self):
        self.model_x = None
        self.model_y = None
        self.poly = PolynomialFeatures(degree=3)
        self.screen_width = None
        self.screen_height = None
        self.sensitivity_factor = 8.0  # Increased sensitivity
        self.edge_compensation = 1.5   # Stronger edge compensation

    def apply_edge_compensation(self, value, max_value):
        """Apply non-linear scaling to help reach screen edges"""
        normalized = value / max_value
        compensated = np.sign(normalized) * (np.abs(normalized) ** (1/self.edge_compensation))
        return compensated * max_value

    def train(self, train_data):
        """Train the gaze prediction model using calibration data"""
        df = pd.DataFrame(train_data)
        
        # Get screen dimensions from calibration data
        self.screen_width = df['screen_width'].iloc[0]
        self.screen_height = df['screen_height'].iloc[0]

        # Calculate relative iris positions
        df['left_eye_width'] = np.sqrt(
            (df['left_eye_outer_x'] - df['left_eye_inner_x'])**2 + 
            (df['left_eye_outer_y'] - df['left_eye_inner_y'])**2
        )
        df['right_eye_width'] = np.sqrt(
            (df['right_eye_outer_x'] - df['right_eye_inner_x'])**2 + 
            (df['right_eye_outer_y'] - df['right_eye_inner_y'])**2
        )

        # Calculate relative iris positions
        df['left_iris_rel_x'] = (df['left_iris_x'] - df['left_eye_outer_x']) / df['left_eye_width']
        df['right_iris_rel_x'] = (df['right_iris_x'] - df['right_eye_inner_x']) / df['right_eye_width']

        # Calculate combined eye movement (90/10 split favoring dominant movement)
        df['combined_x'] = np.where(
            np.abs(df['left_iris_rel_x']) > np.abs(df['right_iris_rel_x']),
            0.9 * df['left_iris_rel_x'] + 0.1 * df['right_iris_rel_x'],
            0.1 * df['left_iris_rel_x'] + 0.9 * df['right_iris_rel_x']
        )

        # Similar calculations for Y coordinates
        df['left_eye_height'] = np.sqrt(
            (df['left_eye_outer_y'] - df['left_eye_inner_y'])**2
        )
        df['right_eye_height'] = np.sqrt(
            (df['right_eye_outer_y'] - df['right_eye_inner_y'])**2
        )
        
        df['left_iris_rel_y'] = (df['left_iris_y'] - df['left_eye_outer_y']) / df['left_eye_height']
        df['right_iris_rel_y'] = (df['right_iris_y'] - df['right_eye_inner_y']) / df['right_eye_height']
        
        df['combined_y'] = np.where(
            np.abs(df['left_iris_rel_y']) > np.abs(df['right_iris_rel_y']),
            0.9 * df['left_iris_rel_y'] + 0.1 * df['right_iris_rel_y'],
            0.1 * df['left_iris_rel_y'] + 0.9 * df['right_iris_rel_y']
        )

        # Prepare training data
        X = df[['combined_x', 'combined_y']].values
        X = self.poly.fit_transform(X)
        
        # Train separate models for x and y coordinates
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        
        # Normalize target values based on screen dimensions
        y_x = df['point_x'].values / self.screen_width
        y_y = df['point_y'].values / self.screen_height
        
        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)

    def predict(self, test_data):
        """Predict gaze position using the trained model"""
        df = pd.DataFrame(test_data)
        
        # Calculate relative iris positions
        df['left_eye_width'] = np.sqrt(
            (df['left_eye_outer_x'] - df['left_eye_inner_x'])**2 + 
            (df['left_eye_outer_y'] - df['left_eye_inner_y'])**2
        )
        df['right_eye_width'] = np.sqrt(
            (df['right_eye_outer_x'] - df['right_eye_inner_x'])**2 + 
            (df['right_eye_outer_y'] - df['right_eye_inner_y'])**2
        )

        df['left_iris_rel_x'] = (df['left_iris_x'] - df['left_eye_outer_x']) / df['left_eye_width']
        df['right_iris_rel_x'] = (df['right_iris_x'] - df['right_eye_inner_x']) / df['right_eye_width']

        # Calculate combined eye movement with sensitivity factor
        df['combined_x'] = self.sensitivity_factor * np.where(
            np.abs(df['left_iris_rel_x']) > np.abs(df['right_iris_rel_x']),
            0.9 * df['left_iris_rel_x'] + 0.1 * df['right_iris_rel_x'],
            0.1 * df['left_iris_rel_x'] + 0.9 * df['right_iris_rel_x']
        )

        # Similar calculations for Y coordinates
        df['left_eye_height'] = np.sqrt(
            (df['left_eye_outer_y'] - df['left_eye_inner_y'])**2
        )
        df['right_eye_height'] = np.sqrt(
            (df['right_eye_outer_y'] - df['right_eye_inner_y'])**2
        )
        
        df['left_iris_rel_y'] = (df['left_iris_y'] - df['left_eye_outer_y']) / df['left_eye_height']
        df['right_iris_rel_y'] = (df['right_iris_y'] - df['right_eye_inner_y']) / df['right_eye_height']
        
        df['combined_y'] = self.sensitivity_factor * np.where(
            np.abs(df['left_iris_rel_y']) > np.abs(df['right_iris_rel_y']),
            0.9 * df['left_iris_rel_y'] + 0.1 * df['right_iris_rel_y'],
            0.1 * df['left_iris_rel_y'] + 0.9 * df['right_iris_rel_y']
        )

        # Prepare test data
        X = df[['combined_x', 'combined_y']].values
        X = self.poly.transform(X)
        
        # Make predictions and scale back to screen coordinates
        pred_x = self.model_x.predict(X) * self.screen_width
        pred_y = self.model_y.predict(X) * self.screen_height
        
        # Apply edge compensation
        pred_x = np.array([self.apply_edge_compensation(x, self.screen_width) for x in pred_x])
        pred_y = np.array([self.apply_edge_compensation(y, self.screen_height) for y in pred_y])
        
        return pred_x, pred_y 
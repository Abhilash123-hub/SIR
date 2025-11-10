import pandas as pd
import pandas.api.types as ptypes
import requests
import io
import base64
import uuid
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from flask import Flask, request, render_template_string, render_template, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Flask App Initialization ---
app = Flask(__name__)

# --- Server-Side Data Cache (for Project 1) ---
data_cache = {}

# --- 2. NEW HOMEPAGE ROUTES ---

@app.route('/')
def index():
    """Renders the NEW main menu page."""
    return render_template('index.html')

@app.route('/visualizer')
def visualizer_page():
    """Renders the CSV Visualizer page."""
    return render_template('visualizer.html')

@app.route('/sir')
def sir_fitter_page():
    """Renders the SIR Fitter page."""
    # We must use the 'dynamic' HTML file
    return render_template('sir_fitter.html') 


# --- 3. PROJECT 1: CSV VISUALIZER PYTHON ---

# --- Helper Functions for "AI" Analysis ---
def analyze_dataframe(df):
    column_types = {}
    columns = df.columns.tolist()
    for col in columns:
        try:
            temp_col = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
            if temp_col.isna().sum() / len(df) < 0.20:
                column_types[col] = 'temporal'
                df[col] = temp_col
                continue
        except Exception:
            pass
        if ptypes.is_numeric_dtype(df[col]):
            column_types[col] = 'numerical'
            continue
        unique_count = df[col].nunique()
        if unique_count < 50 or unique_count / len(df) < 0.1:
            column_types[col] = 'categorical'
        else:
            column_types[col] = 'text'
    return columns, column_types, df

def get_plot_suggestions(columns, column_types):
    temporal_cols = [col for col, ctype in column_types.items() if ctype == 'temporal']
    numerical_cols = [col for col, ctype in column_types.items() if ctype == 'numerical']
    categorical_cols = [col for col, ctype in column_types.items() if ctype == 'categorical']
    if temporal_cols and numerical_cols:
        return {'plot_type': 'line', 'x_col': temporal_cols[0], 'y_col': numerical_cols[0]}
    if categorical_cols and numerical_cols:
        return {'plot_type': 'bar', 'x_col': categorical_cols[0], 'y_col': numerical_cols[0]}
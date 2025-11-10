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
    if len(numerical_cols) >= 2:
        return {'plot_type': 'scatter', 'x_col': numerical_cols[0], 'y_col': numerical_cols[1]}
    if len(categorical_cols) >= 2:
        return {'plot_type': 'bar', 'x_col': categorical_cols[0], 'y_col': categorical_cols[1]}
    return {'plot_type': 'bar', 'x_col': columns[0] if columns else '', 'y_col': columns[1] if len(columns) > 1 else ''}

def create_dynamic_plot(df, x_col, y_col, plot_type, column_types):
    fig, ax = plt.subplots(figsize=(12, 7))
    x_type = column_types.get(x_col)
    y_type = column_types.get(y_col)
    try:
        if x_type in ('categorical', 'text') and y_type == 'numerical':
            if plot_type == 'bar':
                if df[x_col].nunique() > 50:
                    data_to_plot = df.groupby(x_col)[y_col].mean().nlargest(50)
                    ax.set_title(f'Mean of {y_col} vs. {x_col} (Top 50)')
                else:
                    data_to_plot = df.groupby(x_col)[y_col].mean()
                    ax.set_title(f'Mean of {y_col} vs. {x_col}')
                data_to_plot.plot(kind='bar', ax=ax, width=0.8)
                ax.set_ylabel(f'Mean of {y_col}')
                plt.xticks(rotation=75)
            else:
                sns.stripplot(x=x_col, y=y_col, data=df, ax=ax, jitter=True)
                ax.set_ylabel(y_col)
                if df[x_col].nunique() > 20: plt.xticks(rotation=90)
        elif x_type in ('categorical', 'text') and y_type in ('categorical', 'text'):
            if plot_type == 'scatter':
                sns.stripplot(x=x_col, y=y_col, data=df, ax=ax, jitter=0.2)
                ax.set_title(f'{y_col} vs. {x_col}')
                if df[x_col].nunique() > 20: plt.xticks(rotation=90)
                else: plt.xticks(rotation=0)
            else:
                x_unique_count = df[x_col].nunique()
                y_unique_count = df[y_col].nunique()
                if x_unique_count > 50 and y_unique_count < 50:
                    data_to_plot = df.groupby(y_col)[x_col].nunique()
                    data_to_plot.plot(kind='bar', ax=ax, width=0.8)
                    ax.set_xlabel(y_col); ax.set_ylabel(f'Unique Count of {x_col}'); ax.set_title(f'Unique {x_col} Count by {y_col}'); plt.xticks(rotation=0)
                elif x_unique_count < 50 and y_unique_count > 50:
                    data_to_plot = df.groupby(x_col)[y_col].nunique()
                    data_to_plot.plot(kind='bar', ax=ax, width=0.8)
                    ax.set_ylabel(f'Unique Count of {y_col}'); ax.set_title(f'Unique {y_col} Count by {x_col}'); plt.xticks(rotation=0)
                else:
                    cross_tab = pd.crosstab(df[x_col], df[y_col])
                    cross_tab.plot(kind='bar', ax=ax, width=0.8)
                    ax.set_ylabel('Count'); ax.set_title(f'Count of {y_col} by {x_col}'); plt.xticks(rotation=0)
        elif x_type == 'temporal' and y_type == 'numerical':
            df_sorted = df.sort_values(by=x_col)
            if plot_type == 'line' or df[x_col].nunique() < 50:
                ax.plot(df_sorted[x_col], df_sorted[y_col])
                ax.set_ylabel(y_col); ax.set_title(f'{y_col} vs. {x_col} (Daily)')
            elif plot_type == 'bar':
                data_to_plot = df.set_index(x_col).resample('M')[y_col].sum()
                data_to_plot.index = data_to_plot.index.strftime('%Y-%m')
                data_to_plot.plot(kind='bar', ax=ax, width=0.8)
                ax.set_ylabel(f'Monthly Total of {y_col}'); ax.set_title(f'Monthly {y_col} vs. {x_col}'); plt.xticks(rotation=75)
            fig.autofmt_xdate()
        elif x_type == 'numerical' and y_type == 'numerical':
            if plot_type == 'scatter': ax.scatter(df[x_col], df[y_col])
            else: df_sorted = df.sort_values(by=x_col); ax.plot(df_sorted[x_col], df_sorted[y_col])
            ax.set_ylabel(y_col); ax.set_title(f'{y_col} vs. {x_col}')
        else:
            data_to_plot = df[x_col].value_counts().head(50)
            data_to_plot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_ylabel('Count'); ax.set_title(f'Count of {x_col} (Top 50)'); plt.xticks(rotation=75)
        ax.set_xlabel(x_col)
        if not ax.get_title(): ax.set_title(f'{y_col} vs. {x_col}')
        ax.grid(True); plt.tight_layout()
    except Exception as e:
        plt.close(fig)
        raise Exception(f"Plotting Error: Could not plot {x_col} ({x_type}) vs {y_col} ({y_type}). Error: {e}")
    buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
    img_data = base64.b64encode(buf.

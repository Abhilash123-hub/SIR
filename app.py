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
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    return img_data

# --- Web Server Routes (The "Controller") ---
@app.route('/upload_and_analyze', methods=['POST'])
def upload_and_analyze():
    file = request.files.get('file_upload')
    if not file: return jsonify({'error': 'No file provided'}), 400
    try:
        data_stream = io.StringIO(file.read().decode('utf-8'))
        df = pd.read_csv(data_stream)
        columns, column_types, modified_df = analyze_dataframe(df)
        suggestions = get_plot_suggestions(columns, column_types)
        file_id = str(uuid.uuid4())
        data_cache[file_id] = {'df': modified_df, 'column_types': column_types}
        return jsonify({'file_id': file_id, 'columns': columns, 'suggestions': suggestions})
    except Exception as e:
        return jsonify({'error': f'Error processing file: {e}'}), 500

@app.route('/plot_dynamic', methods=['POST'])
def plot_dynamic():
    try:
        config = request.get_json(force=True)
        file_id = config.get('file_id')
        x_col, y_col, plot_type = config.get('x_col'), config.get('y_col'), config.get('plot_type')
        if not all([file_id, x_col, y_col]):
            return jsonify({'error': 'Missing file_id, x_col or y_col.'}), 400
        cached_data = data_cache.get(file_id)
        if cached_data is None:
            return jsonify({'error': 'Data expired. Please upload again.'}), 400
        df = cached_data['df'].copy()
        column_types = cached_data.get('column_types', {}).copy()
        for col in (x_col, y_col):
            if col not in df.columns:
                return jsonify({'error': f'Column "{col}" not found.'}), 400
        img_data = create_dynamic_plot(df, x_col, y_col, plot_type, column_types)
        return jsonify({'img_data': img_data})
    except Exception as e:
        return jsonify({'error': f'{e}'}), 500


# --- 4. PROJECT 3: SIR MODEL FITTER PYTHON ---

def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = (beta * S * I / N) - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def fit_sir_model(real_data, population):
    days = len(real_data)
    t = np.linspace(0, days, days)
    def sir_model_for_fitting(t, beta, gamma):
        I0 = real_data[0]
        S0 = population - I0
        R0 = 0
        y0 = S0, I0, R0
        ret = odeint(sir_model, y0, t, args=(population, beta, gamma))
        S, I, R = ret.T
        return I.clip(min=0)
    param_bounds = ([0.1, 0.01], [1.0, 0.2])
    popt, pcov = curve_fit(sir_model_for_fitting, t, real_data, p0=[0.3, 0.1], bounds=param_bounds)
    beta_fit, gamma_fit = popt
    fitted_simulation = sir_model_for_fitting(t, beta_fit, gamma_fit)
    return beta_fit, gamma_fit, t, fitted_simulation

# -----------------------------------------------------------------
# --- THIS IS THE NEW, FIXED FUNCTION ---
# -----------------------------------------------------------------
def get_real_data(country_name, days=90):
    """
    Fetches real case data from the disease.sh API.
    *** UPDATED to make TWO API calls to fix the 'population' error. ***
    """
    
    population = None
    
    # --- NEW CALL 1: Get population ---
    try:
        # This new URL just gets country info
        pop_url = f"https://disease.sh/v3/covid-19/countries/{country_name}"
        pop_response = requests.get(pop_url)
        pop_response.raise_for_status() # Check for errors
        pop_data = pop_response.json()
        
        population = pop_data.get('population')
        
        if not population:
            raise Exception("No 'population' field found in country data.")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"API Error fetching population for '{country_name}': {e}")
    
    # --- CALL 2: Get historical cases (like before) ---
    try:
        # This is our original URL for cases
        hist_url = f"https://disease.sh/v3/covid-19/historical/{country_name}?lastdays={days}"
        hist_response = requests.get(hist_url)
        hist_response.raise_for_status() 
        hist_data = hist_response.json()
        
        # The structure here is { "timeline": { "cases": { ... } } }
        timeline = hist_data.get('timeline', {}).get('cases', {})
        
        if not timeline:
            raise Exception("Could not find 'cases' in API response.")
            
        case_counts = list(timeline.values())
        
        # --- Same 'active cases' logic as before ---
        active_cases = []
        for i in range(len(case_counts)):
            if i < 14:
                # Can't calculate 14-day recovery, so just use a fraction
                active = case_counts[i] * 0.1 
            else:
                # Active = New cases in last 14 days
                active = case_counts[i] - case_counts[i-14]
            active_cases.append(max(0, active)) # Ensure it's not negative
            
        return np.array(active_cases), population
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"API Error fetching historical data for '{country_name}': {e}")
    except Exception as e:
        raise Exception(f"Error processing data for '{country_name}': {e}")
# -----------------------------------------------------------------
# --- END OF THE NEW, FIXED FUNCTION ---
# -----------------------------------------------------------------

@app.route('/fit')
def fit_and_plot():
    country = request.args.get('country')
    if not country: return jsonify({"error": "Country name is required."}), 400
    try:
        real_data, population = get_real_data(country)
        beta, gamma, t, fitted_I = fit_sir_model(real_data, population)
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        ax.plot(t, real_data, 'ko', label='Real Active Cases', markersize=5)
        ax.plot(t, fitted_I, 'r-', label='Fitted SIR Model (Infected)', linewidth=2)
        ax.set_xlabel('Time (Days)'); ax.set_ylabel('Number of People')
        ax.set_title(f'Real Data vs. Fitted SIR Model for {country}')
        ax.legend(); ax.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_data = base66.b64encode(buf.read()).decode('utf-8') # <- Corrected to b64encode
        return jsonify({"img_data": img_data, "beta": beta, "gamma": gamma})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- 5. Run the Application ---
# (Commented out for Render)
# if __name__ == '__main__':
#     app.run(debug=True)
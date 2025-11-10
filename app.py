import io
import base64
import requests
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from flask import Flask, request, render_template_string, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 1. Flask App Initialization ---
app = Flask(__name__)

# --- 2. HTML Template (Embedded) ---
# This is the entire webpage.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dynamic SIR Model Fitter</title>
    <style>
        body { font-family: sans-serif; margin: 2em; }
        .container { max-width: 900px; margin: auto; }
        form { margin-bottom: 2em; }
        #plot-img { max-width: 100%; border: 1px solid #ccc; }
        #loading { display: none; font-weight: bold; }
        .error { color: red; border: 1px solid red; padding: 10px; }
        #results-text { background: #f4f4f4; padding: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dynamic SIR Model Fitter</h1>
        <p>This tool fits the SIR model to real-world data to find transmission (β) and recovery (γ) rates. It can find two different transmission rates (before and after an "intervention").</p>
        
        <form id="sir-form">
            <label for="country">Country Name (e.g., USA, India):</label><br>
            <input type="text" id="country" name="country" value="USA" required>
            <br><br>
            
            <label for="intervention_day">Intervention Day (e.g., 30):</label><br>
            <input type="number" id="intervention_day" name="intervention_day" value="30" required>
            <small>The day (out of 90) when you want to split Beta (β).</small>
            <br><br>

            <input type="submit" value="Fit Model to Real Data">
        </form>
        
        <div id="loading">Fetching data and running simulations...</div>
        <div id="results"></div>
    </div>

    <script>
        const form = document.getElementById('sir-form');
        const loadingDiv = document.getElementById('loading');
        const resultsDiv = document.getElementById('results');

        form.addEventListener('submit', async function(event) {
            event.preventDefault();
            loadingDiv.style.display = 'block';
            resultsDiv.innerHTML = '';
            
            const country = document.getElementById('country').value;
            const interventionDay = document.getElementById('intervention_day').value;
            
            try {
                // This URL path '/fit' is defined in our Python code below
                const response = await fetch(`/fit?country=${country}&intervention=${interventionDay}`);
                const data = await response.json();
                
                loadingDiv.style.display = 'none';
                if (data.error) {
                    showError(data.error);
                } else if (data.img_data) {
                    resultsDiv.innerHTML = `
                        <h2>Results for ${country}</h2>
                        <div id="results-text">
                            <p><strong>Fitted Beta 1 (β₁) (Before Day ${interventionDay}):</strong> ${data.beta1.toFixed(4)}</p>
                            <p><strong>Fitted Beta 2 (β₂) (After Day ${interventionDay}):</strong> ${data.beta2.toFixed(4)}</p>
                            <p><strong>Fitted Recovery Rate (γ):</strong> ${data.gamma.toFixed(4)}</p>
                            <hr>
                            <p><strong>Estimated R₀ (Before):</strong> ${(data.beta1 / data.gamma).toFixed(2)}</p>
                            <p><strong>Estimated R₀ (After):</strong> ${(data.beta2 / data.gamma).toFixed(2)}</p>
                        </div>
                        <img src="data:image/png;base64,${data.img_data}" 
                             alt="SIR Plot" id="plot-img">
                    `;
                }
            } catch (err) {
                loadingDiv.style.display = 'none';
                showError(`Network Error: ${err.message}`);
            }
        });
        
        function showError(message) {
            resultsDiv.innerHTML = `<div class="error"><strong>Error:</strong> ${message}</div>`;
        }
    </script>
</body>
</html>
"""


# --- 3. SIR MODEL FITTER PYTHON (DYNAMIC VERSION) ---

@app.route('/')
def index():
    """Renders the main page."""
    return render_template_string(HTML_TEMPLATE)

def sir_model(y, t, N, beta, gamma):
    """
    The SIR differential equations.
    """
    S, I, R = y
    
    # We allow beta to be a function of time t
    current_beta = beta(t) if callable(beta) else beta
    
    dSdt = -current_beta * S * I / N
    dIdt = (current_beta * S * I / N) - gamma * I
    dRdt = gamma * I
    
    return dSdt, dIdt, dRdt

def fit_sir_model(real_data, population, intervention_day):
    """
    NEW DYNAMIC FITTER: Fits the SIR model to real-world data 
    to find beta1, beta2, and gamma.
    """
    days = len(real_data)
    t = np.linspace(0, days, days)
    
    def sir_model_for_fitting(t, beta1, beta2, gamma):
        
        # --- DYNAMIC BETA FUNCTION ---
        def dynamic_beta(time):
            if time < intervention_day:
                return beta1
            else:
                return beta2
        
        # Use the first data point as the initial infected number
        I0 = real_data[0]
        S0 = population - I0
        R0 = 0
        y0 = S0, I0, R0
        
        ret = odeint(sir_model, y0, t, args=(population, dynamic_beta, gamma))
        S, I, R = ret.T
        return I.clip(min=0)
        
    # Bounds for beta1, beta2, gamma
    param_bounds = ([0.1, 0.1, 0.01], [1.0, 1.0, 0.2])

    popt, pcov = curve_fit(
        sir_model_for_fitting,
        t,                  
        real_data,          
        p0=[0.3, 0.2, 0.1], 
        bounds=param_bounds 
    )
    
    beta1_fit, beta2_fit, gamma_fit = popt
    
    fitted_simulation = sir_model_for_fitting(t, beta1_fit, beta2_fit, gamma_fit)
    
    return beta1_fit, beta2_fit, gamma_fit, t, fitted_simulation

def get_real_data(country_name, days=90):
    """
    Fetches real case data from the disease.sh API.
    (This is the fixed version with two API calls)
    """
    
    population = None
    
    try:
        pop_url = f"https://disease.sh/v3/covid-19/countries/{country_name}"
        pop_response = requests.get(pop_url)
        pop_response.raise_for_status() 
        pop_data = pop_response.json()
        population = pop_data.get('population')
        if not population:
            raise Exception("No 'population' field found in country data.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"API Error fetching population for '{country_name}': {e}")
    
    try:
        hist_url = f"https://disease.sh/v3/covid-19/historical/{country_name}?lastdays={days}"
        hist_response = requests.get(hist_url)
        hist_response.raise_for_status() 
        hist_data = hist_response.json()
        timeline = hist_data.get('timeline', {}).get('cases', {})
        if not timeline:
            raise Exception("Could not find 'cases' in API response.")
        case_counts = list(timeline.values())
        active_cases = []
        for i in range(len(case_counts)):
            if i < 14:
                active = case_counts[i] * 0.1 
            else:
                active = case_counts[i] - case_counts[i-14]
            active_cases.append(max(0, active))
        # Ensure first data point is at least 1 to avoid math errors
        if active_cases[0] == 0:
            active_cases[0] = 1.0
        return np.array(active_cases), population
    except requests.exceptions.RequestException as e:
        raise Exception(f"API Error fetching historical data for '{country_name}': {e}")
    except Exception as e:
        raise Exception(f"Error processing data for '{country_name}': {e}")

@app.route('/fit')
def fit_and_plot():
    """
    UPDATED route to handle the new intervention_day parameter
    """
    country = request.args.get('country')
    intervention_day_str = request.args.get('intervention')
    
    if not country or not intervention_day_str: 
        return jsonify({"error": "Country and intervention day are required."}), 400
        
    try:
        intervention_day = int(intervention_day_str)
        
        real_data, population = get_real_data(country)
        
        beta1, beta2, gamma, t, fitted_I = fit_sir_model(real_data, population, intervention_day)
        
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        ax.plot(t, real_data, 'ko', label='Real Active Cases', markersize=5)
        ax.plot(t, fitted_I, 'r-', label='Fitted SIR Model (Dynamic)', linewidth=2)
        
        ax.axvline(x=intervention_day, color='b', linestyle='--', label=f'Intervention on Day {intervention_day}')
        
        ax.set_xlabel('Time (Days)'); ax.set_ylabel('Number of People')
        ax.set_title(f'Real Data vs. Dynamic Fitted SIR Model for {country}')
        ax.legend(); ax.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({
            "img_data": img_data,
            "beta1": beta1,
            "beta2": beta2,
            "gamma": gamma
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- 4. Run the Application ---
# (Commented out for Render deployment)
# if __name__ == '__main__':
#     app.run(debug=True)
# --- 4. PROJECT 3: SIR MODEL FITTER PYTHON (DYNAMIC VERSION) ---

def sir_model(y, t, N, beta, gamma):
    """
    The SIR differential equations.
    (This function is unchanged)
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
    
    # We need a function to optimize.
    # This function takes t, beta1, beta2, gamma and returns a *simulated* I curve.
    def sir_model_for_fitting(t, beta1, beta2, gamma):
        
        # --- DYNAMIC BETA FUNCTION ---
        # This helper function returns beta1 or beta2 based on the time t
        def dynamic_beta(time):
            if time < intervention_day:
                return beta1
            else:
                return beta2
        
        I0 = real_data[0]
        S0 = population - I0
        R0 = 0
        y0 = S0, I0, R0
        
        # Run the simulation, passing our *dynamic_beta* function
        ret = odeint(sir_model, y0, t, args=(population, dynamic_beta, gamma))
        S, I, R = ret.T
        return I.clip(min=0) # Return the "Infected" curve
        
    # Set reasonable bounds
    # beta1, beta2, gamma
    param_bounds = ([0.1, 0.1, 0.01], [1.0, 1.0, 0.2])

    # Run the curve_fit algorithm!
    # It will now solve for THREE parameters (beta1, beta2, gamma)
    popt, pcov = curve_fit(
        sir_model_for_fitting,
        t,                  # X-axis (time)
        real_data,          # Y-axis (real infected data)
        p0=[0.3, 0.2, 0.1], # Initial guesses for beta1, beta2, gamma
        bounds=param_bounds 
    )
    
    # Extract the fitted parameters
    beta1_fit, beta2_fit, gamma_fit = popt
    
    # Rerun the simulation with the *final* fitted parameters
    fitted_simulation = sir_model_for_fitting(t, beta1_fit, beta2_fit, gamma_fit)
    
    return beta1_fit, beta2_fit, gamma_fit, t, fitted_simulation

def get_real_data(country_name, days=90):
    """
    Fetches real case data from the disease.sh API.
    (This function is unchanged from the last fix)
    """
    
    population = None
    
    # --- NEW CALL 1: Get population ---
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
    
    # --- CALL 2: Get historical cases (like before) ---
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
    # --- NEW ---
    intervention_day_str = request.args.get('intervention')
    
    if not country or not intervention_day_str: 
        return jsonify({"error": "Country and intervention day are required."}), 400
        
    try:
        # --- NEW ---
        intervention_day = int(intervention_day_str)
        
        real_data, population = get_real_data(country)
        
        # --- PASS IT TO THE FITTER ---
        beta1, beta2, gamma, t, fitted_I = fit_sir_model(real_data, population, intervention_day)
        
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        ax.plot(t, real_data, 'ko', label='Real Active Cases', markersize=5)
        ax.plot(t, fitted_I, 'r-', label='Fitted SIR Model (Dynamic)', linewidth=2)
        
        # --- NEW: Add a vertical line to show the intervention ---
        ax.axvline(x=intervention_day, color='b', linestyle='--', label=f'Intervention on Day {intervention_day}')
        
        ax.set_xlabel('Time (Days)'); ax.set_ylabel('Number of People')
        ax.set_title(f'Real Data vs. Dynamic Fitted SIR Model for {country}')
        ax.legend(); ax.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        
        # --- NEW: Return all 3 parameters ---
        return jsonify({
            "img_data": img_data,
            "beta1": beta1,
            "beta2": beta2,
            "gamma": gamma
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- 5. Run the Application ---
# (Commented out for Render)
# if __name__ == '__main__':
#     app.run(debug=True)
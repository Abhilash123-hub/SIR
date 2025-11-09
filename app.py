import pandas as pd
import io
import base64
import uuid
from flask import Flask, request, render_template_string, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 1. Flask App Initialization ---
app = Flask(__name__)

# --- 2. HTML Template (Single Page) ---
# This is a brand new UI for the SIR model
HTML_SINGLE_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>SIR Epidemic Simulator</title>
    <style>
        body { font-family: sans-serif; margin: 2em; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1em 2em; }
        input[type=number] { width: 100%; padding: 5px; }
        input[type=submit] { padding: 8px 15px; font-size: 16px; margin-top: 1em; }
        #results { margin-top: 2em; }
        .error { color: red; border: 1px solid red; padding: 10px; }
        #loading { display: none; font-weight: bold; }
        #plot-img { max-width: 90%; margin-top: 1em; }
    </style>
</head>
<body>
    <h2>SIR Epidemic Model Simulator</h2>
    <p>
      Enter the parameters to simulate an epidemic. This will plot the curves for
      <b>Susceptible</b>, <b>Infected</b>, and <b>Recovered</b> populations over time.
    </p>

    <form id="sir-form">
        <div class="form-grid">
            <div>
                <label for="N">Total Population (N):</label><br>
                <input type="number" id="N" name="N" value="10000">
            </div>
            <div>
                <label for="days">Days to Simulate:</label><br>
                <input type="number" id="days" name="days" value="160">
            </div>
            <div>
                <label for="I0">Initial Infected (I_0):</label><br>
                <input type="number" id="I0" name="I0" value="10">
            </div>
            <div>
                <label for="R0">Initial Recovered (R_0):</label><br>
                <input type="number" id="R0" name="R0" value="0">
            </div>
            <div>
                <label for="beta">Infection Rate (beta):</label><br>
                <input type="number" id="beta" name="beta" value="0.3" step="0.01">
            </div>
            <div>
                <label for="gamma">Recovery Rate (gamma):</label><br>
                <input type="number" id="gamma" name="gamma" value="0.1" step="0.01">
            </div>
        </div>
        
        <input type="submit" value="Run Simulation">
    </form>

    <div id="results">
        <div id="loading">Loading...</div>
    </div>

    <script>
        // --- Find our key HTML elements ---
        const sirForm = document.getElementById('sir-form');
        const resultsDiv = document.getElementById('results');
        const loadingDiv = document.getElementById('loading');

        // --- Handle Form Submission ---
        sirForm.addEventListener('submit', async function (event) {
            event.preventDefault(); // Stop page reload
            
            loadingDiv.style.display = 'block';
            resultsDiv.innerHTML = ''; // Clear old results
            resultsDiv.appendChild(loadingDiv);

            // 1. Get the parameters from the form
            const formData = new FormData(sirForm);
            
            // 2. Convert form data to a simple JSON object
            const params = {};
            formData.forEach((value, key) => {
                // Convert values to numbers
                params[key] = parseFloat(value); 
            });

            try {
                // 3. Send the parameters to the '/run_simulation' route
                const response = await fetch('/run_simulation', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(params) // Send params as JSON
                });

                const data = await response.json();
                loadingDiv.style.display = 'none';

                // 4. Display the final plot or an error
                if (data.error) {
                    showError(data.error);
                } else if (data.img_data) {
                    resultsDiv.innerHTML = `
                        <img src="data:image/png;base64,${data.img_data}" 
                             alt="SIR Model Plot" id="plot-img">
                    `;
                }

            } catch (err) {
                showError(`Network Error: ${err.message}`);
            }
        });
        
        // --- Helper function to show errors ---
        function showError(message) {
            loadingDiv.style.display = 'none';
            resultsDiv.innerHTML = `<div class="error">
                <strong>Error:</strong> ${message}
            </div>`;
        }
    </script>
</body>
</html>
"""

# --- 3. NEW: SIR Simulation Logic ---

def run_sir_model(N, I0, R0, beta, gamma, days):
    """
    Runs a simple discrete-time SIR model.
    N = Total Population
    I0 = Initial Infected
    R0 = Initial Recovered
    beta = Infection rate
    gamma = Recovery rate
    days = Number of days to simulate
    
    Returns three lists: (S_vec, I_vec, R_vec)
    """
    # Initial Susceptible (S0)
    S0 = N - I0 - R0
    
    # Initialize our vectors (lists) to store the data
    S_vec = [S0]
    I_vec = [I0]
    R_vec = [R0]
    
    # Set the current state
    S, I, R = S0, I0, R0
    
    # Loop for the simulation duration
    for t in range(int(days)):
        
        # --- The SIR Model Equations ---
        # 1. New infections
        # (beta * I * S / N) is the number of new people infected this day
        new_infections = (beta * I * S) / N
        
        # 2. New recoveries
        # (gamma * I) is the number of people who recover this day
        new_recoveries = gamma * I
        
        # 3. Update the compartments
        S_new = S - new_infections
        I_new = I + new_infections - new_recoveries
        R_new = R + new_recoveries
        
        # 4. Save the new values
        S_vec.append(S_new)
        I_vec.append(I_new)
        R_vec.append(R_new)
        
        # 5. Set the state for the next day's loop
        S, I, R = S_new, I_new, R_new
        
    return S_vec, I_vec, R_vec


def create_sir_plot(S_vec, I_vec, R_vec):
    """
    Generates the plot image from the SIR simulation data.
    Returns a base64-encoded image string.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    days = range(len(S_vec)) # Create the x-axis vector
    
    # Plot all three curves
    ax.plot(days, S_vec, label='Susceptible', color='blue', linewidth=2)
    ax.plot(days, I_vec, label='Infected', color='red', linewidth=2)
    ax.plot(days, R_vec, label='Recovered', color='green', linewidth=2)
    
    # Add formatting
    ax.set_xlabel('Days')
    ax.set_ylabel('Population')
    ax.set_title('SIR Model Simulation')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    # --- Save plot to in-memory buffer ---
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_data

# --- 4. Web Server Routes (The "Controller") ---

@app.route('/')
def index():
    """Renders the home page (our only page)."""
    return render_template_string(HTML_SINGLE_PAGE)


@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """
    Gets simulation parameters, runs the model,
    and returns the plot.
    """
    try:
        # 1. Get the parameters from the frontend's JSON request
        params = request.json
        N = params.get('N')
        I0 = params.get('I0')
        R0 = params.get('R0')
        beta = params.get('beta')
        gamma = params.get('gamma')
        days = params.get('days')
        
        # 2. Run the simulation
        S_vec, I_vec, R_vec = run_sir_model(N, I0, R0, beta, gamma, days)
            
        # 3. Create the plot
        img_data = create_sir_plot(S_vec, I_vec, R_vec)
        
        # 4. Send the image back
        return jsonify({'img_data': img_data})
        
    except Exception as e:
        print(f"Simulation/Plotting Error: {e}")
        return jsonify({'error': f'An error occurred: {e}'}), 500

# --- 5. Run the Application ---
#if __name__ == '__main__':
#  print("Starting Flask server...")
#  print("Open http://127.0.0.1:5000 in your web browser.")
#  app.run(debug=True, port=5000)

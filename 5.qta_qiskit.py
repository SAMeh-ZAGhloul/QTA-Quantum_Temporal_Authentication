"""
Quantum Temporal Authentication (QTA) Simulator
GUI implementation with real-time visualization
Qiskit backend for quantum circuit simulation
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import threading
import time
from collections import deque

# Initialize Dash app
app = dash.Dash(__name__,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])
app.title = "Quantum Temporal Authentication Simulator"

# Global simulation state (thread-safe)
class SimulationState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.frame = 0
        self.qber_history = deque(maxlen=500)
        self.delay_history = deque(maxlen=500)
        self.auth_history = deque(maxlen=500)
        self.eve_intercepts = deque(maxlen=500)
        self.thread = None

    def reset(self):
        with self.lock:
            self.frame = 0
            self.qber_history.clear()
            self.delay_history.clear()
            self.auth_history.clear()
            self.eve_intercepts.clear()

    def get_data(self):
        with self.lock:
            return {
                'frame': self.frame,
                'qber_history': list(self.qber_history),
                'delay_history': list(self.delay_history),
                'auth_history': list(self.auth_history),
                'eve_intercepts': list(self.eve_intercepts),
                'running': self.running
            }

    def add_frame_data(self, qber, delay, authenticated, eve_count):
        with self.lock:
            self.frame += 1
            self.qber_history.append(qber)
            self.delay_history.append(delay)
            self.auth_history.append(1 if authenticated else 0)
            self.eve_intercepts.append(eve_count)

sim_state = SimulationState()

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Quantum Temporal Authentication (QTA) Simulator",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
        html.P("Real-time Qiskit-based quantum authentication with temporal verification",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),

    # Control panel
    html.Div([
        html.H3("Simulation Parameters", style={'color': '#2c3e50', 'marginBottom': '15px'}),

        html.Div([
            html.Label("Frames to Simulate:", style={'fontWeight': 'bold'}),
            dcc.Slider(id='frames-slider', min=10, max=500, step=10, value=100,
                      marks={i: str(i) for i in [10, 100, 250, 500]}),
            html.Div(id='frames-display', style={'textAlign': 'center', 'marginTop': '5px'}),

            html.Label("Qubits per Frame:", style={'fontWeight': 'bold', 'marginTop': '15px'}),
            dcc.Slider(id='qubits-slider', min=5, max=50, step=5, value=20,
                      marks={i: str(i) for i in [5, 20, 35, 50]}),
            html.Div(id='qubits-display', style={'textAlign': 'center', 'marginTop': '5px'}),

            html.Label("Eve Attack Probability:", style={'fontWeight': 'bold', 'marginTop': '15px'}),
            dcc.Slider(id='eve-slider', min=0, max=1, step=0.1, value=0.3,
                      marks={i/10: f"{i*10}%" for i in range(0, 11, 2)}),
            html.Div(id='eve-display', style={'textAlign': 'center', 'marginTop': '5px'}),

            html.Div([
                html.Button('Start Simulation', id='start-button', n_clicks=0,
                           style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '10px 20px',
                                 'border': 'none', 'borderRadius': '5px', 'marginTop': '20px',
                                 'fontWeight': 'bold', 'fontSize': '16px', 'width': '48%'}),
                html.Button('Reset', id='reset-button', n_clicks=0,
                           style={'backgroundColor': '#e74c3c', 'color': 'white', 'padding': '10px 20px',
                                 'border': 'none', 'borderRadius': '5px', 'marginTop': '20px',
                                 'fontWeight': 'bold', 'fontSize': '16px', 'width': '48%', 'marginLeft': '4%'})
            ], style={'display': 'flex', 'justifyContent': 'space-between'})
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
                 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'})
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

    # Visualization area
    html.Div([
        dcc.Tabs(id='tabs', value='qber-tab', children=[
            dcc.Tab(label='QBER Analysis', value='qber-tab', style={'fontWeight': 'bold'}),
            dcc.Tab(label='Timing Analysis', value='timing-tab', style={'fontWeight': 'bold'}),
            dcc.Tab(label='Statistics', value='stats-tab', style={'fontWeight': 'bold'}),
        ]),

        html.Div(id='tabs-content-container', children=[
            html.Div(id='qber-content', children=[
                dcc.Graph(id='qber-graph', style={'height': '450px'})
            ], style={'display': 'block'}),

            html.Div(id='timing-content', children=[
                dcc.Graph(id='timing-graph', style={'height': '450px'})
            ], style={'display': 'none'}),

            html.Div(id='stats-content', children=[
                html.Div(id='detailed-stats', style={'padding': '20px', 'fontSize': '16px'})
            ], style={'display': 'none'})
        ]),

        # Real-time status
        html.Div([
            html.H4("Real-time Status", style={'color': '#2c3e50', 'marginTop': '20px'}),
            html.Div(id='stats-display',
                    style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '8px',
                          'minHeight': '120px', 'marginTop': '10px'})
        ])
    ], style={'width': '68%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

    # Interval for updating data
    dcc.Interval(id='interval-component', interval=200, n_intervals=0)  # Update every 200ms
])

# Simulation thread
class QTASimulationThread(threading.Thread):
    def __init__(self, frames, qubits_per_frame, eve_prob, state):
        super().__init__()
        self.frames = frames
        self.qubits_per_frame = qubits_per_frame
        self.eve_prob = eve_prob
        self.state = state
        self.simulator = AerSimulator()  # Use the Aer Simulator
        self.daemon = True

    def run(self):
        print(f"Starting simulation: {self.frames} frames, {self.qubits_per_frame} qubits/frame")
        self.state.running = True

        for frame in range(self.frames):
            if not self.state.running:
                print("Simulation stopped by user")
                break

            # Simulate quantum frame
            qber, delay, authenticated, eve_count = self.simulate_frame()

            # Update state
            self.state.add_frame_data(qber, delay, authenticated, eve_count)

            # Print progress
            if (frame + 1) % 10 == 0:
                print(f"Frame {frame + 1}/{self.frames} completed")

            time.sleep(0.05)  # Simulate real-time (50ms per frame)

        self.state.running = False
        print("Simulation completed")

    def simulate_frame(self):
        """Simulate one QTA frame using Qiskit"""
        # Alice's preparation
        alice_bases = np.random.choice([0, 1], self.qubits_per_frame)  # 0=Z, 1=X
        alice_bits = np.random.choice([0, 1], self.qubits_per_frame)

        # Create quantum circuit
        qc = QuantumCircuit(self.qubits_per_frame, self.qubits_per_frame)

        # Prepare qubits
        for i in range(self.qubits_per_frame):
            if alice_bits[i] == 1:
                qc.x(i)
            if alice_bases[i] == 1:  # X basis
                qc.h(i)

        # Eve's attack
        eve_intercepts = 0
        for i in range(self.qubits_per_frame):
            if np.random.random() < self.eve_prob:
                eve_intercepts += 1
                # Eve measures in random basis
                eve_basis = np.random.randint(0, 2)
                if eve_basis == 1: # If X basis, apply Hadamard before measurement
                    qc.h(i)
                qc.measure(i, i) # Mid-circuit measurement
                # Eve resends (introduces errors if wrong basis)
                if eve_basis != alice_bases[i]:
                    # Wrong basis - 50% error rate
                    if np.random.random() < 0.5:
                        qc.x(i)
                if eve_basis == 1: # Restore Hadamard if measured in X
                    qc.h(i)

        # Bob's measurement
        bob_bases = np.random.choice([0, 1], self.qubits_per_frame)
        for i in range(self.qubits_per_frame):
            if bob_bases[i] == 1: # If X basis, apply Hadamard before measurement
                qc.h(i)
            qc.measure(i, i) # Final measurement

        # Execute circuit
        qc_transpiled = transpile(qc, self.simulator)  # Use the Aer Simulator
        job = self.simulator.run(qc_transpiled, shots=1)
        result = job.result()
        # Retrieve counts from the result object
        counts = result.get_counts()

        # The result from QasmSimulator.run().result().get_counts() is a dictionary
        # like {'001': 1}, where keys are bitstrings. We need to convert it to a list of ints.
        # Since shots=1, there will be only one key in counts.
        measurement_bitstring = list(counts.keys())[0]
        bob_results = [int(b) for b in measurement_bitstring[::-1]] # Qiskit bitstring is little-endian

        # Calculate QBER (only on matched bases)
        matched_indices = [i for i in range(self.qubits_per_frame)
                          if alice_bases[i] == bob_bases[i]]

        if matched_indices:
            errors = sum(1 for i in matched_indices
                        if alice_bits[i] != bob_results[i])
            qber = errors / len(matched_indices)
        else:
            qber = 0.5  # No matched bases means high error or undefined

        # Timing simulation
        base_delay = 5.0  # ns
        timing_jitter = np.random.normal(0, 0.05)  # Small jitter
        eve_delay = eve_intercepts * 0.5 if eve_intercepts > 0 else 0
        total_delay = base_delay + timing_jitter + eve_delay

        # Authentication decision
        qber_threshold = 0.11
        timing_window = 0.3  # ns
        qber_ok = qber < qber_threshold
        timing_ok = abs(total_delay - base_delay) < timing_window
        authenticated = qber_ok and timing_ok

        return qber, total_delay, authenticated, eve_intercepts

# Callbacks for slider displays
@app.callback(
    Output('frames-display', 'children'),
    Input('frames-slider', 'value')
)
def update_frames_display(value):
    return f"Selected: {value} frames"

@app.callback(
    Output('qubits-display', 'children'),
    Input('qubits-slider', 'value')
)
def update_qubits_display(value):
    return f"Selected: {value} qubits/frame"

@app.callback(
    Output('eve-display', 'children'),
    Input('eve-slider', 'value')
)
def update_eve_display(value):
    return f"Selected: {value*100:.0f}% intercept probability"

# Callback for tab switching
@app.callback(
    [Output('qber-content', 'style'),
     Output('timing-content', 'style'),
     Output('stats-content', 'style')],
    Input('tabs', 'value')
)
def switch_tab(tab):
    qber_style = {'display': 'block'} if tab == 'qber-tab' else {'display': 'none'}
    timing_style = {'display': 'block'} if tab == 'timing-tab' else {'display': 'none'}
    stats_style = {'display': 'block'} if tab == 'stats-tab' else {'display': 'none'}
    return qber_style, timing_style, stats_style

# Callback for start/reset buttons
@app.callback(
    Output('start-button', 'children'),
    [Input('start-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('frames-slider', 'value'),
     State('qubits-slider', 'value'),
     State('eve-slider', 'value')]
)
def control_simulation(start_clicks, reset_clicks, frames, qubits, eve_prob):
    ctx = callback_context

    if not ctx.triggered:
        return 'Start Simulation'

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'reset-button':
        sim_state.running = False
        sim_state.reset()
        print("Simulation reset")
        return 'Start Simulation'

    if button_id == 'start-button':
        if sim_state.running:
            # Stop simulation
            sim_state.running = False
            print("Stopping simulation...")
            return 'Start Simulation'
        else:
            # Start new simulation
            sim_state.reset()
            sim_state.thread = QTASimulationThread(frames, qubits, eve_prob, sim_state)
            sim_state.thread.start()
            return 'Stop Simulation'

    return 'Start Simulation'

# Callback for updating graphs and stats
@app.callback(
    [Output('qber-graph', 'figure'),
     Output('timing-graph', 'figure'),
     Output('stats-display', 'children'),
     Output('detailed-stats', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_all_displays(n):
    data = sim_state.get_data()

    # QBER Graph
    if data['qber_history']:
        qber_fig = {
            'data': [
                go.Scatter(
                    x=list(range(len(data['qber_history']))),
                    y=data['qber_history'],
                    mode='lines+markers',
                    name='QBER',
                    line={'color': '#3498db', 'width': 2},
                    marker={'size': 6, 'color': ['#2ecc71' if auth else '#e74c3c'
                                                 for auth in data['auth_history']]}
                ),
                go.Scatter(
                    x=[0, len(data['qber_history'])-1],
                    y=[0.11, 0.11],
                    mode='lines',
                    name='Threshold (11%)',
                    line={'color': '#e74c3c', 'dash': 'dash', 'width': 2}
                )
            ],
            'layout': {
                'title': 'Quantum Bit Error Rate (QBER)',
                'xaxis': {'title': 'Frame', 'gridcolor': '#ecf0f1'},
                'yaxis': {'title': 'QBER', 'range': [0, max(0.2, max(data['qber_history'])*1.2)],
                         'gridcolor': '#ecf0f1'},
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'hovermode': 'closest',
                'showlegend': True
            }
        }
    else:
        qber_fig = {
            'data': [],
            'layout': {
                'title': 'Waiting for simulation data...',
                'xaxis': {'title': 'Frame'},
                'yaxis': {'title': 'QBER'},
                'plot_bgcolor': 'white'
            }
        }

    # Timing Graph
    if data['delay_history']:
        timing_fig = {
            'data': [
                go.Scatter(
                    x=list(range(len(data['delay_history']))),
                    y=data['delay_history'],
                    mode='lines+markers',
                    name='Delay',
                    line={'color': '#9b59b6', 'width': 2},
                    marker={'size': 6, 'color': ['#2ecc71' if auth else '#e74c3c'
                                                 for auth in data['auth_history']]}
                ),
                go.Scatter(
                    x=[0, len(data['delay_history'])-1],
                    y=[5.0, 5.0],
                    mode='lines',
                    name='Expected (5ns)',
                    line={'color': '#3498db', 'width': 2}
                ),
                go.Scatter(
                    x=[0, len(data['delay_history'])-1],
                    y=[5.3, 5.3],
                    mode='lines',
                    name='Window (+0.3ns)',
                    line={'color': '#e74c3c', 'dash': 'dash', 'width': 1}
                ),
                go.Scatter(
                    x=[0, len(data['delay_history'])-1],
                    y=[4.7, 4.7],
                    mode='lines',
                    name='Window (-0.3ns)',
                    line={'color': '#e74c3c', 'dash': 'dash', 'width': 1}
                )
            ],
            'layout': {
                'title': 'Detection Timing Analysis',
                'xaxis': {'title': 'Frame', 'gridcolor': '#ecf0f1'},
                'yaxis': {'title': 'Delay (ns)', 'gridcolor': '#ecf0f1'},
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'hovermode': 'closest',
                'showlegend': True
            }
        }
    else:
        timing_fig = {
            'data': [],
            'layout': {
                'title': 'Waiting for simulation data...',
                'xaxis': {'title': 'Frame'},
                'yaxis': {'title': 'Delay (ns)'},
                'plot_bgcolor': 'white'
            }
        }

    # Stats Display
    if data['frame'] > 0:
        success_rate = sum(data['auth_history']) / len(data['auth_history']) * 100
        avg_qber = np.mean(data['qber_history'])
        avg_delay = np.mean(data['delay_history'])
        total_eve = sum(data['eve_intercepts'])

        status = "🔄 Running..." if data['running'] else "✅ Completed"

        stats_html = html.Div([
            html.Div([
                html.Span(f"{status}", style={'fontSize': '18px', 'fontWeight': 'bold'}),
                html.Br(),
                html.Span(f"Frame: {data['frame']}", style={'fontSize': '16px'}),
                html.Br(),
                html.Span(f"Success Rate: {success_rate:.1f}%",
                         style={'color': '#2ecc71' if success_rate > 80 else '#e74c3c',
                               'fontSize': '18px', 'fontWeight': 'bold'}),
                html.Br(),
                html.Span(f"Avg QBER: {avg_qber:.4f}", style={'fontSize': '14px'}),
                html.Br(),
                html.Span(f"Avg Delay: {avg_delay:.3f} ns", style={'fontSize': '14px'}),
                html.Br(),
                html.Span(f"Eve Intercepts: {total_eve}",
                         style={'fontSize': '14px', 'color': '#e74c3c' if total_eve > 0 else '#2ecc71'})
            ])
        ])

        # Detailed stats
        detailed_html = html.Div([
            html.H4("Detailed Statistics", style={'color': '#2c3e50'}),
            html.Div([
                html.P(f"Total Frames Processed: {data['frame']}"),
                html.P(f"Authenticated Frames: {sum(data['auth_history'])}"),
                html.P(f"Rejected Frames: {len(data['auth_history']) - sum(data['auth_history'])}"),
                html.P(f"Authentication Success Rate: {success_rate:.2f}%"),
                html.Hr(),
                html.P(f"Average QBER: {avg_qber:.4f}"),
                html.P(f"Min QBER: {min(data['qber_history']):.4f}"),
                html.P(f"Max QBER: {max(data['qber_history']):.4f}"),
                html.Hr(),
                html.P(f"Average Delay: {avg_delay:.3f} ns"),
                html.P(f"Min Delay: {min(data['delay_history']):.3f} ns"),
                html.P(f"Max Delay: {max(data['delay_history']):.3f} ns"),
                html.Hr(),
                html.P(f"Total Eve Interceptions: {total_eve}"),
                html.P(f"Avg Intercepts per Frame: {np.mean(data['eve_intercepts']):.2f}"),
            ], style={'lineHeight': '1.8'})
        ])
    else:
        stats_html = html.Div("Click 'Start Simulation' to begin")
        detailed_html = html.Div("No data available yet")

    return qber_fig, timing_fig, stats_html, detailed_html

# Run the app
if __name__ == '__main__':
    print("Starting Dash QTA Simulator on http://127.0.0.1:8050")
    app.run(debug=True, port=8050)

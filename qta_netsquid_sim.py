# hybrid_qta_architecture.py
"""
Production-Grade QTA Simulator
Custom quantum simulation implementation (no external quantum libraries required)
"""

# Layer 2: Network simulation - Custom network layer
import networkx as nx  # For network topology

# Layer 3: Visualization - Dash/Plotly for web-based GUI
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Layer 4: Analysis - Pandas for data analysis
import pandas as pd

# Layer 5: Security analysis - Custom crypto libraries
from cryptography.hazmat.primitives import hashes

# production_qta_simulator.py
"""
Production-Grade Quantum Temporal Authentication Simulator
Hybrid approach using best tools for each layer
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import json

# Use custom quantum backend
QUANTUM_BACKEND = "Custom"
print("Using custom quantum simulation backend")


# ==================== Configuration ====================

@dataclass
class QTAConfig:
    """Enhanced configuration with persistence"""
    # Simulation parameters
    num_frames: int = 200
    qubits_per_frame: int = 30
    frame_interval_ms: float = 50.0
    
    # Physical parameters
    channel_length_km: float = 10.0
    fiber_loss_db_per_km: float = 0.2
    detector_efficiency: float = 0.7
    detector_dark_count_rate: float = 1e-6
    detector_timing_jitter_ps: float = 50.0
    
    # Temporal authentication parameters
    timing_window_ns: float = 100.0
    expected_delay_ns: float = 50.0
    
    # Security thresholds
    qber_threshold: float = 0.11
    min_detection_rate: float = 0.25
    
    # Attack parameters
    eve_active: bool = True
    eve_strategy: str = "intercept_resend"  # or "photon_number_splitting", "beam_splitting"
    eve_intercept_prob: float = 0.3
    eve_added_delay_ns: float = 500.0
    
    # Advanced features
    enable_decoy_states: bool = True
    decoy_intensities: List[float] = None
    error_correction: bool = True
    privacy_amplification: bool = True
    
    # Backend selection
    quantum_backend: str = QUANTUM_BACKEND
    
    def __post_init__(self):
        if self.decoy_intensities is None:
            self.decoy_intensities = [0.1, 0.3, 0.5]  # Signal, decoy1, decoy2
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


# ==================== Quantum Backend ====================

class QuantumBackend:
    """Abstract base for quantum simulation backends"""
    
    def __init__(self, config: QTAConfig):
        self.config = config
    
    def prepare_qubit(self, bit: int, basis: str) -> any:
        raise NotImplementedError
    
    def measure_qubit(self, qubit: any, basis: str) -> int:
        raise NotImplementedError
    
    def apply_channel_effects(self, qubit: any) -> Tuple[any, bool]:
        """Apply channel loss, returns (qubit, survived)"""
        raise NotImplementedError


class CustomQuantumBackend(QuantumBackend):
    """Custom lightweight quantum backend with realistic physics simulation"""
    
    def prepare_qubit(self, bit: int, basis: str):
        """
        Prepare a qubit in the specified basis
        Returns a dictionary representing the qubit state
        """
        return {
            'bit': bit,
            'basis': basis,
            'phase': 0.0,  # Could be used for advanced simulation
            'fidelity': 1.0  # Perfect preparation initially
        }
    
    def measure_qubit(self, qubit: dict, basis: str) -> int:
        """
        Measure qubit in specified basis
        Implements quantum measurement postulates
        """
        if qubit is None:
            return None
            
        # If measuring in the same basis as preparation
        if qubit['basis'] == basis:
            # Matching basis: should get correct result with high probability
            # Account for any accumulated errors (via fidelity)
            if np.random.random() < qubit['fidelity']:
                return qubit['bit']
            else:
                # Bit flip error
                return 1 - qubit['bit']
        else:
            # Wrong basis: completely random result (50/50)
            # This models the quantum no-cloning theorem
            return np.random.randint(0, 2)
    
    def apply_channel_effects(self, qubit: dict) -> Tuple[dict, bool]:
        """
        Apply realistic channel effects:
        - Fiber attenuation (exponential loss)
        - Depolarization noise
        - Dephasing
        """
        if qubit is None:
            return None, False
        
        # 1. Fiber loss (Beer-Lambert law for optical fibers)
        # Loss coefficient in dB/km converted to probability
        loss_coefficient = self.config.fiber_loss_db_per_km * self.config.channel_length_km
        transmission_prob = 10 ** (-loss_coefficient / 10)
        
        # Check if photon is lost
        if np.random.random() > transmission_prob:
            return None, False
        
        # 2. Depolarization noise (reduces fidelity)
        # Models decoherence during transmission
        depolarization_rate = 0.001 * self.config.channel_length_km  # Small per km
        qubit['fidelity'] *= (1 - depolarization_rate)
        
        # 3. Phase noise (dephasing) - affects superposition states
        # For BB84, this primarily affects X-basis measurements
        if qubit['basis'] == 'X':
            phase_noise = np.random.normal(0, 0.1)  # radians
            qubit['phase'] += phase_noise
            # Phase errors can cause measurement errors
            if abs(qubit['phase']) > np.pi / 4:
                qubit['fidelity'] *= 0.9
        
        return qubit, True


# ==================== Protocol Implementation ====================

class QuantumTemporalAuthentication:
    """Core QTA protocol implementation"""
    
    def __init__(self, config: QTAConfig):
        self.config = config
        self.backend = CustomQuantumBackend(config)
        
        # Statistics tracking
        self.reset_statistics()
    
    def reset_statistics(self):
        self.stats = {
            'frame_results': [],
            'qber_history': [],
            'delay_history': [],
            'detection_rate_history': [],
            'auth_success_history': [],
            'eve_detection_history': [],
            'key_rate_history': []
        }
    
    def simulate_frame(self, frame_id: int) -> Dict:
        """Simulate one authentication frame"""
        start_time = time.time()
        
        # Alice preparation phase
        alice_data = self._alice_prepare_frame()
        
        # Channel transmission with optional Eve attack
        transmitted_data = self._transmit_through_channel(alice_data)
        
        # Bob measurement phase
        bob_data = self._bob_measure_frame(transmitted_data)
        
        # Verification and authentication
        auth_result = self._verify_and_authenticate(alice_data, bob_data)
        
        # Calculate timing
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Compile frame result
        frame_result = {
            'frame_id': frame_id,
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': processing_time,
            **auth_result,
            'alice_stats': self._calculate_alice_stats(alice_data),
            'bob_stats': self._calculate_bob_stats(bob_data),
            'channel_stats': transmitted_data.get('channel_stats', {})
        }
        
        # Update statistics
        self._update_statistics(frame_result)
        
        return frame_result
    
    def _alice_prepare_frame(self) -> Dict:
        """Alice prepares qubits with optional decoy states"""
        frame_data = {
            'qubits': [],
            'bits': [],
            'bases': [],
            'intensities': [],
            'send_times': []
        }
        
        for i in range(self.config.qubits_per_frame):
            bit = np.random.randint(0, 2)
            basis = np.random.choice(['Z', 'X'])
            
            # Decoy state selection
            if self.config.enable_decoy_states:
                intensity = np.random.choice(self.config.decoy_intensities)
            else:
                intensity = self.config.decoy_intensities[0]
            
            qubit = self.backend.prepare_qubit(bit, basis)
            
            frame_data['qubits'].append(qubit)
            frame_data['bits'].append(bit)
            frame_data['bases'].append(basis)
            frame_data['intensities'].append(intensity)
            frame_data['send_times'].append(time.time() * 1e9)  # nanoseconds
        
        return frame_data
    
    def _transmit_through_channel(self, alice_data: Dict) -> Dict:
        """Simulate quantum channel with optional Eve attack"""
        transmitted = {
            'qubits': [],
            'survived': [],
            'eve_intercepted': [],
            'receive_times': [],
            'channel_stats': {}
        }
        
        total_lost = 0
        total_eve_intercepts = 0
        
        for i, qubit in enumerate(alice_data['qubits']):
            # Eve attack simulation
            eve_intercepted = False
            if self.config.eve_active:
                if np.random.random() < self.config.eve_intercept_prob:
                    eve_intercepted = True
                    total_eve_intercepts += 1
                    
                    # Eve measures in random basis (intercept-resend attack)
                    eve_basis = np.random.choice(['Z', 'X'])
                    eve_result = self.backend.measure_qubit(qubit, eve_basis)
                    
                    # Eve resends (prepares new qubit)
                    # This introduces errors when Eve guesses wrong basis
                    if eve_result is not None:
                        qubit = self.backend.prepare_qubit(eve_result, eve_basis)
                    else:
                        qubit = None
            
            # Channel effects (loss, noise, etc.)
            if qubit is not None:
                qubit, survived = self.backend.apply_channel_effects(qubit)
            else:
                survived = False
            
            if not survived:
                total_lost += 1
            
            # Detector dark counts (false positives)
            if not survived and np.random.random() < self.config.detector_dark_count_rate:
                survived = True  # False positive detection
                # Create a "fake" qubit from dark count
                qubit = self.backend.prepare_qubit(np.random.randint(0, 2), 
                                                   np.random.choice(['Z', 'X']))
                qubit['fidelity'] = 0.5  # Mark as unreliable
            
            # Timing calculation
            base_delay = self.config.expected_delay_ns
            
            # Eve's equipment adds delay
            if eve_intercepted:
                base_delay += self.config.eve_added_delay_ns
            
            # Detector timing jitter
            jitter = np.random.normal(0, self.config.detector_timing_jitter_ps)
            
            # Propagation delay (speed of light in fiber ~= 2/3 c)
            propagation_delay = self.config.channel_length_km * 5000  # ns per km
            
            total_delay = base_delay + jitter + propagation_delay
            receive_time = alice_data['send_times'][i] + total_delay
            
            transmitted['qubits'].append(qubit if survived else None)
            transmitted['survived'].append(survived)
            transmitted['eve_intercepted'].append(eve_intercepted)
            transmitted['receive_times'].append(receive_time if survived else None)
        
        transmitted['channel_stats'] = {
            'total_sent': len(alice_data['qubits']),
            'total_lost': total_lost,
            'loss_rate': total_lost / len(alice_data['qubits']) if alice_data['qubits'] else 0,
            'eve_intercepts': total_eve_intercepts,
            'eve_detection_probability': total_eve_intercepts / len(alice_data['qubits']) if alice_data['qubits'] else 0
        }
        
        return transmitted
    
    def _bob_measure_frame(self, transmitted_data: Dict) -> Dict:
        """Bob measures received qubits"""
        bob_data = {
            'results': [],
            'bases': [],
            'receive_times': transmitted_data['receive_times'],
            'detected_count': 0
        }
        
        for qubit in transmitted_data['qubits']:
            if qubit is not None:
                # Bob chooses random measurement basis
                basis = np.random.choice(['Z', 'X'])
                
                # Apply detector efficiency
                if np.random.random() < self.config.detector_efficiency:
                    result = self.backend.measure_qubit(qubit, basis)
                    bob_data['results'].append(result)
                    bob_data['bases'].append(basis)
                    bob_data['detected_count'] += 1
                else:
                    # Detector failed to detect the photon
                    bob_data['results'].append(None)
                    bob_data['bases'].append(None)
            else:
                bob_data['results'].append(None)
                bob_data['bases'].append(None)
        
        return bob_data
    
    def _verify_and_authenticate(self, alice_data: Dict, bob_data: Dict) -> Dict:
        """Verify measurements and make authentication decision"""
        # 1. Sift: keep only matching bases
        sifted_alice = []
        sifted_bob = []
        sifted_times_alice = []
        sifted_times_bob = []
        
        for i in range(len(alice_data['bases'])):
            if i < len(bob_data['results']) and bob_data['results'][i] is not None:
                if alice_data['bases'][i] == bob_data['bases'][i]:
                    sifted_alice.append(alice_data['bits'][i])
                    sifted_bob.append(bob_data['results'][i])
                    sifted_times_alice.append(alice_data['send_times'][i])
                    sifted_times_bob.append(bob_data['receive_times'][i])
        
        # 2. Calculate QBER
        if len(sifted_alice) > 0:
            errors = sum(1 for a, b in zip(sifted_alice, sifted_bob) if a != b)
            qber = errors / len(sifted_alice)
        else:
            qber = 1.0
        
        # 3. Calculate timing statistics
        if len(sifted_times_alice) > 0:
            delays = [b - a for a, b in zip(sifted_times_alice, sifted_times_bob)]
            avg_delay = np.mean(delays)
            std_delay = np.std(delays)
            
            # Calculate expected delay including propagation
            expected_total_delay = (self.config.expected_delay_ns + 
                                   self.config.channel_length_km * 5000)
            
            timing_violations = sum(1 for d in delays 
                                   if abs(d - expected_total_delay) > self.config.timing_window_ns)
        else:
            avg_delay = float('inf')
            std_delay = 0
            timing_violations = self.config.qubits_per_frame
        
        # 4. Calculate detection rate
        detection_rate = bob_data['detected_count'] / self.config.qubits_per_frame
        
        # 5. Authentication criteria
        qber_ok = qber <= self.config.qber_threshold
        timing_ok = timing_violations == 0
        detection_ok = detection_rate >= self.config.min_detection_rate
        
        authenticated = qber_ok and timing_ok and detection_ok
        
        # 6. Estimate secure key rate (simplified Shannon limit)
        if authenticated and self.config.error_correction:
            # Binary entropy function
            def h2(x):
                if x <= 0 or x >= 1:
                    return 0
                return -x * np.log2(x) - (1-x) * np.log2(1-x)
            
            # Secure key rate per sifted bit (GLLP formula approximation)
            if qber < 0.5:
                key_rate_per_bit = 1 - 2 * h2(qber)
            else:
                key_rate_per_bit = 0
            
            key_rate = detection_rate * len(sifted_alice) * max(0, key_rate_per_bit)
        else:
            key_rate = 0
        
        return {
            'authenticated': authenticated,
            'qber': qber,
            'qber_ok': qber_ok,
            'avg_delay_ns': avg_delay if avg_delay != float('inf') else 0,
            'std_delay_ns': std_delay,
            'timing_ok': timing_ok,
            'timing_violations': timing_violations,
            'detection_rate': detection_rate,
            'detection_ok': detection_ok,
            'sifted_key_length': len(sifted_alice),
            'secure_key_rate_bits': key_rate,
            'errors': sum(1 for a, b in zip(sifted_alice, sifted_bob) if a != b) if sifted_alice else 0,
            'total_bits_compared': len(sifted_alice)
        }
    
    def _calculate_alice_stats(self, alice_data: Dict) -> Dict:
        return {
            'total_qubits_sent': len(alice_data['qubits']),
            'basis_distribution': {
                'Z': alice_data['bases'].count('Z'),
                'X': alice_data['bases'].count('X')
            }
        }
    
    def _calculate_bob_stats(self, bob_data: Dict) -> Dict:
        return {
            'total_detected': bob_data['detected_count'],
            'detection_efficiency': bob_data['detected_count'] / len(bob_data['results']) if bob_data['results'] else 0
        }
    
    def _update_statistics(self, frame_result: Dict):
        self.stats['frame_results'].append(frame_result)
        self.stats['qber_history'].append(frame_result['qber'])
        self.stats['delay_history'].append(frame_result['avg_delay_ns'])
        self.stats['detection_rate_history'].append(frame_result['detection_rate'])
        self.stats['auth_success_history'].append(1 if frame_result['authenticated'] else 0)
        self.stats['key_rate_history'].append(frame_result['secure_key_rate_bits'])
    
    def export_results(self, filename: str = "qta_results.json"):
        """Export results to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        print(f"Results exported to {filename}")
    
    def get_summary_statistics(self) -> Dict:
        """Calculate summary statistics"""
        if not self.stats['frame_results']:
            return {
                'total_frames': 0,
                'authentication_success_rate': 0,
                'average_qber': 0,
                'average_detection_rate': 0,
                'average_key_rate_bits_per_frame': 0,
                'total_secure_bits': 0
            }
        
        return {
            'total_frames': len(self.stats['frame_results']),
            'authentication_success_rate': np.mean(self.stats['auth_success_history']),
            'average_qber': np.mean(self.stats['qber_history']),
            'average_detection_rate': np.mean(self.stats['detection_rate_history']),
            'average_key_rate_bits_per_frame': np.mean(self.stats['key_rate_history']),
            'total_secure_bits': sum(self.stats['key_rate_history'])
        }


# ==================== Web-Based Dashboard ====================

class QTADashboard:
    """Interactive web dashboard using Plotly Dash"""
    
    def __init__(self, config: QTAConfig):
        self.config = config
        self.qta = QuantumTemporalAuthentication(config)
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
        self.simulation_running = False
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Create dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🔐 Quantum Temporal Authentication", 
                           className="text-center mb-4 mt-4",
                           style={'color': '#00ff00'})
                ])
            ]),
            
            dbc.Row([
                # Left sidebar - Controls
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Simulation Controls"),
                        dbc.CardBody([
                            html.Label("Number of Frames:"),
                            dcc.Input(id='frames-input', type='number', 
                                     value=self.config.num_frames, className="mb-2"),
                            
                            html.Label("Qubits per Frame:"),
                            dcc.Input(id='qubits-input', type='number', 
                                     value=self.config.qubits_per_frame, className="mb-2"),
                            
                            html.Label("QBER Threshold:"),
                            dcc.Input(id='qber-threshold', type='number', step=0.01,
                                     value=self.config.qber_threshold, className="mb-2"),
                            
                            html.Hr(),
                            
                            dbc.Checklist(
                                options=[{"label": "Enable Eve Attack", "value": 1}],
                                value=[1] if self.config.eve_active else [],
                                id="eve-checkbox",
                                switch=True,
                            ),
                            
                            html.Label("Eve Intercept Probability:"),
                            dcc.Slider(0, 1, 0.1, value=self.config.eve_intercept_prob,
                                      id='eve-prob-slider', className="mb-2"),
                            
                            html.Hr(),
                            
                            dbc.Button("Start Simulation", id="start-btn", 
                                      color="success", className="me-2"),
                            dbc.Button("Stop", id="stop-btn", 
                                      color="danger", className="me-2"),
                            dbc.Button("Export Data", id="export-btn", 
                                      color="info"),
                        ])
                    ], className="mb-3"),
                    
                    dbc.Card([
                        dbc.CardHeader("Live Statistics"),
                        dbc.CardBody(id='stats-display')
                    ])
                ], width=3),
                
                # Main area - Visualizations
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(label="QBER Analysis", tab_id="qber"),
                        dbc.Tab(label="Timing Analysis", tab_id="timing"),
                        dbc.Tab(label="Detection Rate", tab_id="detection"),
                        dbc.Tab(label="Key Generation", tab_id="keyrate"),
                        dbc.Tab(label="Summary", tab_id="summary"),
                    ], id="tabs", active_tab="qber"),
                    html.Div(id="tab-content", className="p-4")
                ], width=9)
            ]),
            
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
            dcc.Store(id='simulation-data', data={'frames': []})
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup Dash callbacks"""
        
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('tabs', 'active_tab'),
            Input('simulation-data', 'data')
        )
        def render_tab_content(active_tab, data):
            if not data or not data.get('frames'):
                return html.Div("No data yet. Start simulation to see results.", 
                               className="text-center")
            
            df = pd.DataFrame(data['frames'])
            
            if active_tab == "qber":
                return self.create_qber_plot(df)
            elif active_tab == "timing":
                return self.create_timing_plot(df)
            elif active_tab == "detection":
                return self.create_detection_plot(df)
            elif active_tab == "keyrate":
                return self.create_keyrate_plot(df)
            elif active_tab == "summary":
                return self.create_summary_view(df)
        
        @self.app.callback(
            Output('simulation-data', 'data'),
            Output('stats-display', 'children'),
            Input('interval-component', 'n_intervals'),
            Input('start-btn', 'n_clicks'),
            State('simulation-data', 'data'),
            prevent_initial_call=True
        )
        def update_simulation(n_intervals, n_clicks, current_data):
            if current_data is None:
                current_data = {'frames': []}
            
            # Simulate one frame
            if len(current_data['frames']) < self.config.num_frames:
                frame_result = self.qta.simulate_frame(len(current_data['frames']))
                current_data['frames'].append(frame_result)
            
            # Generate stats display
            stats_html = self.generate_stats_html(current_data)
            
            return current_data, stats_html
    
    def create_qber_plot(self, df):
        """Create QBER visualization"""
        fig = go.Figure()
        
        colors = ['green' if auth else 'red' for auth in df['authenticated']]
        
        fig.add_trace(go.Scatter(
            x=df['frame_id'],
            y=df['qber'],
            mode='markers+lines',
            marker=dict(color=colors, size=8),
            name='QBER'
        ))
        
        fig.add_hline(y=self.config.qber_threshold, line_dash="dash", 
                     line_color="red", annotation_text="Threshold")
        
        fig.update_layout(
            title="Quantum Bit Error Rate",
            xaxis_title="Frame",
            yaxis_title="QBER",
            template="plotly_dark",
            height=500
        )
        
        return dcc.Graph(figure=fig)
    
    def create_timing_plot(self, df):
        """Create timing analysis visualization"""
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=("Average Delay", "Timing Violations"))
        
        colors = ['green' if auth else 'red' for auth in df['authenticated']]
        
        fig.add_trace(go.Scatter(
            x=df['frame_id'],
            y=df['avg_delay_ns'],
            mode='markers+lines',
            marker=dict(color=colors, size=8),
            name='Avg Delay'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=df['frame_id'],
            y=df['timing_violations'],
            marker=dict(color=colors),
            name='Violations'
        ), row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_dark")
        
        return dcc.Graph(figure=fig)
    
    def create_detection_plot(self, df):
        """Create detection rate visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['frame_id'],
            y=df['detection_rate'],
            mode='markers+lines',
            fill='tozeroy',
            name='Detection Rate'
        ))
        
        fig.add_hline(y=self.config.min_detection_rate, line_dash="dash",
                     annotation_text="Minimum Required")
        
        fig.update_layout(
            title="Qubit Detection Rate",
            xaxis_title="Frame",
            yaxis_title="Detection Rate",
            template="plotly_dark",
            height=500
        )
        
        return dcc.Graph(figure=fig)
    
    def create_keyrate_plot(self, df):
        """Create key generation rate plot"""
        fig = go.Figure()
        
        cumulative_bits = df['secure_key_rate_bits'].cumsum()
        
        fig.add_trace(go.Scatter(
            x=df['frame_id'],
            y=cumulative_bits,
            mode='lines',
            fill='tozeroy',
            name='Cumulative Secure Bits'
        ))
        
        fig.update_layout(
            title="Secure Key Generation",
            xaxis_title="Frame",
            yaxis_title="Cumulative Bits",
            template="plotly_dark",
            height=500
        )
        
        return dcc.Graph(figure=fig)
    
    def create_summary_view(self, df):
        """Create summary statistics view"""
        summary = self.qta.get_summary_statistics()
        
        cards = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{summary.get('total_frames', 0)}", className="text-success"),
                    html.P("Total Frames")
                ])
            ]), width=3),
            
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{summary.get('authentication_success_rate', 0)*100:.1f}%", 
                           className="text-info"),
                    html.P("Success Rate")
                ])
            ]), width=3),
            
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{summary.get('average_qber', 0):.4f}", 
                           className="text-warning"),
                    html.P("Avg QBER")
                ])
            ]), width=3),
            
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{summary.get('total_secure_bits', 0):.0f}", 
                           className="text-primary"),
                    html.P("Total Secure Bits")
                ])
            ]), width=3),
        ])
        
        return html.Div([cards, html.Hr(), self.create_detailed_table(df)])
    
    def create_detailed_table(self, df):
        """Create detailed results table"""
        return dbc.Table.from_dataframe(
            df[['frame_id', 'authenticated', 'qber', 'detection_rate', 
                'secure_key_rate_bits']].head(20),
            striped=True,
            bordered=True,
            hover=True,
            dark=True
        )
    
    def generate_stats_html(self, data):
        """Generate HTML for live statistics"""
        if not data or not data.get('frames'):
            return html.Div("No data")
        
        latest = data['frames'][-1]
        
        return html.Div([
            html.P(f"Frame: {latest['frame_id']}/{self.config.num_frames}"),
            html.P(f"Status: {'✓ Authenticated' if latest['authenticated'] else '✗ Rejected'}",
                  style={'color': 'green' if latest['authenticated'] else 'red'}),
            html.Hr(),
            html.P(f"QBER: {latest['qber']:.4f}"),
            html.P(f"Detection: {latest['detection_rate']:.2%}"),
            html.P(f"Delay: {latest['avg_delay_ns']:.1f} ns"),
        ])
    
    def run(self, debug=False):
        """Run the dashboard"""
        # Use the new API: app.run instead of app.run_server
        self.app.run(debug=debug, port=8050)


# ==================== Main Entry Point ====================

def main():
    """Main entry point"""
    print("="*70)
    print("QUANTUM TEMPORAL AUTHENTICATION - Production Simulator")
    print(f"Quantum Backend: {QUANTUM_BACKEND}")
    print("="*70)
    
    # Create configuration
    config = QTAConfig(
        num_frames=100,
        qubits_per_frame=30,
        eve_active=True,
        enable_decoy_states=True
    )
    
    print(f"\nStarting web dashboard on http://localhost:8050")
    print("Press Ctrl+C to stop\n")
    
    # Create and run dashboard
    dashboard = QTADashboard(config)
    dashboard.run(debug=False)


if __name__ == "__main__":
    main()

    
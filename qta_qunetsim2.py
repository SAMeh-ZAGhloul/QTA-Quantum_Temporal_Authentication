# qta_qunetim_gui.py
"""
Quantum Temporal Authentication (QTA) Simulator
Professional GUI implementation with real-time visualization
Simplified quantum simulation (no QuNetSim required)
FIXED: Eve checkbox now properly controls Eve's attack
"""

import sys
import numpy as np
from threading import Thread, Lock
import time
from collections import deque
from datetime import datetime

# PyQt5 for GUI
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                  QHBoxLayout, QPushButton, QLabel, QGroupBox, 
                                  QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
                                  QGridLayout, QTabWidget, QProgressBar)
    from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
    from PyQt5.QtGui import QFont, QPalette, QColor
except ImportError:
    print("Error: PyQt5 not installed. Install with: pip install PyQt5")
    sys.exit(1)

# Matplotlib for embedded plots
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ==================== Configuration ====================

class QTAConfig:
    """Configuration for QTA simulation"""
    def __init__(self):
        # Simulation parameters
        self.num_frames = 100
        self.qubits_per_frame = 20
        self.frame_interval_ms = 100  # Time between frames
        
        # Timing parameters (in milliseconds for simulation)
        self.timing_window_ms = 10.0
        self.expected_delay_ms = 5.0
        
        # Security parameters
        self.qber_threshold = 0.11  # 11% is typical BB84 threshold
        self.min_detection_rate = 0.3  # At least 30% qubits detected
        
        # Eve parameters
        self.eve_active = True
        self.eve_intercept_prob = 0.3
        self.eve_delay_ms = 15.0  # Eve causes detectable delay
        
        # Channel parameters
        self.channel_loss_prob = 0.2
        self.detector_efficiency = 0.85
        self.measurement_error_rate = 0.01
        
        # Basis choices (computational and Hadamard)
        self.bases = ['Z', 'X']


# ==================== Simplified Qubit Class ====================

class SimpleQubit:
    """Simplified qubit representation"""
    def __init__(self, bit, basis):
        self.bit = bit
        self.basis = basis
    
    def measure(self, measurement_basis):
        """Measure qubit in given basis"""
        if measurement_basis == self.basis:
            # Matching basis - correct result (with small error)
            if np.random.random() > 0.01:  # 99% accurate
                return self.bit
            else:
                return 1 - self.bit
        else:
            # Non-matching basis - random result
            return np.random.randint(0, 2)


# ==================== QTA Protocol Implementation ====================

class QTAProtocol:
    """Implements Quantum Temporal Authentication protocol"""
    
    def __init__(self, config: QTAConfig):
        self.config = config
        self.lock = Lock()
        
        # Statistics
        self.stats = {
            'frames_completed': 0,
            'total_qubits_sent': 0,
            'total_qubits_received': 0,
            'total_authenticated': 0,
            'qber_history': deque(maxlen=100),
            'delay_history': deque(maxlen=100),
            'detection_rate_history': deque(maxlen=100),
            'auth_history': deque(maxlen=100),
            'eve_intercept_history': deque(maxlen=100)
        }
    
    def alice_prepare_frame(self):
        """Alice prepares a frame of qubits"""
        frame_data = {
            'qubits': [],
            'bits': [],
            'bases': [],
            'send_times': []
        }
        
        for _ in range(self.config.qubits_per_frame):
            bit = np.random.randint(0, 2)
            basis = np.random.choice(self.config.bases)
            qubit = SimpleQubit(bit, basis)
            
            frame_data['qubits'].append(qubit)
            frame_data['bits'].append(bit)
            frame_data['bases'].append(basis)
            frame_data['send_times'].append(time.time() * 1000)  # ms
        
        return frame_data
    
    def simulate_channel(self, qubits):
        """Simulate quantum channel with loss"""
        transmitted = []
        for qubit in qubits:
            # Channel loss
            if np.random.random() > self.config.channel_loss_prob:
                transmitted.append(qubit)
            else:
                transmitted.append(None)
        return transmitted
    
    def eve_attack(self, qubits, alice_data):
        """Simulate Eve's intercept-resend attack"""
        intercepted_indices = []
        
        # Check if Eve is active RIGHT NOW (not cached value)
        if not self.config.eve_active:
            return qubits, intercepted_indices
        
        modified_qubits = []
        for i, qubit in enumerate(qubits):
            if qubit is None:
                modified_qubits.append(None)
                continue
            
            # Eve intercepts with probability
            if np.random.random() < self.config.eve_intercept_prob:
                intercepted_indices.append(i)
                
                # Eve measures in random basis
                eve_basis = np.random.choice(self.config.bases)
                eve_result = qubit.measure(eve_basis)
                
                # Eve resends new qubit
                new_qubit = SimpleQubit(eve_result, eve_basis)
                modified_qubits.append(new_qubit)
            else:
                modified_qubits.append(qubit)
        
        return modified_qubits, intercepted_indices
    
    def bob_measure_frame(self, qubits, receive_times):
        """Bob measures received qubits"""
        measurements = {
            'results': [],
            'bases': [],
            'receive_times': receive_times,
            'measured_count': 0
        }
        
        for qubit in qubits:
            if qubit is not None:
                # Detector efficiency
                if np.random.random() < self.config.detector_efficiency:
                    basis = np.random.choice(self.config.bases)
                    result = qubit.measure(basis)
                    measurements['results'].append(result)
                    measurements['bases'].append(basis)
                    measurements['measured_count'] += 1
                else:
                    measurements['results'].append(None)
                    measurements['bases'].append(None)
            else:
                measurements['results'].append(None)
                measurements['bases'].append(None)
        
        return measurements
    
    def verify_frame(self, alice_data, bob_data):
        """Verify frame and make authentication decision"""
        # 1. Calculate QBER on matching bases
        matched_indices = []
        errors = 0
        
        for i in range(len(alice_data['bases'])):
            if i < len(bob_data['results']) and bob_data['results'][i] is not None:
                if alice_data['bases'][i] == bob_data['bases'][i]:
                    matched_indices.append(i)
                    if alice_data['bits'][i] != bob_data['results'][i]:
                        errors += 1
        
        qber = errors / len(matched_indices) if matched_indices else 1.0
        
        # 2. Calculate timing delays
        delays = []
        for i in range(min(len(alice_data['send_times']), len(bob_data['receive_times']))):
            if bob_data['receive_times'][i] is not None:
                delay = bob_data['receive_times'][i] - alice_data['send_times'][i]
                delays.append(delay)
        
        avg_delay = np.mean(delays) if delays else float('inf')
        
        # 3. Calculate detection rate
        detection_rate = bob_data['measured_count'] / self.config.qubits_per_frame
        
        # 4. Authentication decision
        qber_ok = qber <= self.config.qber_threshold
        timing_ok = abs(avg_delay - self.config.expected_delay_ms) <= self.config.timing_window_ms
        detection_ok = detection_rate >= self.config.min_detection_rate
        
        authenticated = qber_ok and timing_ok and detection_ok
        
        # 5. Update statistics
        with self.lock:
            self.stats['frames_completed'] += 1
            self.stats['total_qubits_sent'] += self.config.qubits_per_frame
            self.stats['total_qubits_received'] += bob_data['measured_count']
            if authenticated:
                self.stats['total_authenticated'] += 1
            
            self.stats['qber_history'].append(qber)
            self.stats['delay_history'].append(avg_delay)
            self.stats['detection_rate_history'].append(detection_rate)
            self.stats['auth_history'].append(1 if authenticated else 0)
        
        return {
            'authenticated': authenticated,
            'qber': qber,
            'avg_delay': avg_delay,
            'detection_rate': detection_rate,
            'matched_count': len(matched_indices),
            'qber_ok': qber_ok,
            'timing_ok': timing_ok,
            'detection_ok': detection_ok
        }


# ==================== Simulation Engine ====================

class QTASimulation:
    """Manages QTA simulation"""
    
    def __init__(self, config: QTAConfig, protocol: QTAProtocol, signals):
        self.config = config
        self.protocol = protocol
        self.signals = signals
        self.running = False
    
    def run_frame(self):
        """Execute one QTA frame"""
        # Alice prepares frame
        alice_data = self.protocol.alice_prepare_frame()
        
        # Channel transmission
        transmitted_qubits = self.protocol.simulate_channel(alice_data['qubits'])
        
        # Eve attack simulation (checks config.eve_active inside)
        qubits_after_eve, intercepted = self.protocol.eve_attack(transmitted_qubits, alice_data)
        
        # Eve's attack introduces delay ONLY IF EVE IS ACTIVE
        if intercepted and self.config.eve_active:
            time.sleep(self.config.eve_delay_ms / 1000.0)
        
        # Simulate transmission delay
        time.sleep(self.config.expected_delay_ms / 1000.0)
        
        # Bob receives and measures
        receive_times = [time.time() * 1000] * len(qubits_after_eve)
        bob_data = self.protocol.bob_measure_frame(qubits_after_eve, receive_times)
        
        # Verification and authentication
        result = self.protocol.verify_frame(alice_data, bob_data)
        result['eve_intercepted'] = len(intercepted)
        result['eve_active'] = self.config.eve_active  # Add current Eve status
        
        # Store Eve intercepts in stats
        self.protocol.stats['eve_intercept_history'].append(len(intercepted))
        
        # Emit signal to update GUI
        self.signals.frame_completed.emit(result)
        
        return result
    
    def run_simulation(self):
        """Run complete simulation"""
        self.running = True
        
        for frame_idx in range(self.config.num_frames):
            if not self.running:
                break
            
            self.run_frame()
            time.sleep(self.config.frame_interval_ms / 1000.0)
        
        self.signals.simulation_completed.emit()
    
    def stop(self):
        """Stop simulation"""
        self.running = False


# ==================== GUI Signals ====================

class GUISignals(QObject):
    """Qt signals for thread-safe GUI updates"""
    frame_completed = pyqtSignal(dict)
    simulation_completed = pyqtSignal()


# ==================== Matplotlib Canvas Widgets ====================

class QBERCanvas(FigureCanvas):
    """Canvas for QBER plot"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('QBER')
        self.ax.set_title('Quantum Bit Error Rate')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(0, 0.3)
        
    def update_plot(self, qber_history, auth_history, threshold):
        self.ax.clear()
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('QBER')
        self.ax.set_title('Quantum Bit Error Rate')
        self.ax.grid(True, alpha=0.3)
        
        if qber_history:
            frames = list(range(len(qber_history)))
            colors = ['green' if a else 'red' for a in auth_history]
            
            self.ax.scatter(frames, qber_history, c=colors, s=30, alpha=0.6)
            self.ax.plot(frames, qber_history, alpha=0.3, color='blue')
            self.ax.axhline(threshold, color='red', linestyle='--', 
                           label=f'Threshold: {threshold}', alpha=0.7)
            self.ax.legend()
            self.ax.set_ylim(0, max(0.3, max(qber_history) * 1.2))
        
        self.draw()


class TimingCanvas(FigureCanvas):
    """Canvas for timing delay plot"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Delay (ms)')
        self.ax.set_title('Detection Timing Delays')
        self.ax.grid(True, alpha=0.3)
        
    def update_plot(self, delay_history, auth_history, expected, window):
        self.ax.clear()
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Delay (ms)')
        self.ax.set_title('Detection Timing Delays')
        self.ax.grid(True, alpha=0.3)
        
        if delay_history:
            frames = list(range(len(delay_history)))
            colors = ['green' if a else 'red' for a in auth_history]
            
            self.ax.scatter(frames, delay_history, c=colors, s=30, alpha=0.6)
            self.ax.plot(frames, delay_history, alpha=0.3, color='blue')
            
            self.ax.axhline(expected, color='gray', linestyle='-', 
                           label='Expected', alpha=0.5)
            self.ax.axhline(expected + window, color='red', linestyle='--', 
                           label=f'Window: ±{window} ms', alpha=0.7)
            self.ax.axhline(expected - window, color='red', linestyle='--', alpha=0.7)
            
            self.ax.legend()
        
        self.draw()


class DetectionCanvas(FigureCanvas):
    """Canvas for detection rate plot"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Detection Rate')
        self.ax.set_title('Qubit Detection Rate')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(0, 1.1)
        
    def update_plot(self, detection_history, auth_history, threshold):
        self.ax.clear()
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Detection Rate')
        self.ax.set_title('Qubit Detection Rate')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(0, 1.1)
        
        if detection_history:
            frames = list(range(len(detection_history)))
            colors = ['green' if a else 'red' for a in auth_history]
            
            self.ax.scatter(frames, detection_history, c=colors, s=30, alpha=0.6)
            self.ax.plot(frames, detection_history, alpha=0.3, color='blue')
            self.ax.axhline(threshold, color='red', linestyle='--', 
                           label=f'Min: {threshold}', alpha=0.7)
            self.ax.legend()
        
        self.draw()


# ==================== Main GUI Window ====================

class QTASimulatorGUI(QMainWindow):
    """Main GUI window for QTA simulator"""
    
    def __init__(self):
        super().__init__()
        
        self.config = QTAConfig()
        self.protocol = QTAProtocol(self.config)
        self.signals = GUISignals()
        self.simulation = None
        self.simulation_thread = None
        
        # Connect signals
        self.signals.frame_completed.connect(self.on_frame_completed)
        self.signals.simulation_completed.connect(self.on_simulation_completed)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle('Quantum Temporal Authentication (QTA) Simulator')
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Visualizations
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, 3)
        
    def apply_dark_theme(self):
        """Apply dark color theme"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)
        
    def create_control_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel('QTA Simulator Control')
        title.setFont(QFont('Arial', 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Configuration group
        config_group = QGroupBox('Simulation Parameters')
        config_layout = QGridLayout()
        
        row = 0
        
        # Number of frames
        config_layout.addWidget(QLabel('Frames:'), row, 0)
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(10, 1000)
        self.frames_spin.setValue(self.config.num_frames)
        config_layout.addWidget(self.frames_spin, row, 1)
        row += 1
        
        # Qubits per frame
        config_layout.addWidget(QLabel('Qubits/Frame:'), row, 0)
        self.qubits_spin = QSpinBox()
        self.qubits_spin.setRange(5, 100)
        self.qubits_spin.setValue(self.config.qubits_per_frame)
        config_layout.addWidget(self.qubits_spin, row, 1)
        row += 1
        
        # Frame interval
        config_layout.addWidget(QLabel('Frame Interval (ms):'), row, 0)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(10, 1000)
        self.interval_spin.setValue(self.config.frame_interval_ms)
        config_layout.addWidget(self.interval_spin, row, 1)
        row += 1
        
        # QBER threshold
        config_layout.addWidget(QLabel('QBER Threshold:'), row, 0)
        self.qber_spin = QDoubleSpinBox()
        self.qber_spin.setRange(0.01, 0.5)
        self.qber_spin.setSingleStep(0.01)
        self.qber_spin.setValue(self.config.qber_threshold)
        config_layout.addWidget(self.qber_spin, row, 1)
        row += 1
        
        # Timing window
        config_layout.addWidget(QLabel('Timing Window (ms):'), row, 0)
        self.timing_spin = QDoubleSpinBox()
        self.timing_spin.setRange(1.0, 100.0)
        self.timing_spin.setValue(self.config.timing_window_ms)
        config_layout.addWidget(self.timing_spin, row, 1)
        row += 1
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Eve configuration
        eve_group = QGroupBox('Attacker (Eve) Configuration')
        eve_layout = QGridLayout()
        
        self.eve_checkbox = QCheckBox('Enable Eve Attack')
        self.eve_checkbox.setChecked(self.config.eve_active)
        # Connect checkbox to live update - THIS IS THE KEY FIX
        self.eve_checkbox.stateChanged.connect(self.on_eve_checkbox_changed)
        eve_layout.addWidget(self.eve_checkbox, 0, 0, 1, 2)
        
        # Eve status indicator
        self.eve_status_label = QLabel('Status: ACTIVE' if self.config.eve_active else 'Status: INACTIVE')
        self.eve_status_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.eve_status_label.setStyleSheet('color: red;' if self.config.eve_active else 'color: green;')
        eve_layout.addWidget(self.eve_status_label, 1, 0, 1, 2)
        
        eve_layout.addWidget(QLabel('Intercept Probability:'), 2, 0)
        self.eve_prob_spin = QDoubleSpinBox()
        self.eve_prob_spin.setRange(0.0, 1.0)
        self.eve_prob_spin.setSingleStep(0.1)
        self.eve_prob_spin.setValue(self.config.eve_intercept_prob)
        self.eve_prob_spin.valueChanged.connect(self.on_eve_prob_changed)
        eve_layout.addWidget(self.eve_prob_spin, 2, 1)
        
        eve_layout.addWidget(QLabel('Eve Delay (ms):'), 3, 0)
        self.eve_delay_spin = QDoubleSpinBox()
        self.eve_delay_spin.setRange(0.0, 100.0)
        self.eve_delay_spin.setValue(self.config.eve_delay_ms)
        self.eve_delay_spin.valueChanged.connect(self.on_eve_delay_changed)
        eve_layout.addWidget(self.eve_delay_spin, 3, 1)
        
        eve_group.setLayout(eve_layout)
        layout.addWidget(eve_group)
        
        # Control buttons
        button_layout = QVBoxLayout()
        
        self.start_btn = QPushButton('Start Simulation')
        self.start_btn.setFont(QFont('Arial', 12, QFont.Bold))
        self.start_btn.clicked.connect(self.start_simulation)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton('Stop Simulation')
        self.stop_btn.setFont(QFont('Arial', 12, QFont.Bold))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_simulation)
        button_layout.addWidget(self.stop_btn)
        
        self.reset_btn = QPushButton('Reset')
        self.reset_btn.clicked.connect(self.reset_simulation)
        button_layout.addWidget(self.reset_btn)
        
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Statistics display
        stats_group = QGroupBox('Real-Time Statistics')
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        stats_layout.addWidget(self.stats_text)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        return panel
    
    def on_eve_checkbox_changed(self, state):
        """Handle Eve checkbox state change - LIVE UPDATE"""
        self.config.eve_active = (state == Qt.Checked)
        status_text = 'Status: ACTIVE' if self.config.eve_active else 'Status: INACTIVE'
        status_color = 'color: red;' if self.config.eve_active else 'color: green;'
        self.eve_status_label.setText(status_text)
        self.eve_status_label.setStyleSheet(status_color)
        print(f"Eve attack {'ENABLED' if self.config.eve_active else 'DISABLED'}")
    
    def on_eve_prob_changed(self, value):
        """Handle Eve intercept probability change"""
        self.config.eve_intercept_prob = value
    
    def on_eve_delay_changed(self, value):
        """Handle Eve delay change"""
        self.config.eve_delay_ms = value
    
    def create_visualization_panel(self):
        """Create right visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different visualizations
        tabs = QTabWidget()
        
        # QBER tab
        qber_tab = QWidget()
        qber_layout = QVBoxLayout(qber_tab)
        self.qber_canvas = QBERCanvas()
        qber_layout.addWidget(self.qber_canvas)
        tabs.addTab(qber_tab, 'QBER Analysis')
        
        # Timing tab
        timing_tab = QWidget()
        timing_layout = QVBoxLayout(timing_tab)
        self.timing_canvas = TimingCanvas()
        timing_layout.addWidget(self.timing_canvas)
        tabs.addTab(timing_tab, 'Timing Analysis')
        
        # Detection tab
        detection_tab = QWidget()
        detection_layout = QVBoxLayout(detection_tab)
        self.detection_canvas = DetectionCanvas()
        detection_layout.addWidget(self.detection_canvas)
        tabs.addTab(detection_tab, 'Detection Rate')
        
        layout.addWidget(tabs)
        
        # Status display
        status_group = QGroupBox('Current Frame Status')
        status_layout = QGridLayout()
        
        self.frame_label = QLabel('Frame: 0')
        self.frame_label.setFont(QFont('Arial', 12))
        status_layout.addWidget(self.frame_label, 0, 0)
        
        self.auth_label = QLabel('Status: Ready')
        self.auth_label.setFont(QFont('Arial', 12, QFont.Bold))
        status_layout.addWidget(self.auth_label, 0, 1)
        
        self.qber_label = QLabel('QBER: -')
        status_layout.addWidget(self.qber_label, 1, 0)
        
        self.delay_label = QLabel('Delay: -')
        status_layout.addWidget(self.delay_label, 1, 1)
        
        self.detection_label = QLabel('Detection: -')
        status_layout.addWidget(self.detection_label, 2, 0)
        
        self.eve_label = QLabel('Eve Intercepts: -')
        status_layout.addWidget(self.eve_label, 2, 1)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        return panel
    
    def update_config_from_gui(self):
        """Update configuration from GUI inputs"""
        self.config.num_frames = self.frames_spin.value()
        self.config.qubits_per_frame = self.qubits_spin.value()
        self.config.frame_interval_ms = self.interval_spin.value()
        self.config.qber_threshold = self.qber_spin.value()
        self.config.timing_window_ms = self.timing_spin.value()
        # Eve config is updated live, no need to set here
    
    def start_simulation(self):
        """Start simulation in background thread"""
        self.update_config_from_gui()
        
        # Reset protocol
        self.protocol = QTAProtocol(self.config)
        
        # Create simulation
        self.simulation = QTASimulation(self.config, self.protocol, self.signals)
        
        # Start simulation thread
        self.simulation_thread = Thread(target=self.simulation.run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.auth_label.setText('Status: Running...')
        self.auth_label.setStyleSheet('color: yellow;')
        self.progress_bar.setMaximum(self.config.num_frames)
        self.progress_bar.setValue(0)
        
    def stop_simulation(self):
        """Stop running simulation"""
        if self.simulation:
            self.simulation.stop()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.auth_label.setText('Status: Stopped')
        self.auth_label.setStyleSheet('color: orange;')
    
    def reset_simulation(self):
        """Reset simulation and clear data"""
        self.stop_simulation()
        
        self.protocol = QTAProtocol(self.config)
        self.progress_bar.setValue(0)
        self.frame_label.setText('Frame: 0')
        self.auth_label.setText('Status: Ready')
        self.auth_label.setStyleSheet('')
        self.qber_label.setText('QBER: -')
        self.delay_label.setText('Delay: -')
        self.detection_label.setText('Detection: -')
        self.eve_label.setText('Eve Intercepts: -')
        self.stats_text.clear()
        
        # Clear plots
        self.qber_canvas.update_plot([], [], self.config.qber_threshold)
        self.timing_canvas.update_plot([], [], self.config.expected_delay_ms, 
                                      self.config.timing_window_ms)
        self.detection_canvas.update_plot([], [], self.config.min_detection_rate)
    
    def on_frame_completed(self, result):
        """Handle frame completion event"""
        frame = self.protocol.stats['frames_completed']
        
        # Update progress
        self.progress_bar.setValue(frame)
        self.frame_label.setText(f'Frame: {frame}')
        
        # Update status labels
        if result['authenticated']:
            self.auth_label.setText('Status: [AUTHENTICATED]')
            self.auth_label.setStyleSheet('color: green; font-weight: bold;')
        else:
            self.auth_label.setText('Status: [REJECTED]')
            self.auth_label.setStyleSheet('color: red; font-weight: bold;')
        
        check_ok = '[OK]' if result['qber_ok'] else '[FAIL]'
        self.qber_label.setText(f"QBER: {result['qber']:.4f} {check_ok}")
        
        check_timing = '[OK]' if result['timing_ok'] else '[FAIL]'
        self.delay_label.setText(f"Delay: {result['avg_delay']:.2f} ms {check_timing}")
        
        check_det = '[OK]' if result['detection_ok'] else '[FAIL]'
        self.detection_label.setText(f"Detection: {result['detection_rate']:.2%} {check_det}")
        
        # Show Eve status in intercepts label
        eve_status = "ACTIVE" if result.get('eve_active', False) else "INACTIVE"
        self.eve_label.setText(f"Eve: {eve_status} | Intercepts: {result.get('eve_intercepted', 0)}")
        
        # Update statistics text
        stats = self.protocol.stats
        success_rate = stats['total_authenticated'] / stats['frames_completed'] * 100 if stats['frames_completed'] > 0 else 0
        
        eve_indicator = "[ACTIVE]" if self.config.eve_active else "[INACTIVE]"
        
        stats_text = f"""
=== CUMULATIVE STATISTICS ===
Frames Completed: {stats['frames_completed']}/{self.config.num_frames}
Authentication Success Rate: {success_rate:.1f}%

Eve Status: {eve_indicator}

Qubits Sent: {stats['total_qubits_sent']}
Qubits Received: {stats['total_qubits_received']}
Detection Rate: {stats['total_qubits_received']/max(1, stats['total_qubits_sent'])*100:.1f}%

Average QBER: {np.mean(stats['qber_history']):.4f}
Average Delay: {np.mean(stats['delay_history']):.2f} ms

Current Frame Details:
- Matched Bases: {result['matched_count']}
- QBER: {result['qber']:.4f} (Threshold: {self.config.qber_threshold})
- Delay: {result['avg_delay']:.2f} ms (Window: +/-{self.config.timing_window_ms} ms)
- Detection: {result['detection_rate']:.2%} (Min: {self.config.min_detection_rate:.0%})
- Eve Intercepts This Frame: {result.get('eve_intercepted', 0)}
"""
        self.stats_text.setText(stats_text)
        
        # Update plots
        self.qber_canvas.update_plot(
            list(self.protocol.stats['qber_history']),
            list(self.protocol.stats['auth_history']),
            self.config.qber_threshold
        )
        
        self.timing_canvas.update_plot(
            list(self.protocol.stats['delay_history']),
            list(self.protocol.stats['auth_history']),
            self.config.expected_delay_ms,
            self.config.timing_window_ms
        )
        
        self.detection_canvas.update_plot(
            list(self.protocol.stats['detection_rate_history']),
            list(self.protocol.stats['auth_history']),
            self.config.min_detection_rate
        )
    
    def on_simulation_completed(self):
        """Handle simulation completion"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.auth_label.setText('Status: Completed')
        self.auth_label.setStyleSheet('color: cyan; font-weight: bold;')
        
        # Show final summary
        stats = self.protocol.stats
        success_rate = stats['total_authenticated'] / stats['frames_completed'] * 100 if stats['frames_completed'] > 0 else 0
        
        summary = f"""
========================================
   SIMULATION COMPLETED SUCCESSFULLY    
========================================

Final Results:
----------------------------------------
Frames Simulated: {stats['frames_completed']}
Authentication Success Rate: {success_rate:.1f}%
Successful Authentications: {stats['total_authenticated']}/{stats['frames_completed']}

Quantum Channel Statistics:
----------------------------------------
Total Qubits Transmitted: {stats['total_qubits_sent']}
Total Qubits Detected: {stats['total_qubits_received']}
Overall Detection Rate: {stats['total_qubits_received']/max(1, stats['total_qubits_sent'])*100:.1f}%

Security Metrics:
----------------------------------------
Average QBER: {np.mean(stats['qber_history']):.4f}
QBER Threshold: {self.config.qber_threshold}
QBER Violation Rate: {sum(1 for q in stats['qber_history'] if q > self.config.qber_threshold)/max(1, len(stats['qber_history']))*100:.1f}%

Timing Analysis:
----------------------------------------
Average Delay: {np.mean(stats['delay_history']):.2f} ms
Expected Delay: {self.config.expected_delay_ms} ms
Timing Window: +/-{self.config.timing_window_ms} ms

Eve Attack Analysis:
----------------------------------------
Eve Active: {'Yes' if self.config.eve_active else 'No'}
Total Intercepts: {sum(stats['eve_intercept_history'])}
Avg Intercepts/Frame: {np.mean(stats['eve_intercept_history']):.1f}
Attack Detection: {'Successful - Low Auth Rate' if success_rate < 80 else 'Partial - Some Frames Rejected'}
"""
        
        self.stats_text.setText(summary)


# ==================== Main Entry Point ====================

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = QTASimulatorGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

    
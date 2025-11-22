import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import random
import hashlib
import hmac
import json
from collections import deque
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import threading
import queue

# Constants based on the protocol specification
PULSES_PER_FRAME = 1024
FRAME_DURATION_MS = 1
TIMING_WINDOW_PS = 50  # Timing window in picoseconds
QBER_THRESHOLD = 0.05  # 5% QBER threshold
MIN_SUCCESS_RATE = 0.95  # Minimum success rate for authentication
PHOTON_LOSS_PROB = 0.2  # Probability of photon loss in the channel

class Basis(Enum):
    Z = 0  # Computational basis (|0⟩, |1⟩)
    X = 1  # Hadamard basis (|+⟩, |-⟩)

@dataclass
class QuantumState:
    """Represents a quantum state with basis and value"""
    basis: Basis
    value: int  # 0 or 1
    
    def measure(self, basis: Basis) -> Tuple[int, bool]:
        """
        Measure the quantum state in the given basis.
        Returns the measurement result and whether the basis matched.
        """
        if self.basis == basis:
            # Same basis: deterministic measurement
            return self.value, True
        else:
            # Different basis: random measurement (50/50)
            return random.randint(0, 1), False

@dataclass
class Pulse:
    """Represents a single quantum pulse with emission time"""
    state: QuantumState
    emission_time_ps: int  # Emission time in picoseconds
    detected: bool = False
    detection_time_ps: Optional[int] = None
    measurement_basis: Optional[Basis] = None
    measurement_result: Optional[int] = None

class TrafficEvent:
    """Represents a traffic event in the QTA protocol"""
    def __init__(self, sender, receiver, event_type, timestamp=None, details=None):
        self.sender = sender
        self.receiver = receiver
        self.event_type = event_type  # 'quantum', 'classical', 'intercept'
        self.timestamp = timestamp if timestamp else time.time()
        self.details = details if details else {}

class QuantumChannel:
    """Simulates a quantum communication channel with potential eavesdropping"""
    
    def __init__(self, loss_prob=PHOTON_LOSS_PROB):
        self.loss_prob = loss_prob
        self.eve_active = False
        self.eve_intercept_prob = 0.3  # Probability Eve attempts to intercept
        self.traffic_log = []  # Log of traffic events
        self.buffer = queue.Queue()  # Buffer for transmitted pulses
        
    def transmit(self, pulses: List[Pulse], eve=None, sender="Alice", receiver="Bob") -> List[Pulse]:
        """Transmit pulses through the quantum channel, with potential eavesdropping"""
        received_pulses = []
        
        # Log transmission event
        self.traffic_log.append(TrafficEvent(
            sender=sender,
            receiver=receiver,
            event_type='quantum',
            details={'pulse_count': len(pulses)}
        ))
        
        for pulse in pulses:
            # Simulate photon loss
            if random.random() < self.loss_prob:
                continue
                
            # Eve's interception attempt
            if self.eve_active and eve and random.random() < self.eve_intercept_prob:
                # Log interception
                self.traffic_log.append(TrafficEvent(
                    sender="Eve",
                    receiver=receiver,
                    event_type='intercept',
                    details={'original_sender': sender}
                ))
                pulse = eve.intercept(pulse)
                
            received_pulses.append(pulse)
            
        # Store in buffer for receiver to retrieve
        self.buffer.put({
            'receiver': receiver,
            'pulses': received_pulses
        })
            
        return received_pulses
        
    def receive(self, receiver):
        """Receive pulses intended for the specified receiver"""
        received_pulses = []
        temp_items = []
        
        # Get all items from buffer
        while not self.buffer.empty():
            item = self.buffer.get()
            if item['receiver'] == receiver:
                received_pulses.extend(item['pulses'])
            else:
                temp_items.append(item)
                
        # Put back items not intended for this receiver
        for item in temp_items:
            self.buffer.put(item)
            
        return received_pulses

class ClassicalChannel:
    """Simulates a classical communication channel for control messages"""
    
    def __init__(self):
        self.messages = queue.Queue()
        self.traffic_log = []  # Log of traffic events
        
    def send(self, sender, receiver, message):
        """Send a message from sender to receiver"""
        timestamp = time.time()
        self.messages.put({
            'sender': sender,
            'receiver': receiver,
            'message': message,
            'timestamp': timestamp
        })
        
        # Log transmission event
        self.traffic_log.append(TrafficEvent(
            sender=sender,
            receiver=receiver,
            event_type='classical',
            timestamp=timestamp,
            details={'message_type': message.get('type', 'unknown')}
        ))
        
    def receive(self, receiver):
        """Receive messages intended for the specified receiver"""
        received = []
        temp_messages = []
        
        while not self.messages.empty():
            msg = self.messages.get()
            if msg['receiver'] == receiver:
                received.append(msg)
            else:
                temp_messages.append(msg)
                
        # Put back messages not intended for this receiver
        for msg in temp_messages:
            self.messages.put(msg)
            
        return received

class Alice:
    """Server in the QTA protocol"""
    
    def __init__(self, classical_channel, quantum_channel, eve):
        self.classical_channel = classical_channel
        self.quantum_channel = quantum_channel
        self.eve = eve # Keep a reference to Eve
        self.frame_id = 0
        self.nonce = None
        self.current_frame = []
        self.authenticated = False
        self.history = {
            'qber': deque(maxlen=20),
            'success_rate': deque(maxlen=20),
            'timestamps': deque(maxlen=20)
        }
        
    def generate_nonce(self):
        """Generate a new nonce for the current frame"""
        self.nonce = hashlib.sha256(f"{self.frame_id}-{time.time()}".encode()).hexdigest()
        return self.nonce
        
    def create_quantum_challenge(self):
        """Create a new quantum challenge frame"""
        self.frame_id += 1
        self.nonce = self.generate_nonce()
        self.current_frame = []
        
        # Generate random quantum states
        for i in range(PULSES_PER_FRAME):
            basis = random.choice(list(Basis))
            value = random.randint(0, 1)
            emission_time = i * (FRAME_DURATION_MS * 1000000 / PULSES_PER_FRAME)  # in ps
            
            state = QuantumState(basis=basis, value=value)
            pulse = Pulse(state=state, emission_time_ps=emission_time)
            self.current_frame.append(pulse)
            
        return self.current_frame, self.nonce
        
    def send_challenge(self):
        """Send quantum challenge to Bob"""
        pulses, nonce = self.create_quantum_challenge()
        # FIX: Pass the eve object to the transmit method
        self.quantum_channel.transmit(pulses, eve=self.eve, sender="Alice", receiver="Bob")
        
        # Send classical message with nonce
        self.classical_channel.send(
            sender="Alice",
            receiver="Bob",
            message={
                'type': 'challenge',
                'frame_id': self.frame_id,
                'nonce': nonce,
                'timestamp': time.time()
            }
        )
        
        return self.frame_id
        
    def verify_response(self, response):
        """Verify Bob's response to the challenge"""
        if response['nonce'] != self.nonce:
            return False, "Nonce mismatch"
            
        # Check timing constraints
        timing_errors = 0
        basis_matches = 0
        measurement_errors = 0
        
        for i, pulse in enumerate(self.current_frame):
            if i >= len(response['measurements']):
                break
                
            measurement = response['measurements'][i]
            
            if measurement['detected']:
                # Check if detection is within timing window
                if abs(pulse.emission_time_ps - measurement['detection_time_ps']) > TIMING_WINDOW_PS:
                    timing_errors += 1
                    
                # Check if basis matches
                if pulse.state.basis == measurement['basis']:
                    basis_matches += 1
                    
                    # Check if measurement matches expected value
                    if pulse.state.value != measurement['value']:
                        measurement_errors += 1
        
        # Calculate QBER
        if basis_matches > 0:
            qber = measurement_errors / basis_matches
        else:
            qber = 1.0  # Maximum error if no basis matches
            
        # Calculate success rate
        detected_pulses = sum(1 for m in response['measurements'] if m['detected'])
        success_rate = detected_pulses / len(self.current_frame)
        
        # Store history for visualization
        self.history['qber'].append(qber)
        self.history['success_rate'].append(success_rate)
        self.history['timestamps'].append(time.time())
        
        # Determine authentication result
        if qber <= QBER_THRESHOLD and success_rate >= MIN_SUCCESS_RATE:
            self.authenticated = True
            return True, f"Authentication successful (QBER: {qber:.4f}, Success: {success_rate:.4f})"
        else:
            self.authenticated = False
            return False, f"Authentication failed (QBER: {qber:.4f}, Success: {success_rate:.4f})"

class Bob:
    """Client in the QTA protocol"""
    
    def __init__(self, classical_channel, quantum_channel):
        self.classical_channel = classical_channel
        self.quantum_channel = quantum_channel
        self.current_nonce = None
        self.measurements = []
        self.authenticated = False
        
    def receive_quantum_states(self):
        """Receive quantum states from Alice"""
        # Receive quantum states from the channel
        received_pulses = self.quantum_channel.receive("Bob")
        
        self.measurements = []
        for pulse in received_pulses:
            # Randomly choose measurement basis
            measurement_basis = random.choice(list(Basis))
            
            # Measure the quantum state
            measurement_result, basis_match = pulse.state.measure(measurement_basis)
            
            # Simulate detection time with some jitter
            detection_time = pulse.emission_time_ps + random.gauss(0, 10)  # 10ps jitter
            
            # Store measurement
            measurement = {
                'basis': measurement_basis,
                'value': measurement_result,
                'detection_time_ps': detection_time,
                'detected': True
            }
            
            self.measurements.append(measurement)
            
        return len(self.measurements)
        
    def send_response(self):
        """Send measurement results to Alice"""
        if not self.measurements:
            return False
            
        # Create HMAC for integrity
        hmac_key = b"shared_secret_key"  # In a real implementation, this would be securely established
        message = json.dumps(self.measurements).encode()
        signature = hmac.new(hmac_key, message, hashlib.sha256).hexdigest()
        
        self.classical_channel.send(
            sender="Bob",
            receiver="Alice",
            message={
                'type': 'response',
                'nonce': self.current_nonce,
                'measurements': self.measurements,
                'signature': signature
            }
        )
        
        return True
        
    def process_classical_messages(self):
        """Process classical messages from Alice"""
        messages = self.classical_channel.receive("Bob")
        
        for msg in messages:
            if msg['message']['type'] == 'challenge':
                self.current_nonce = msg['message']['nonce']
                # Receive quantum states
                self.receive_quantum_states()
                # Send response
                self.send_response()
                
            elif msg['message']['type'] == 'auth_result':
                self.authenticated = msg['message']['result']

class Eve:
    """Eavesdropper in the QTA protocol"""
    
    def __init__(self, quantum_channel):
        self.quantum_channel = quantum_channel
        self.intercepted_pulses = []
        self.active = False
        
    def intercept(self, pulse):
        """Intercept and resend a quantum pulse"""
        if not self.active:
            return pulse
            
        # Randomly choose measurement basis
        measurement_basis = random.choice(list(Basis))
        
        # Measure the quantum state
        measurement_result, _ = pulse.state.measure(measurement_basis)
        
        # Resend a new quantum state based on measurement
        new_state = QuantumState(basis=measurement_basis, value=measurement_result)
        new_pulse = Pulse(
            state=new_state,
            emission_time_ps=pulse.emission_time_ps + random.gauss(0, 5)  # Add small delay
        )
        
        self.intercepted_pulses.append(new_pulse)
        return new_pulse

class QTASimulator:
    """Main simulator for QTA protocol with visualization"""
    
    def __init__(self):
        # Create channels
        self.classical_channel = ClassicalChannel()
        self.quantum_channel = QuantumChannel()
        
        # Create parties
        self.eve = Eve(self.quantum_channel)
        self.alice = Alice(self.classical_channel, self.quantum_channel, self.eve)
        self.bob = Bob(self.classical_channel, self.quantum_channel)
        
        # Visualization setup
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('Quantum Temporal Authentication (QTA) Simulator', fontsize=16)
        
        # Create grid for subplots
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
        
        # Setup axes
        self.qber_ax = self.fig.add_subplot(gs[0, 0])
        self.success_ax = self.fig.add_subplot(gs[0, 1])
        self.traffic_ax = self.fig.add_subplot(gs[1, :])
        self.quantum_ax = self.fig.add_subplot(gs[2, 0])
        self.status_ax = self.fig.add_subplot(gs[2, 1])
        
        # Initialize plots
        self.qber_line, = self.qber_ax.plot([], [], 'r-', label='QBER')
        self.success_line, = self.success_ax.plot([], [], 'g-', label='Success Rate')
        
        self.qber_ax.set_xlim(0, 20)
        self.qber_ax.set_ylim(0, 0.2)
        self.qber_ax.set_xlabel('Frame')
        self.qber_ax.set_ylabel('QBER')
        self.qber_ax.axhline(y=QBER_THRESHOLD, color='r', linestyle='--', label='Threshold')
        self.qber_ax.legend()
        
        self.success_ax.set_xlim(0, 20)
        self.success_ax.set_ylim(0, 1.1)
        self.success_ax.set_xlabel('Frame')
        self.success_ax.set_ylabel('Success Rate')
        self.success_ax.axhline(y=MIN_SUCCESS_RATE, color='g', linestyle='--', label='Threshold')
        self.success_ax.legend()
        
        # Quantum state visualization
        self.quantum_ax.set_title('Quantum State Visualization')
        self.quantum_ax.set_xlabel('Pulse Index')
        self.quantum_ax.set_ylabel('Basis/Value')
        
        self.status_ax.axis('off')
        
        # Animation control
        self.animation = None
        self.running = False
        self.frame_count = 0
        
        # Control panel
        self.setup_controls()
        
    def setup_controls(self):
        """Setup control buttons for the simulator"""
        from matplotlib.widgets import Button, CheckButtons
        
        # Start/Stop button
        ax_button = plt.axes([0.45, 0.01, 0.1, 0.03])
        self.btn_startstop = Button(ax_button, 'Start')
        self.btn_startstop.on_clicked(self.toggle_simulation)
        
        # Eve toggle
        ax_eve = plt.axes([0.8, 0.01, 0.15, 0.03])
        self.chk_eve = CheckButtons(ax_eve, ['Eve Active'])
        self.chk_eve.on_clicked(self.toggle_eve)
        
    def toggle_simulation(self, event):
        """Start or stop the simulation"""
        self.running = not self.running
        self.btn_startstop.label.set_text('Stop' if self.running else 'Start')
        
        if self.running and self.animation is None:
            self.animation = FuncAnimation(
                self.fig, self.update_frame, interval=1000, blit=False
            )
            plt.draw()
            
    def toggle_eve(self, label):
        """Toggle Eve's activity"""
        self.eve.active = not self.eve.active
        self.quantum_channel.eve_active = self.eve.active
        
    def update_frame(self, frame):
        """Update the visualization for each frame"""
        if not self.running:
            return self.qber_line, self.success_line
            
        # Run one frame of the protocol
        self.run_frame()
        
        # Update QBER plot
        if self.alice.history['qber']:
            x = list(range(len(self.alice.history['qber'])))
            self.qber_line.set_data(x, list(self.alice.history['qber']))
            
        # Update success rate plot
        if self.alice.history['success_rate']:
            x = list(range(len(self.alice.history['success_rate'])))
            self.success_line.set_data(x, list(self.alice.history['success_rate']))
            
        # Update traffic visualization
        self.update_traffic_visualization()
        
        # Update quantum state visualization
        self.quantum_ax.clear()
        self.quantum_ax.set_title('Quantum State Visualization')
        self.quantum_ax.set_xlabel('Pulse Index')
        self.quantum_ax.set_ylabel('Basis/Value')
        
        if self.alice.current_frame:
            # Show a subset of pulses for clarity
            sample_size = min(50, len(self.alice.current_frame))
            indices = random.sample(range(len(self.alice.current_frame)), sample_size)
            
            for i in indices:
                pulse = self.alice.current_frame[i]
                basis_str = 'Z' if pulse.state.basis == Basis.Z else 'X'
                value_str = str(pulse.state.value)
                
                # Plot as a scatter point with color based on basis and marker based on value
                color = 'blue' if pulse.state.basis == Basis.Z else 'red'
                marker = 'o' if pulse.state.value == 0 else '^'
                
                self.quantum_ax.scatter(i, 0, color=color, marker=marker, alpha=0.7)
                
        # Update status display
        self.status_ax.clear()
        self.status_ax.axis('off')
        
        status_text = f"Frame: {self.frame_count}\n"
        status_text += f"Alice Authenticated: {self.alice.authenticated}\n"
        status_text += f"Bob Authenticated: {self.bob.authenticated}\n"
        status_text += f"Eve Active: {self.eve.active}\n"
        
        if self.alice.history['qber']:
            status_text += f"Current QBER: {self.alice.history['qber'][-1]:.4f}\n"
            
        if self.alice.history['success_rate']:
            status_text += f"Current Success Rate: {self.alice.history['success_rate'][-1]:.4f}\n"
            
        self.status_ax.text(0.1, 0.5, status_text, fontsize=12, 
                           verticalalignment='center', family='monospace')
        
        self.frame_count += 1
        
        return self.qber_line, self.success_line
        
    def update_traffic_visualization(self):
        """Update the traffic flow visualization"""
        # Clear the entire axis for a fresh redraw
        self.traffic_ax.cla()

        # Reset axis properties
        self.traffic_ax.set_title('Traffic Flow Visualization')
        self.traffic_ax.set_xlim(0, 10)
        self.traffic_ax.set_ylim(0, 6) # Adjusted y-limits slightly
        self.traffic_ax.set_xlabel('Time (last 10s)')
        self.traffic_ax.set_yticks([1, 3, 5])
        self.traffic_ax.set_yticklabels(['Alice', 'Bob', 'Eve'])
        self.traffic_ax.grid(True, linestyle='--', alpha=0.7)

        # Get current time
        current_time = time.time()

        # Get all traffic events from channels
        all_events = []
        all_events.extend(self.quantum_channel.traffic_log)
        all_events.extend(self.classical_channel.traffic_log)

        # Filter events to show only recent ones (last 10 seconds)
        recent_events = [e for e in all_events if current_time - e.timestamp <= 10]

        if not recent_events:
            return # Nothing to plot

        # Normalize timestamps to fit in the plot
        min_time = min(e.timestamp for e in recent_events)
        max_time = max(e.timestamp for e in recent_events)
        time_range = max_time - min_time if max_time > min_time else 1

        # Plot events
        for event in recent_events:
            # Normalize time to [0, 10] range
            norm_time = (event.timestamp - min_time) / time_range * 10

            # Determine y-position based on sender
            y_pos_map = {"Alice": 1, "Bob": 3, "Eve": 5}
            y_pos = y_pos_map.get(event.sender)
            if y_pos is None:
                continue

            # Determine color and marker based on event type
            style_map = {
                'quantum': {'color': 'blue', 'marker': 'o'},
                'classical': {'color': 'green', 'marker': 's'},
                'intercept': {'color': 'red', 'marker': 'x'}
            }
            style = style_map.get(event.event_type)
            if style is None:
                continue

            # Plot the event
            self.traffic_ax.scatter(norm_time, y_pos, color=style['color'], marker=style['marker'], s=100, alpha=0.7)

            # Add arrow to show direction
            end_y_pos = y_pos_map.get(event.receiver)
            if end_y_pos is not None and event.event_type != 'intercept' and y_pos != end_y_pos:
                self.traffic_ax.arrow(norm_time, y_pos, 0, end_y_pos - y_pos,
                                     head_width=0.2, head_length=0.3,
                                     fc=style['color'], ec=style['color'], alpha=0.5)

    def run_frame(self):
        """Run one frame of the QTA protocol"""
        # Alice sends challenge
        frame_id = self.alice.send_challenge()
        
        # Bob processes messages and responds
        self.bob.process_classical_messages()
        
        # Alice verifies response
        messages = self.alice.classical_channel.receive("Alice")
        for msg in messages:
            if msg['message']['type'] == 'response':
                success, message = self.alice.verify_response(msg['message'])
                
                # Send result to Bob
                self.alice.classical_channel.send(
                    sender="Alice",
                    receiver="Bob",
                    message={
                        'type': 'auth_result',
                        'result': success,
                        'message': message
                    }
                )
                
        # Bob processes the result
        self.bob.process_classical_messages()
        
    def run(self):
        """Run the simulator"""
        plt.show()

# Main execution
if __name__ == "__main__":
    simulator = QTASimulator()
    simulator.run()
    
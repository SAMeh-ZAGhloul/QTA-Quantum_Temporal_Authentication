import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import time
import threading
import queue
from collections import deque
import random
import hashlib
import hmac
import json

# Constants for QTA protocol
PULSES_PER_FRAME = 1024
FRAME_DURATION_MS = 1
TIMING_WINDOW_PS = 50  # Timing window in picoseconds
QBER_THRESHOLD = 0.05  # 5% QBER threshold
MIN_SUCCESS_RATE = 0.95  # Minimum success rate for authentication
PHOTON_LOSS_PROB = 0.2  # Probability of photon loss in the channel

# Use stable SimPy-inspired engine for simulation, avoiding QuNetSim threading issues

# --- QTA SIMULATION ENGINE (Using SimPy Logic for Stability) ---

class QTA_Simulated_Engine:
    """
    A stable, simplified engine that mimics the QTA protocol using local
    randomization and time simulation, avoiding QuNetSim's threading conflicts.
    """
    def __init__(self):
        # We don't start the full QuNetSim network, just the simulation state
        print(">>> Starting Hybrid SimPy/QuNetSim Simulation Engine.")
        self.eve_active = False
        self.total_sent = 0
        self.total_detected = 0
        self.total_errors = 0

    def run_protocol_step(self, batch_size=PULSES_PER_FRAME):
        results = []

        for _ in range(batch_size):
            self.total_sent += 1

            # 1. Alice Prepares
            original_bit = np.random.randint(0, 2)
            basis = np.random.randint(0, 2)  # Alice's basis (0=Z, 1=X)

            offset = np.random.normal(0, 0.02)  # Base timing offset (natural jitter)
            is_attacked = False
            is_error = False

            # 2. Eve's Action
            if self.eve_active:
                is_attacked = True
                offset += 0.8  # Add the large, easily detected QTA offset

                # Eve's intercept-resend attack:
                eve_basis = np.random.randint(0, 2)

                # Check for QBER: An error (bit flip) occurs if Alice and Eve
                # chose different bases, which is 50% of the time.
                if eve_basis != basis:
                    # In this 50% case, the qubit state collapses randomly.
                    # Bob will measure incorrectly 50% of the time,
                    # leading to an overall 25% error rate (QBER).
                    if np.random.rand() < 0.5:
                        is_error = True

            # 3. Bob Measures (Simulated)
            bob_basis = np.random.randint(0, 2)

            # 4. Sift & Error Check
            if bob_basis == basis:
                # Bases match, this is a sifting success event
                self.total_detected += 1

                # Check if an error was caused by Eve's previous action,
                # or if a natural/external error occurred (small chance)
                if not is_attacked:
                    # Small natural quantum error rate (simulated dark counts/loss)
                    if np.random.rand() < 0.01:
                        is_error = True

                if is_error:
                    self.total_errors += 1

                results.append({
                    "idx": self.total_detected,
                    "offset": offset,
                    "error": is_error,
                    "attacked": is_attacked
                })

        return results

    def stop(self):
        # No network to stop
        print(">>> Hybrid Engine Stopped.")
        pass

class QTASimulator:
    """Main QTA simulator using QuNetSim with GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Temporal Authentication (QTA) Simulator")
        self.root.geometry("1200x800")

        # Use the stable SimPy-inspired engine
        self.engine = QTA_Simulated_Engine()

        # Simulation parameters
        self.running = False
        self.frame_count = 0
        self.authenticated = False

        # Data storage for visualization
        self.master_buffer = deque(maxlen=100)

        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Simulation Controls", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Simulation", command=self.toggle_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        reset_button = ttk.Button(control_frame, text="Reset", command=self.reset_simulation)
        reset_button.pack(side=tk.LEFT, padx=5)
        
        # Eve active checkbox
        self.eve_var = tk.BooleanVar()
        eve_checkbox = ttk.Checkbutton(control_frame, text="Eve Active", variable=self.eve_var, command=self.toggle_eve)
        eve_checkbox.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Status: Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=5)
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(main_frame, text="Real-time Visualization", padding="10")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create matplotlib figure for visualization
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 6))
        self.fig.tight_layout(pad=3.0)

        # Photon time-of-arrival plot
        self.timing_ax = self.axes[0, 0]
        self.timing_ax.set_title("Photon Time-of-Arrival (QTA)")
        self.timing_ax.set_xlabel("Frame")
        self.timing_ax.set_ylabel("Offset (ns)")
        self.timing_ax.set_ylim(-0.5, 1.5)
        self.timing_ax.axhspan(-0.15, 0.15, color="green", alpha=0.15)  # Window tolerance
        self.plot_safe, = self.timing_ax.plot([], [], 'o', color="green", markersize=5, alpha=0.7)
        self.plot_attack, = self.timing_ax.plot([], [], 'o', color="red", markersize=5, alpha=0.8)
        self.timing_ax.grid(True, alpha=0.3)

        # QBER plot
        self.qber_ax = self.axes[0, 1]
        self.qber_ax.set_title("QBER (Bit Error Rate)")
        self.qber_ax.set_xlabel("Frame")
        self.qber_ax.set_ylabel("Error %")
        self.qber_ax.set_ylim(0, 0.6)
        self.qber_ax.axhline(y=QBER_THRESHOLD, color='red', linestyle='--')
        self.qber_line, = self.qber_ax.plot([], [], color="#333", lw=2)
        self.qber_ax.grid(True, alpha=0.3)

        # Quantum state visualization
        self.quantum_ax = self.axes[1, 0]
        self.quantum_ax.set_title("Quantum State Visualization")
        self.quantum_ax.set_xlabel("Pulse Index")
        self.quantum_ax.set_ylabel("Basis/Value")

        # Authentication status
        self.auth_ax = self.axes[1, 1]
        self.auth_ax.set_title("Authentication Status")
        self.auth_ax.axis('off')
        
        # Embed matplotlib figure in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Info panel
        info_frame = ttk.LabelFrame(main_frame, text="Protocol Information", padding="10")
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Protocol info text
        info_text = """Quantum Temporal Authentication (QTA) Protocol:
        
1. Alice (Server) generates a quantum challenge with random states and timestamps.
2. Bob (Client) measures the quantum states and records detection times.
3. Bob sends measurement results back to Alice over a classical channel.
4. Alice verifies the measurements based on expected values and timing constraints.
5. Authentication is successful if QBER is below threshold and timing constraints are met.

Eve (Eavesdropper) can attempt to intercept and measure quantum states,
which introduces errors that increase the QBER and can be detected."""
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(side=tk.LEFT, padx=5)
        
    def toggle_simulation(self):
        """Start or stop the simulation"""
        if not self.running:
            self.running = True
            self.start_button.config(text="Stop Simulation")
            self.status_var.set("Status: Running")
            
            # Start simulation in a separate thread
            self.simulation_thread = threading.Thread(target=self.run_simulation)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
        else:
            self.running = False
            self.start_button.config(text="Start Simulation")
            self.status_var.set("Status: Stopped")
            
    def reset_simulation(self):
        """Reset the simulation"""
        self.running = False
        self.start_button.config(text="Start Simulation")
        self.status_var.set("Status: Ready")
        self.frame_count = 0
        self.authenticated = False

        # Reset engine
        self.engine.eve_active = False
        self.eve_var.set(False)
        self.engine.total_sent = 0
        self.engine.total_detected = 0
        self.engine.total_errors = 0

        # Clear history
        self.master_buffer.clear()

        # Update visualization
        self.update_visualization()
        
    def toggle_eve(self):
        """Toggle Eve's activity"""
        self.eve_active = self.eve_var.get()
        self.engine.eve_active = self.eve_active

    def run_simulation(self):
        """Run the QTA simulation"""
        while self.running:
            # Run one frame of the protocol
            self.run_frame()
            
            # Update visualization
            self.root.after(0, self.update_visualization)
            
            # Sleep for a short time to control simulation speed
            time.sleep(0.5)
            
    def run_frame(self):
        """Run one frame of the QTA protocol"""
        # Get Data from engine
        batch_data = self.engine.run_protocol_step(PULSES_PER_FRAME)

        for d in batch_data:
            err_val = 1.0 if d["error"] else 0.0
            self.master_buffer.append({
                "x": d["idx"],
                "y_time": d["offset"],
                "attack": d["attacked"],
                "err": err_val,
                "lost": False  # Not simulating loss here
            })

            self.frame_count = d["idx"]

        # Determine authentication based on recent QBER
        if len(self.master_buffer) > 1:
            recent_errs = [p["err"] for p in self.master_buffer]
            avg_qber = sum(recent_errs) / len(recent_errs)
            self.authenticated = avg_qber < QBER_THRESHOLD
        
    def update_visualization(self):
        """Update the visualization plots"""
        if len(self.master_buffer) > 1:
            # Sort data into buckets
            xs_safe = [p["x"] for p in self.master_buffer if not p["attack"]]
            ys_safe = [p["y_time"] for p in self.master_buffer if not p["attack"]]
            
            xs_att = [p["x"] for p in self.master_buffer if p["attack"]]
            ys_att = [p["y_time"] for p in self.master_buffer if p["attack"]]

            self.plot_safe.set_data(xs_safe, ys_safe)
            self.plot_attack.set_data(xs_att, ys_att)
            
            last_x = self.master_buffer[-1]["x"]
            self.timing_ax.set_xlim(max(0, last_x - self.master_buffer.maxlen), last_x + 10)

            # QBER
            qber_x = []
            qber_y = []
            recent_errs = deque(maxlen=20)
            
            for p in self.master_buffer:
                recent_errs.append(p["err"])
                avg = sum(recent_errs) / len(recent_errs)
                qber_x.append(p["x"])
                qber_y.append(avg)
            
            self.qber_line.set_data(qber_x, qber_y)
            self.qber_ax.set_xlim(max(0, last_x - self.master_buffer.maxlen), last_x + 10)
            
            # Update quantum state visualization
            self.quantum_ax.clear()
            self.quantum_ax.set_title("Quantum State Visualization")
            self.quantum_ax.set_xlabel("Pulse Index")
            self.quantum_ax.set_ylabel("Basis/Value")
            
            # Show a sample of quantum states
            sample_size = min(50, PULSES_PER_FRAME)
            if sample_size > 0:
                indices = random.sample(range(PULSES_PER_FRAME), sample_size)
                
                for i in indices:
                    # Randomly generate a state for visualization
                    basis = random.choice(["Z", "X"])
                    value = random.randint(0, 1)
                    
                    # Plot as a scatter point with color based on basis and marker based on value
                    color = 'blue' if basis == "Z" else 'red'
                    marker = 'o' if value == 0 else '^'
                    
                    self.quantum_ax.scatter(i, 0, color=color, marker=marker, alpha=0.7)
            
            self.canvas.draw_idle()

            ratio = self.engine.total_errors / self.engine.total_detected if self.engine.total_detected > 0 else 0
            self.status_var.set(f"TOTAL SENT: {self.engine.total_sent}\nDETECTED: {self.engine.total_detected}\nAVG QBER: {ratio:.1%}")

        # Update authentication status
        self.auth_ax.clear()
        self.auth_ax.set_title("Authentication Status")
        self.auth_ax.axis('off')

        status_text = f"Frame: {self.frame_count}\n"
        status_text += f"Authentication: {'Success' if self.authenticated else 'Failed'}\n"
        status_text += f"Eve Active: {'Yes' if self.eve_active else 'No'}\n"
        status_text += f"Eve Active on Engine: {'Yes' if self.engine.eve_active else 'No'}\n"

        if self.master_buffer:
            recent_buffer = list(self.master_buffer)[-20:]
            last_errs = [p["err"] for p in recent_buffer]
            if last_errs:
                current_qber = sum(last_errs) / len(last_errs)
                status_text += f"Current QBER: {current_qber:.4f}\n"

        self.auth_ax.text(0.1, 0.5, status_text, fontsize=12,
                         verticalalignment='center', family='monospace')
        
    def on_closing(self):
        """Handle window closing event"""
        self.running = False
        self.engine.stop()
        self.root.destroy()

# Main function
def main():
    root = tk.Tk()
    app = QTASimulator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()

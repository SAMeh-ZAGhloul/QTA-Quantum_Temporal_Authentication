import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
import queue

# QuNetSim Imports
from qunetsim.components import Host, Network
from qunetsim.objects import Qubit, Logger
Logger.DISABLED = True  # Disable verbose logging for performance

# --- CONFIGURATION ---
FRAME_SIZE = 20        # Qubits per visual update (Lower than numpy for performance)
SAFE_QBER = 0.05       # 5% threshold
SAFE_WINDOW = 30       # +/- 30 ps
EVE_DELAY_ADD = 60     # Eve adds 60ps delay
JITTER = 10            # Network jitter

class QTANetworkEngine:
    """
    Manages the QuNetSim Network backend.
    """
    def __init__(self):
        self.network = Network.get_instance()
        self.network.start()

        self.alice = Host('Alice')
        self.bob = Host('Bob')
        self.eve = Host('Eve')

        # Add hosts
        self.network.add_host(self.alice)
        self.network.add_host(self.bob)
        self.network.add_host(self.eve)

        # Topology: Alice <-> Eve <-> Bob
        # In QuNetSim, we simulate the attack by intercepting transmissions through Eve
        self.alice.add_connection(self.eve.host_id)
        self.eve.add_connection(self.alice.host_id)
        self.eve.add_connection(self.bob.host_id)
        self.bob.add_connection(self.eve.host_id)

        # Ensure quantum network graph has the edges (workaround for QuNetSim setup)
        import networkx as nx
        if not self.network.quantum_network.has_edge(self.alice.host_id, self.eve.host_id):
            self.network.quantum_network.add_edge(self.alice.host_id, self.eve.host_id)
        if not self.network.quantum_network.has_edge(self.eve.host_id, self.bob.host_id):
            self.network.quantum_network.add_edge(self.eve.host_id, self.bob.host_id)

        # Start host listeners
        self.alice.start()
        self.bob.start()
        self.eve.start()

        self.is_attack_active = False

    def run_protocol_step(self):
        """
        Executes one 'frame' of QTA.
        Returns: (calculated_qber, list_of_delays)
        """
        alice_bits = []
        alice_bases = []
        bob_results = []
        bob_bases = []
        delays = []

        # Alice prepares a frame
        for _ in range(FRAME_SIZE):
            bit = np.random.randint(0, 2)
            basis = np.random.randint(0, 2) # 0=Z, 1=X
            alice_bits.append(bit)
            alice_bases.append(basis)

            # Create Qubit
            q = Qubit(self.alice)
            if basis == 1:
                q.H()
            if bit == 1:
                q.X()

            # --- TRANSMISSION ---
            # To simulate ToA, we attach metadata to the qubit transmission
            # QuNetSim abstracts time, so we model the latency physically here.
            
            t_start = 0 
            
            if self.is_attack_active:
                # ATTACK PATH: Alice -> Eve (Measure/Resend) -> Bob

                # 1. Send to Eve
                self.alice.send_qubit(self.eve.host_id, q)
                time.sleep(0.05)  # Allow network to process
                q_eve = self.eve.get_qubit(self.alice.host_id)

                if q_eve is not None:
                    # 2. Eve Measures (Attack!)
                    # Eve guesses basis
                    eve_basis = np.random.randint(0, 2)
                    if eve_basis == 1:
                        q_eve.H()
                    _ = q_eve.measure() # State collapsed!

                    # 3. Eve Resends (Intercept-Resend)
                    # She creates a new qubit based on her measurement
                    q_fake = Qubit(self.eve)
                    # (Simplified: Eve just resends a random state or tries to clone)
                    # Let's assume she resends based on her measurement, which might be wrong
                    if eve_basis == 1:
                        q_fake.H()
                    
                    # 4. Send to Bob with ADDED DELAY
                    self.eve.send_qubit(self.bob.host_id, q_fake)
                    time.sleep(0.01)  # Allow network to process
                    q_recv = self.bob.get_qubit(self.eve.host_id)
                else:
                    # No qubit received, skip to dummy
                    q_recv = Qubit(self.bob)
                
                # Add physical delay simulation
                current_delay = np.random.normal(EVE_DELAY_ADD, JITTER)
                
            else:
                # SECURE PATH: Alice -> Eve (Transparent) -> Bob
                # Or conceptual direct link
                self.alice.send_qubit(self.eve.host_id, q)
                time.sleep(0.05)  # Allow network to process
                q_at_eve = self.eve.get_qubit(self.alice.host_id)

                # Eve just forwards without measuring
                if q_at_eve is not None:
                    self.eve.send_qubit(self.bob.host_id, q_at_eve)
                    time.sleep(0.05)  # Allow network to process
                    q_recv = self.bob.get_qubit(self.eve.host_id)
                else:
                    # If no qubit received, create a dummy for measurement
                    q_recv = Qubit(self.bob)  # Dummy qubit
                
                # Normal fiber delay
                current_delay = np.random.normal(0, JITTER)

            delays.append(current_delay)

            # --- BOB MEASUREMENT ---
            b_basis = np.random.randint(0, 2)
            bob_bases.append(b_basis)
            
            if b_basis == 1:
                q_recv.H()
            
            meas = q_recv.measure()
            bob_results.append(meas)

        # --- SIFTING & METRICS ---
        errors = 0
        sifted_count = 0
        valid_delays = []

        for i in range(FRAME_SIZE):
            # Sifting: Only compare where bases matched
            if alice_bases[i] == bob_bases[i]:
                sifted_count += 1
                valid_delays.append(delays[i])
                if alice_bits[i] != bob_results[i]:
                    errors += 1

        qber = errors / sifted_count if sifted_count > 0 else 0
        return qber, valid_delays

    def stop(self):
        self.network.stop()


class QTAGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("QTA Protocol Simulator (QuNetSim Engine)")
        self.root.geometry("1200x700")
        
        # Simulation Engine
        self.engine = QTANetworkEngine()
        self.running = True
        
        # Data Stores
        self.history_qber = [0] * 50
        self.history_toa = []

        self._setup_ui()
        self._start_simulation_loop()

    def _setup_ui(self):
        # Layout
        left_panel = tk.Frame(self.root, width=200, bg="#f0f0f0")
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        right_panel = tk.Frame(self.root)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # -- CONTROLS --
        tk.Label(left_panel, text="QTA Control", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=20)
        
        self.btn_attack = tk.Button(left_panel, text="ACTIVATE EVE", bg="#dddddd", fg="black",
                                    font=("Arial", 12), height=2, width=15,
                                    command=self.toggle_attack)
        self.btn_attack.pack(pady=20)

        self.lbl_status = tk.Label(left_panel, text="Status: SECURE", fg="green", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.lbl_status.pack(pady=10)

        self.lbl_stats = tk.Label(left_panel, text="QBER: 0.0%\nDelay: 0ps", justify=tk.LEFT, bg="#f0f0f0")
        self.lbl_stats.pack(pady=20)

        # -- CHARTS --
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(211) # QBER
        self.ax2 = self.fig.add_subplot(212) # ToA
        
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def toggle_attack(self):
        self.engine.is_attack_active = not self.engine.is_attack_active
        if self.engine.is_attack_active:
            self.btn_attack.config(text="DEACTIVATE EVE", bg="red", fg="white")
            self.lbl_status.config(text="Status: ATTACK", fg="red")
        else:
            self.btn_attack.config(text="ACTIVATE EVE", bg="#dddddd", fg="black")
            self.lbl_status.config(text="Status: SECURE", fg="green")

    def _update_charts(self, qber, delays):
        # Update Data
        self.history_qber.append(qber)
        self.history_qber.pop(0)
        
        if len(delays) > 0:
            self.history_toa.extend(delays)
            if len(self.history_toa) > 500: # Keep histograms fresh
                self.history_toa = self.history_toa[-500:]

        # Draw QBER
        self.ax1.clear()
        self.ax1.set_title("Real-time QBER Monitor (Quantum Bit Error Rate)")
        self.ax1.set_ylabel("Error Rate")
        self.ax1.set_ylim(0, 0.5)
        self.ax1.axhline(SAFE_QBER, color='red', linestyle='--', label="Max Threshold")
        self.ax1.plot(self.history_qber, color='blue', linewidth=2)
        self.ax1.legend()

        # Draw ToA
        self.ax2.clear()
        self.ax2.set_title("Time-of-Arrival Distribution (Picoseconds)")
        self.ax2.set_xlim(-100, 200)
        self.ax2.axvspan(-SAFE_WINDOW, SAFE_WINDOW, color='green', alpha=0.2, label="Valid Auth Window")
        
        if self.history_toa:
            self.ax2.hist(self.history_toa, bins=30, color='purple', alpha=0.7)
        
        self.canvas.draw()
        
        # Update Stats Label
        avg_delay = np.mean(delays) if len(delays) > 0 else 0
        self.lbl_stats.config(text=f"QBER: {qber:.2%}\nAvg Delay: {avg_delay:.1f}ps")

    def _start_simulation_loop(self):
        def loop():
            if self.running:
                # Run one physics frame
                qber, delays = self.engine.run_protocol_step()
                
                # Update GUI in main thread
                self.root.after(0, self._update_charts, qber, delays)
                
                # Schedule next run
                self.root.after(5000, loop) # 5000ms refresh rate to allow processing
        
        loop()

    def on_closing(self):
        self.running = False
        self.engine.stop()
        self.root.destroy()

# --- MAIN ---
if __name__ == "__main__":
    root = tk.Tk()
    app = QTAGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

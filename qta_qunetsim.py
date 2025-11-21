import tkinter as tk
import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

# --- CONFIGURATION (Common to both SimPy and QuNetSim concepts) ---
BATCH_SIZE = 1            # 1 Qubit per tick for GUI stability
ATTACK_DELAY = 0.8        
WINDOW_TOLERANCE = 0.15   
GUI_REFRESH_MS = 100      
MAX_HISTORY = 100         

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

    def run_protocol_step(self):
        results = []
        
        for _ in range(BATCH_SIZE):
            self.total_sent += 1
            
            # 1. Alice Prepares
            original_bit = np.random.randint(0, 2)
            basis = np.random.randint(0, 2) # Alice's basis (0 or 1)
            
            offset = np.random.normal(0, 0.02) # Base timing offset (natural jitter)
            is_attacked = False
            is_error = False
            
            # 2. Eve's Action
            if self.eve_active:
                is_attacked = True
                offset += ATTACK_DELAY # Add the large, easily detected QTA offset
                
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

# --- TKINTER DASHBOARD (Reuses the existing GUI structure) ---

class QTA_Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("QTA Protocol (Hybrid SimPy/QuNetSim Engine)")
        self.root.geometry("1100x750")
        self.root.bind('<space>', self.toggle_eve)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Use the stable SimPy-inspired engine
        self.engine = QTA_Simulated_Engine() 

        self.master_buffer = collections.deque(maxlen=MAX_HISTORY)
        self._build_gui()
        self.run_update_loop()

    def _build_gui(self):
        # ... (GUI setup is identical to previous versions) ...
        panel = tk.Frame(self.root, bg="#1a1a1a", width=250)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(panel, text="QUANTUM\nSECURITY", fg="#00d4ff", bg="#1a1a1a", font=("Impact", 20)).pack(pady=30)
        
        self.lbl_status = tk.Label(panel, text="LINK SECURE", fg="#00ff00", bg="black", 
                                   font=("Consolas", 14, "bold"), width=15, pady=10, relief="sunken")
        self.lbl_status.pack(pady=10)

        self.btn = tk.Button(panel, text="ACTIVATE EVE\n(Spacebar)", highlightbackground="#444", 
                             font=("Arial", 12, "bold"), height=3, command=self.toggle_eve_event)
        self.btn.pack(pady=20, padx=15, fill=tk.X)
        
        self.lbl_stats = tk.Label(panel, text="Initializing...", fg="#888", bg="#1a1a1a", justify=tk.LEFT, font=("Consolas", 10))
        self.lbl_stats.pack(pady=20)

        self.fig = plt.Figure(figsize=(8, 8), dpi=100, facecolor="#f0f0f0")
        gs = self.fig.add_gridspec(2, 1)
        
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax1.set_title("Photon Time-of-Arrival (QTA)")
        self.ax1.set_ylabel("Offset (ns)")
        self.ax1.set_ylim(-0.5, 1.5) 
        self.ax1.axhspan(-WINDOW_TOLERANCE, WINDOW_TOLERANCE, color="green", alpha=0.15)
        self.plot_safe, = self.ax1.plot([], [], 'o', color="green", markersize=5, alpha=0.7)
        self.plot_attack, = self.ax1.plot([], [], 'o', color="red", markersize=5, alpha=0.8)
        self.plot_lost, = self.ax1.plot([], [], 'x', color="gray", markersize=5, alpha=0.3) 
        self.ax1.grid(True, alpha=0.3)

        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax2.set_title("QBER (Bit Error Rate)")
        self.ax2.set_ylabel("Error %")
        self.ax2.set_ylim(0, 0.6)
        self.ax2.axhline(0.05, color="red", linestyle="--")
        self.plot_qber, = self.ax2.plot([], [], color="#333", lw=2)
        self.ax2.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        # ... (End of GUI setup) ...

    def toggle_eve_event(self): self.toggle_eve(None)

    def toggle_eve(self, event=None):
        self.engine.eve_active = not self.engine.eve_active
        if self.engine.eve_active:
            self.lbl_status.config(text="INTERCEPTED", fg="red")
            self.btn.config(text="DEACTIVATE EVE")
        else:
            self.lbl_status.config(text="LINK SECURE", fg="#00ff00")
            self.btn.config(text="ACTIVATE EVE")

    def run_update_loop(self):
        self.root.update_idletasks()

        # Get Data
        batch_data = self.engine.run_protocol_step()
        
        for d in batch_data:
            # Note: The SimPy engine doesn't return None, so we handle valid results only
            err_val = 1.0 if d["error"] else 0.0
            self.master_buffer.append({
                "x": d["idx"],
                "y_time": d["offset"],
                "attack": d["attacked"],
                "err": err_val,
                "lost": False # Loss is not simulated in this stable version
            })

        if len(self.master_buffer) > 1:
            # Sort data into buckets
            xs_safe = [p["x"] for p in self.master_buffer if not p["attack"]]
            ys_safe = [p["y_time"] for p in self.master_buffer if not p["attack"]]
            
            xs_att = [p["x"] for p in self.master_buffer if p["attack"]]
            ys_att = [p["y_time"] for p in self.master_buffer if p["attack"]]

            self.plot_safe.set_data(xs_safe, ys_safe)
            self.plot_attack.set_data(xs_att, ys_att)
            
            last_x = self.master_buffer[-1]["x"]
            self.ax1.set_xlim(max(0, last_x - MAX_HISTORY), last_x + 10)

            # QBER
            qber_x = []
            qber_y = []
            recent_errs = collections.deque(maxlen=20) 
            
            for p in self.master_buffer:
                recent_errs.append(p["err"])
                avg = sum(recent_errs) / len(recent_errs)
                qber_x.append(p["x"])
                qber_y.append(avg)
            
            self.plot_qber.set_data(qber_x, qber_y)
            self.ax2.set_xlim(max(0, last_x - MAX_HISTORY), last_x + 10)
            
            self.canvas.draw_idle()

            ratio = self.engine.total_errors/self.engine.total_detected if self.engine.total_detected > 0 else 0
            self.lbl_stats.config(text=f"TOTAL SENT: {self.engine.total_sent}\nDETECTED: {self.engine.total_detected}\nAVG QBER: {ratio:.1%}")

        self.root.after(GUI_REFRESH_MS, self.run_update_loop)

    def on_closing(self):
        print("Shutting down...")
        self.engine.stop()
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = QTA_Dashboard(root)
    root.mainloop()
    
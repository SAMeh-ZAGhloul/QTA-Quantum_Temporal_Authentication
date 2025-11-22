"""
Quantum Temporal Authentication (QTA) Simulator
GUI implementation with real-time visualization
NumPy backend for fast vectorized simulation
"""

import tkinter as tk
import numpy as np # <--- NUMPY BACKEND
import collections
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- CONFIGURATION ---
BATCH_SIZE = 5            # Generate 5 photons per frame for speed
FIBER_DELAY = 0.2         
ATTACK_DELAY = 0.8        
WINDOW_TOLERANCE = 0.15   
GUI_REFRESH_MS = 100      # 10 FPS
MAX_HISTORY = 100         

class QTA_Numpy_Engine:
    def __init__(self):
        self.eve_active = False
        self.total_sent = 0
        self.total_detected = 0
        self.total_errors = 0

    def next_batch(self):
        """
        Generates a batch of photon events using Vectorized Math.
        Returns a list of dicts: {'idx', 'offset', 'error', 'attacked'}
        """
        # 1. Alice Generates
        alice_bits = np.random.randint(0, 2, BATCH_SIZE)
        alice_bases = np.random.randint(0, 2, BATCH_SIZE)
        
        # 2. Channel Physics (Time of Arrival)
        # Base noise
        noise = np.random.normal(0, 0.01, BATCH_SIZE)
        offsets = np.full(BATCH_SIZE, FIBER_DELAY) + noise
        
        is_attacked = np.full(BATCH_SIZE, False)
        
        # 3. Eve Logic (Vectorized)
        if self.eve_active:
            # Add Delay
            offsets += ATTACK_DELAY
            is_attacked[:] = True
            
            # Intercept-Resend (Probability Math)
            # Eve guesses basis. If wrong (50%), she randomizes bit.
            eve_bases = np.random.randint(0, 2, BATCH_SIZE)
            # Mask where Eve chose wrong basis
            basis_mismatch = (eve_bases != alice_bases)
            # Randomize bits where basis was wrong
            alice_bits[basis_mismatch] = np.random.randint(0, 2, np.sum(basis_mismatch))

        # 4. Bob Measures
        bob_bases = np.random.randint(0, 2, BATCH_SIZE)
        
        # 5. Sifting & Error Checking
        results = []
        for i in range(BATCH_SIZE):
            self.total_sent += 1
            
            # Sift: Did Bob guess the right basis?
            if bob_bases[i] == alice_bases[i]:
                self.total_detected += 1
                
                # Check Bit Error
                # (In simulation, we compare modified alice_bits vs original, 
                # but here we just assume Bob measured alice_bits directly if basis matched)
                # Note: In real physics, Bob measures the photon state. 
                # Here, alice_bits already holds the state 'as it arrived'.
                
                # If Eve randomized it, it might be wrong.
                # We need original bits to compare? 
                # Simplified: Assume alice_bits IS the measured result.
                # Wait, we need to know if it CHANGED from original.
                # Let's refine:
                
                # Re-roll logic for clarity:
                # If Eve messed it up, there is 50% chance of error IF Bob measures correct basis.
                # Actually, the vector logic above modified alice_bits in place. 
                # We need the original intent. 
                # For simulation visuals, we can just assume "Error" if bit flip happened.
                # But we lost the original bit. Let's just simplify:
                
                is_error = False
                if self.eve_active and (eve_bases[i] != alice_bases[i]):
                     # Eve used wrong basis -> Result is random
                     # 50% chance it differs from what Alice intended
                     if np.random.random() < 0.5:
                         is_error = True
                
                if is_error:
                    self.total_errors += 1

                # Recenter offset for graph (so 0.0 is perfect arrival)
                # (The offsets array has FIBER_DELAY added, we want deviation)
                display_offset = offsets[i] - FIBER_DELAY

                results.append({
                    "idx": self.total_detected,
                    "offset": display_offset,
                    "error": is_error,
                    "attacked": is_attacked[i]
                })
                
        return results

class QTA_Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("QTA Protocol (NumPy Backend)")
        self.root.geometry("1100x750")
        
        self.root.bind('<space>', self.toggle_eve)
        
        # --- BACKEND SWAP ---
        self.engine = QTA_Numpy_Engine()
        # --------------------

        self.master_buffer = collections.deque(maxlen=MAX_HISTORY)
        self._build_gui()
        self.run_update_loop()

    def _build_gui(self):
        panel = tk.Frame(self.root, bg="#1a1a1a", width=250)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(panel, text="QUANTUM\nSECURITY", fg="#00d4ff", bg="#1a1a1a", font=("Impact", 20)).pack(pady=30)
        
        self.lbl_status = tk.Label(panel, text="LINK SECURE", fg="#00ff00", bg="black", 
                                   font=("Consolas", 14, "bold"), width=15, pady=10, relief="sunken")
        self.lbl_status.pack(pady=10)

        self.btn = tk.Button(panel, text="ACTIVATE EVE\n(Spacebar)", highlightbackground="#444", 
                             font=("Arial", 12, "bold"), height=3, command=self.toggle_eve_event)
        self.btn.pack(pady=20, padx=15, fill=tk.X)
        
        self.lbl_stats = tk.Label(panel, text="--", fg="#888", bg="#1a1a1a", justify=tk.LEFT, font=("Consolas", 10))
        self.lbl_stats.pack(pady=20)

        self.fig = plt.Figure(figsize=(8, 8), dpi=100, facecolor="#f0f0f0")
        gs = self.fig.add_gridspec(2, 1)
        
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax1.set_title("Photon Time-of-Arrival (NumPy)")
        self.ax1.set_ylabel("Offset (ns)")
        self.ax1.set_ylim(-0.5, 1.5) 
        self.ax1.axhspan(-WINDOW_TOLERANCE, WINDOW_TOLERANCE, color="green", alpha=0.15)
        self.plot_safe, = self.ax1.plot([], [], 'o', color="green", markersize=5, alpha=0.7)
        self.plot_attack, = self.ax1.plot([], [], 'o', color="red", markersize=5, alpha=0.8)
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

        # --- GET DATA FROM NUMPY ---
        batch_data = self.engine.next_batch()
        
        for d in batch_data:
            err_val = 1.0 if d["error"] else 0.0
            self.master_buffer.append({
                "x": d["idx"],
                "y_time": d["offset"],
                "attack": d["attacked"],
                "err": err_val
            })

        if len(self.master_buffer) > 1:
            # Rebuild Plot Arrays
            xs_safe = [p["x"] for p in self.master_buffer if not p["attack"]]
            ys_safe = [p["y_time"] for p in self.master_buffer if not p["attack"]]
            xs_att = [p["x"] for p in self.master_buffer if p["attack"]]
            ys_att = [p["y_time"] for p in self.master_buffer if p["attack"]]
            
            self.plot_safe.set_data(xs_safe, ys_safe)
            self.plot_attack.set_data(xs_att, ys_att)
            
            last_x = self.master_buffer[-1]["x"]
            self.ax1.set_xlim(max(0, last_x - MAX_HISTORY), last_x + 10)

            # QBER Rolling Avg
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
            self.lbl_stats.config(text=f"PHOTONS: {self.engine.total_sent}\nDETECTED: {self.engine.total_detected}\nAVG QBER: {ratio:.1%}")

        self.root.after(GUI_REFRESH_MS, self.run_update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = QTA_Dashboard(root)
    root.mainloop()
    
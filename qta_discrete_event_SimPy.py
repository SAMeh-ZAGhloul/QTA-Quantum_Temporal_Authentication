import tkinter as tk
import simpy
import random
import queue
import collections
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- CONFIGURATION ---
# Physics
PULSE_INTERVAL = 0.1      # Alice sends every 0.1 ticks
FIBER_DELAY = 0.2         # Short flight time
ATTACK_DELAY = 0.6        # Eve adds 0.6 delay (High visibility)
WINDOW_TOLERANCE = 0.15   # +/- 0.15 is the green zone

# System
GUI_REFRESH_MS = 50       # Update screen every 50ms (20 FPS)
MAX_HISTORY = 150         # Keep last 150 photons on screen

class QTA_Physics_Engine:
    def __init__(self, env):
        self.env = env
        self.channel = simpy.Store(env)
        self.eve_active = False
        self.stats = {"sent": 0, "detected": 0, "errors": 0}
        self.gui_queue = queue.Queue()

    def alice_sender(self):
        """Alice generates quantum states."""
        while True:
            photon = {
                "id": self.stats["sent"],
                "bit": random.randint(0, 1),
                "basis": random.randint(0, 1), # 0=Rectilinear, 1=Diagonal
                "t_emit": self.env.now,
                "attacked": False
            }
            self.env.process(self.fiber_link(photon))
            self.stats["sent"] += 1
            yield self.env.timeout(PULSE_INTERVAL)

    def fiber_link(self, photon):
        """The Optical Fiber. Eve lives here."""
        flight_time = FIBER_DELAY + random.gauss(0, 0.01) # Natural jitter
        
        # Check Eve status at this exact moment
        if self.eve_active:
            flight_time += ATTACK_DELAY
            photon["attacked"] = True
            
            # Eve Measure-Resend Attack
            eve_basis = random.randint(0, 1)
            if eve_basis != photon["basis"]:
                # If Eve uses wrong basis, she destroys the bit info
                photon["bit"] = random.randint(0, 1)

        yield self.env.timeout(flight_time)
        self.channel.put(photon)

    def bob_receiver(self):
        """Bob measures and calculates stats."""
        while True:
            photon = yield self.channel.get()
            t_arrival = self.env.now
            
            # Bob Measures
            bob_basis = random.randint(0, 1)
            measured_bit = photon["bit"]
            
            # Quantum Mechanics: Basis mismatch = 50% random
            if bob_basis != photon["basis"]:
                if random.random() < 0.5: measured_bit = 1 - measured_bit
            
            # Sifting: Only keep matching bases for QTA
            if bob_basis == photon["basis"]:
                # 1. Timing Check
                expected = photon["t_emit"] + FIBER_DELAY
                offset = t_arrival - expected
                
                # 2. Error Check
                is_error = (measured_bit != photon["bit"])
                
                self.stats["detected"] += 1
                if is_error: self.stats["errors"] += 1
                
                # Send to GUI
                self.gui_queue.put({
                    "idx": self.stats["detected"],
                    "offset": offset,
                    "error": is_error,
                    "attacked": photon["attacked"]
                })

class QTA_Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("QTA Protocol: Robust Simulation")
        self.root.geometry("1100x750")
        
        # SimPy Setup
        self.env = simpy.Environment()
        self.physics = QTA_Physics_Engine(self.env)
        self.env.process(self.physics.alice_sender())
        self.env.process(self.physics.bob_receiver())
        
        # Data Storage (The Master Buffer)
        # Stores tuples: (index, offset, is_attacked, is_error)
        self.master_buffer = collections.deque(maxlen=MAX_HISTORY)
        
        # GUI Layout
        self._build_gui()
        
        # Start Loop
        self.run_update_loop()

    def _build_gui(self):
        # 1. Control Panel
        panel = tk.Frame(self.root, bg="#1a1a1a", width=250)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(panel, text="QUANTUM\nSECURITY", fg="#00d4ff", bg="#1a1a1a", font=("Impact", 20)).pack(pady=30)
        
        # Interactive Status
        self.lbl_status = tk.Label(panel, text="LINK SECURE", fg="#00ff00", bg="black", 
                                   font=("Consolas", 14, "bold"), width=15, pady=10, relief="sunken")
        self.lbl_status.pack(pady=10)

        self.btn = tk.Button(panel, text="ACTIVATE EVE", bg="#444", fg="white", 
                             font=("Arial", 12, "bold"), height=2, command=self.toggle_eve)
        self.btn.pack(pady=20, padx=15, fill=tk.X)
        
        self.lbl_stats = tk.Label(panel, text="--", fg="#888", bg="#1a1a1a", justify=tk.LEFT, font=("Consolas", 10))
        self.lbl_stats.pack(pady=20)

        # 2. Plot Area
        self.fig = plt.Figure(figsize=(8, 8), dpi=100, facecolor="#f0f0f0")
        gs = self.fig.add_gridspec(2, 1)
        
        # Chart A: Timing (Scatter)
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax1.set_title("Photon Time-of-Arrival Analysis")
        self.ax1.set_ylabel("Timing Offset (ns)")
        self.ax1.set_ylim(-0.5, 1.5) # Space for the jump
        self.ax1.axhspan(-WINDOW_TOLERANCE, WINDOW_TOLERANCE, color="green", alpha=0.15, label="Safety Window")
        
        # Two layers: One for safe dots, one for attack dots
        self.plot_safe, = self.ax1.plot([], [], 'o', color="green", markersize=5, alpha=0.7)
        self.plot_attack, = self.ax1.plot([], [], 'o', color="red", markersize=5, alpha=0.8)
        self.ax1.legend(loc="upper right")
        self.ax1.grid(True, alpha=0.3)

        # Chart B: QBER (Line)
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax2.set_title("Real-time QBER (Bit Error Rate)")
        self.ax2.set_ylabel("Error %")
        self.ax2.set_ylim(0, 0.6)
        self.ax2.axhline(0.05, color="red", linestyle="--", label="Max Threshold (5%)")
        self.plot_qber, = self.ax2.plot([], [], color="#333", lw=2)
        self.ax2.fill_between([], [], color="#333", alpha=0.1) # Just to init fill
        self.ax2.legend(loc="upper right")
        self.ax2.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def toggle_eve(self):
        self.physics.eve_active = not self.physics.eve_active
        if self.physics.eve_active:
            self.lbl_status.config(text="INTERCEPTED", fg="red", bg="#400000")
            self.btn.config(text="DEACTIVATE EVE", bg="#800000")
        else:
            self.lbl_status.config(text="LINK SECURE", fg="#00ff00", bg="black")
            self.btn.config(text="ACTIVATE EVE", bg="#444")

    def run_update_loop(self):
        # 1. Advance Physics
        try:
            self.env.run(until=self.env.now + 0.2)
        except:
            pass

        # 2. Drain Queue into Master Buffer
        while not self.physics.gui_queue.empty():
            d = self.physics.gui_queue.get()
            # Calculate rolling QBER right here
            err_val = 1.0 if d["error"] else 0.0
            
            # Get recent history for QBER avg
            # (We just look at the last N errors in the buffer)
            # Simple approach: Store raw error in buffer
            self.master_buffer.append({
                "x": d["idx"],
                "y_time": d["offset"],
                "attack": d["attacked"],
                "err": err_val
            })

        # 3. Rebuild Plots from Master Buffer
        if len(self.master_buffer) > 1:
            # -- Plot A: Timing --
            # Filter buffer into X/Y lists
            xs_safe = [p["x"] for p in self.master_buffer if not p["attack"]]
            ys_safe = [p["y_time"] for p in self.master_buffer if not p["attack"]]
            
            xs_att = [p["x"] for p in self.master_buffer if p["attack"]]
            ys_att = [p["y_time"] for p in self.master_buffer if p["attack"]]
            
            self.plot_safe.set_data(xs_safe, ys_safe)
            self.plot_attack.set_data(xs_att, ys_att)
            
            # Adjust View
            last_x = self.master_buffer[-1]["x"]
            self.ax1.set_xlim(max(0, last_x - MAX_HISTORY), last_x + 10)

            # -- Plot B: QBER --
            # Calculate moving average over the buffer
            # This is O(N) which is fine for N=150
            qber_x = []
            qber_y = []
            
            # Helper list to calc rolling avg
            recent_errs = collections.deque(maxlen=30) 
            
            # Re-calculate curve for visible points (Smoothed)
            for p in self.master_buffer:
                recent_errs.append(p["err"])
                avg = sum(recent_errs) / len(recent_errs)
                qber_x.append(p["x"])
                qber_y.append(avg)
            
            self.plot_qber.set_data(qber_x, qber_y)
            self.ax2.set_xlim(max(0, last_x - MAX_HISTORY), last_x + 10)
            
            self.canvas.draw()

            # Stats Text
            s = self.physics.stats
            ratio = s['errors']/s['detected'] if s['detected'] > 0 else 0
            self.lbl_stats.config(text=f"PHOTONS: {s['sent']}\nDETECTED: {s['detected']}\nAVG QBER: {ratio:.1%}")

        # 4. Schedule Next Frame
        self.root.after(GUI_REFRESH_MS, self.run_update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = QTA_Dashboard(root)
    root.mainloop()
    
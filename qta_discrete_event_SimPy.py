import tkinter as tk
import simpy
import random
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import collections

# --- CONFIGURATION ---
PULSE_INTERVAL = 0.1      # Fast pulse generation
FIBER_DELAY = 0.1         # almost zero delay for INSTANT reaction
WINDOW_TOLERANCE = 0.1    # +/- tolerance
ATTACK_DELAY = 0.5        # Eve adds 0.5 delay (Massive visible jump)
GUI_REFRESH_MS = 50       # 20 FPS

class QTA_Model:
    def __init__(self, env):
        self.env = env
        self.channel = simpy.Store(env)
        self.eve_active = False
        self.stats = {"sent": 0, "detected": 0, "errors": 0}
        self.data_queue = queue.Queue()

    def alice_sender(self):
        while True:
            photon = {
                "id": self.stats["sent"],
                "bit": random.randint(0, 1),
                "basis": random.randint(0, 1),
                "t_emit": self.env.now,
                "attacked": False
            }
            self.env.process(self.fiber_channel(photon))
            self.stats["sent"] += 1
            yield self.env.timeout(PULSE_INTERVAL)

    def fiber_channel(self, photon):
        flight_time = FIBER_DELAY + random.gauss(0, 0.01)
        
        # CRITICAL: Check Eve status continuously
        if self.eve_active:
            flight_time += ATTACK_DELAY
            photon["attacked"] = True
            # Eve messes up the bits
            eve_basis = random.randint(0, 1)
            if eve_basis != photon["basis"]:
                photon["bit"] = random.randint(0, 1)

        yield self.env.timeout(flight_time)
        self.channel.put(photon)

    def bob_receiver(self):
        while True:
            photon = yield self.channel.get()
            t_arrival = self.env.now
            
            # Measure
            bob_basis = random.randint(0, 1)
            measured_bit = photon["bit"]
            
            if bob_basis != photon["basis"]:
                if random.random() < 0.5: measured_bit = 1 - measured_bit
            
            if bob_basis == photon["basis"]:
                expected = photon["t_emit"] + FIBER_DELAY
                offset = t_arrival - expected
                is_err = (measured_bit != photon["bit"])
                
                self.stats["detected"] += 1
                if is_err: self.stats["errors"] += 1
                
                # Send to GUI
                self.data_queue.put({
                    "idx": self.stats["detected"], # Use detection count as X-axis
                    "offset": offset,
                    "error": is_err,
                    "attacked": photon["attacked"]
                })

class QTA_App:
    def __init__(self, root):
        self.root = root
        self.root.title("QTA Final Simulator")
        self.root.geometry("1100x700")
        
        self.env = simpy.Environment()
        self.model = QTA_Model(self.env)
        self.env.process(self.model.alice_sender())
        self.env.process(self.model.bob_receiver())
        
        self._init_ui()
        self.run_simulation_step()

    def _init_ui(self):
        # --- Controls ---
        p = tk.Frame(self.root, bg="#222", width=250)
        p.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(p, text="QTA CONTROL", fg="white", bg="#222", font=("Arial", 16, "bold")).pack(pady=20)
        
        self.lbl_status = tk.Label(p, text="SECURE", fg="#0f0", bg="black", font=("Arial", 14), width=15, pady=10)
        self.lbl_status.pack(pady=10)

        self.btn = tk.Button(p, text="ACTIVATE EVE", bg="#444", fg="black", font=("Arial", 12), 
                             command=self.toggle_eve, height=2)
        self.btn.pack(pady=20, padx=20, fill=tk.X)
        
        self.lbl_stats = tk.Label(p, text="...", fg="#aaa", bg="#222", justify=tk.LEFT)
        self.lbl_stats.pack(pady=20)

        # --- Graphs ---
        self.fig = plt.Figure(figsize=(8, 8), dpi=100, facecolor="#f0f0f0")
        gs = self.fig.add_gridspec(2, 1)
        
        # GRAPH 1: Timing (The Fix is here - Two separate layers)
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax1.set_title("Photon Timing (Split Layers)")
        self.ax1.set_ylabel("Offset")
        self.ax1.set_ylim(-0.5, 1.0)
        self.ax1.axhspan(-WINDOW_TOLERANCE, WINDOW_TOLERANCE, color="green", alpha=0.15)
        
        # LAYER 1: SAFE (Green dots)
        self.line_safe, = self.ax1.plot([], [], 'o', color="green", markersize=5, label="Valid")
        # LAYER 2: ATTACK (Red dots)
        self.line_attack, = self.ax1.plot([], [], 'o', color="red", markersize=5, label="Attacked")
        self.ax1.legend(loc="upper right")

        # GRAPH 2: QBER
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax2.set_title("QBER %")
        self.ax2.set_ylim(0, 0.6)
        self.ax2.axhline(0.05, color="red", linestyle="--")
        self.line_qber, = self.ax2.plot([], [], color="#333", lw=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Buffers
        self.x_safe = collections.deque(maxlen=100)
        self.y_safe = collections.deque(maxlen=100)
        
        self.x_attack = collections.deque(maxlen=100)
        self.y_attack = collections.deque(maxlen=100)
        
        self.qber_hist = collections.deque(maxlen=100)
        self.qber_x = collections.deque(maxlen=100)

    def toggle_eve(self):
        self.model.eve_active = not self.model.eve_active
        if self.model.eve_active:
            self.lbl_status.config(text="ATTACK ACTIVE", fg="red")
            self.btn.config(text="DEACTIVATE")
        else:
            self.lbl_status.config(text="SECURE", fg="#0f0")
            self.btn.config(text="ACTIVATE EVE")

    def run_simulation_step(self):
        # Force SimPy step
        self.env.run(until=self.env.now + 0.2)
        
        while not self.model.data_queue.empty():
            d = self.model.data_queue.get()
            
            # SPLIT DATA INTO TWO BUCKETS
            if d["attacked"]:
                self.x_attack.append(d["idx"])
                self.y_attack.append(d["offset"])
            else:
                self.x_safe.append(d["idx"])
                self.y_safe.append(d["offset"])
            
            # QBER Logic
            self.qber_x.append(d["idx"])
            # Simple rolling average calculation
            err = 1.0 if d["error"] else 0.0
            # If queue is full, remove old impact from average? 
            # Simpler: just store raw errors and avg list
            self.qber_hist.append(err)

        # Draw Lines
        self.line_safe.set_data(self.x_safe, self.y_safe)
        self.line_attack.set_data(self.x_attack, self.y_attack)
        
        # Calculate Rolling QBER for display
        if self.qber_hist:
            avg = sum(self.qber_hist) / len(self.qber_hist)
            # For visualization, we just plot a straight line of history? 
            # Actually, we need a list of averages. 
            # Let's just plot the current average repeated for simplicity or maintain a separate list
            # REFINED: Just append current average to a plot list.
            current_len = len(self.line_qber.get_xdata())
            new_x = list(self.line_qber.get_xdata()) + [self.model.stats["detected"]]
            new_y = list(self.line_qber.get_ydata()) + [avg]
            
            # Trim
            if len(new_x) > 100:
                new_x = new_x[-100:]
                new_y = new_y[-100:]
                
            self.line_qber.set_data(new_x, new_y)
            
            # Dynamic X-Axis Scroll
            latest_idx = self.model.stats["detected"]
            min_x = max(0, latest_idx - 100)
            self.ax1.set_xlim(min_x, latest_idx + 10)
            self.ax2.set_xlim(min_x, latest_idx + 10)

        self.canvas.draw()
        
        # Stats Text
        s = self.model.stats
        q = s['errors']/s['detected'] if s['detected'] > 0 else 0
        self.lbl_stats.config(text=f"Sent: {s['sent']}\nQBER: {q:.1%}")

        self.root.after(GUI_REFRESH_MS, self.run_simulation_step)

if __name__ == "__main__":
    root = tk.Tk()
    app = QTA_App(root)
    root.mainloop()

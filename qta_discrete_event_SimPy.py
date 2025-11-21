import tkinter as tk
import simpy
import random
import queue
import collections
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- CONFIGURATION ---
PULSE_INTERVAL = 0.1      
FIBER_DELAY = 0.2         
ATTACK_DELAY = 0.8        # Large delay for visibility
WINDOW_TOLERANCE = 0.15   
GUI_REFRESH_MS = 150      # 150ms = ~7 FPS (Smooth enough, but leaves CPU time for buttons)
MAX_HISTORY = 100         

class QTA_Physics_Engine:
    def __init__(self, env):
        self.env = env
        self.channel = simpy.Store(env)
        self.eve_active = False
        self.stats = {"sent": 0, "detected": 0, "errors": 0}
        self.gui_queue = queue.Queue()

    def alice_sender(self):
        while True:
            photon = {
                "id": self.stats["sent"],
                "bit": random.randint(0, 1),
                "basis": random.randint(0, 1), 
                "t_emit": self.env.now,
                "attacked": False
            }
            self.env.process(self.fiber_link(photon))
            self.stats["sent"] += 1
            yield self.env.timeout(PULSE_INTERVAL)

    def fiber_link(self, photon):
        flight_time = FIBER_DELAY + random.gauss(0, 0.01)
        
        # RE-CHECK EVE STATUS EVERY SINGLE PHOTON
        if self.eve_active:
            flight_time += ATTACK_DELAY
            photon["attacked"] = True
            # Eve Attack Logic (BB84 Intercept-Resend)
            eve_basis = random.randint(0, 1)
            if eve_basis != photon["basis"]:
                photon["bit"] = random.randint(0, 1)

        yield self.env.timeout(flight_time)
        self.channel.put(photon)

    def bob_receiver(self):
        while True:
            photon = yield self.channel.get()
            t_arrival = self.env.now
            
            bob_basis = random.randint(0, 1)
            measured_bit = photon["bit"]
            if bob_basis != photon["basis"]:
                if random.random() < 0.5: measured_bit = 1 - measured_bit
            
            if bob_basis == photon["basis"]:
                expected = photon["t_emit"] + FIBER_DELAY
                offset = t_arrival - expected
                is_error = (measured_bit != photon["bit"])
                
                self.stats["detected"] += 1
                if is_error: self.stats["errors"] += 1
                
                self.gui_queue.put({
                    "idx": self.stats["detected"],
                    "offset": offset,
                    "error": is_error,
                    "attacked": photon["attacked"]
                })

class QTA_Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("QTA Final (Keyboard Enabled)")
        self.root.geometry("1100x750")
        
        # Bind Spacebar as backup!
        self.root.bind('<space>', self.toggle_eve)
        
        self.env = simpy.Environment()
        self.physics = QTA_Physics_Engine(self.env)
        self.env.process(self.physics.alice_sender())
        self.env.process(self.physics.bob_receiver())
        
        self.master_buffer = collections.deque(maxlen=MAX_HISTORY)
        
        self._build_gui()
        self.run_update_loop()

    def _build_gui(self):
        # PANEL
        panel = tk.Frame(self.root, bg="#1a1a1a", width=250)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(panel, text="QUANTUM\nSECURITY", fg="#00d4ff", bg="#1a1a1a", font=("Impact", 20)).pack(pady=30)
        
        self.lbl_status = tk.Label(panel, text="LINK SECURE", fg="#00ff00", bg="black", 
                                   font=("Consolas", 14, "bold"), width=15, pady=10, relief="sunken")
        self.lbl_status.pack(pady=10)

        # NOTE: Added hint text
        self.btn = tk.Button(panel, text="ACTIVATE EVE\n(or Press Space)", highlightbackground="#444", 
                             font=("Arial", 12, "bold"), height=3, command=self.toggle_eve_event)
        self.btn.pack(pady=20, padx=15, fill=tk.X)
        
        self.lbl_stats = tk.Label(panel, text="--", fg="#888", bg="#1a1a1a", justify=tk.LEFT, font=("Consolas", 10))
        self.lbl_stats.pack(pady=20)

        # PLOTS
        self.fig = plt.Figure(figsize=(8, 8), dpi=100, facecolor="#f0f0f0")
        gs = self.fig.add_gridspec(2, 1)
        
        # Plot 1
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax1.set_title("Photon Time-of-Arrival")
        self.ax1.set_ylabel("Offset (ns)")
        self.ax1.set_ylim(-0.5, 1.5) 
        self.ax1.axhspan(-WINDOW_TOLERANCE, WINDOW_TOLERANCE, color="green", alpha=0.15)
        self.plot_safe, = self.ax1.plot([], [], 'o', color="green", markersize=5, alpha=0.7)
        self.plot_attack, = self.ax1.plot([], [], 'o', color="red", markersize=5, alpha=0.8)
        self.ax1.grid(True, alpha=0.3)

        # Plot 2
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax2.set_title("QBER (Bit Error Rate)")
        self.ax2.set_ylabel("Error %")
        self.ax2.set_ylim(0, 0.6)
        self.ax2.axhline(0.05, color="red", linestyle="--")
        self.plot_qber, = self.ax2.plot([], [], color="#333", lw=2)
        self.ax2.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Wrapper for button click (ignores the 'event' argument)
    def toggle_eve_event(self):
        self.toggle_eve(None)

    # Main toggle logic
    def toggle_eve(self, event=None):
        print(">>> TOGGLE SIGNAL RECEIVED") # Debug print
        self.physics.eve_active = not self.physics.eve_active
        
        if self.physics.eve_active:
            self.lbl_status.config(text="INTERCEPTED", fg="red")
            self.btn.config(text="DEACTIVATE EVE\n(or Press Space)")
        else:
            self.lbl_status.config(text="LINK SECURE", fg="#00ff00")
            self.btn.config(text="ACTIVATE EVE\n(or Press Space)")

    def run_update_loop(self):
        # Force UI to process pending clicks/keys BEFORE running heavy physics
        self.root.update_idletasks()

        # Run SimPy
        try:
            self.env.run(until=self.env.now + 0.2)
        except:
            pass

        # Drain Queue
        while not self.physics.gui_queue.empty():
            d = self.physics.gui_queue.get()
            err_val = 1.0 if d["error"] else 0.0
            self.master_buffer.append({
                "x": d["idx"],
                "y_time": d["offset"],
                "attack": d["attacked"],
                "err": err_val
            })

        # Update Plot Data
        if len(self.master_buffer) > 1:
            xs_safe = [p["x"] for p in self.master_buffer if not p["attack"]]
            ys_safe = [p["y_time"] for p in self.master_buffer if not p["attack"]]
            xs_att = [p["x"] for p in self.master_buffer if p["attack"]]
            ys_att = [p["y_time"] for p in self.master_buffer if p["attack"]]
            
            self.plot_safe.set_data(xs_safe, ys_safe)
            self.plot_attack.set_data(xs_att, ys_att)
            
            last_x = self.master_buffer[-1]["x"]
            self.ax1.set_xlim(max(0, last_x - MAX_HISTORY), last_x + 10)

            # QBER Line
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
            
            # VITAL FIX: Use draw_idle() instead of draw()
            # This prevents the graph from blocking the button click events
            self.canvas.draw_idle()

            # Stats
            s = self.physics.stats
            ratio = s['errors']/s['detected'] if s['detected'] > 0 else 0
            self.lbl_stats.config(text=f"PHOTONS: {s['sent']}\nDETECTED: {s['detected']}\nAVG QBER: {ratio:.1%}")

        self.root.after(GUI_REFRESH_MS, self.run_update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = QTA_Dashboard(root)
    root.mainloop()

    
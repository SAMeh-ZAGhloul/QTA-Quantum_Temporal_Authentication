import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import matplotlib.gridspec as gridspec

# --- CONFIGURATION ---
# Physics Parameters
FRAME_SIZE = 50         # Pulses per animation frame
WINDOW_SIZE = 200       # Rolling window for graphs
SAFE_QBER = 0.05        # 5% Max allowed error
SAFE_TIME_WINDOW = 20   # +/- 20 picoseconds allowed
BASE_JITTER = 5.0       # Standard deviation of fiber jitter (ps)

# Attack Parameters (Eve)
EVE_DELAY_MEAN = 40.0   # Eve adds 40ps processing delay
EVE_DELAY_JITTER = 10.0 # Eve adds extra noise
INTERCEPT_RESEND_ERROR = 0.25 # Theoretical error rate for BB84 intercept-resend

class QTASimulator:
    def __init__(self):
        # State
        self.is_eve_active = False
        self.frame_count = 0
        
        # Data Buffers for Plotting
        self.qber_history = deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
        self.toa_history = deque([], maxlen=1000) # Keep last 1000 photon arrivals
        self.score_history = deque([1.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
        
    def toggle_eve(self, event):
        """Event handler to toggle attack mode."""
        if event.key == 'e':
            self.is_eve_active = not self.is_eve_active
            state = "ACTIVE" if self.is_eve_active else "INACTIVE"
            print(f"[SIM] Eve is now {state}")

    def run_physics_step(self):
        """Simulates one frame of photon transmission."""
        # 1. Alice prepares states (0=Rectilinear, 1=Diagonal)
        alice_bases = np.random.randint(0, 2, FRAME_SIZE)
        alice_bits = np.random.randint(0, 2, FRAME_SIZE)
        
        # 2. Transmission Channel
        # Base timing is 0 (perfect sync) + Jitter
        toa = np.random.normal(0, BASE_JITTER, FRAME_SIZE)
        bob_bases = np.random.randint(0, 2, FRAME_SIZE) # Bob chooses random bases
        bob_bits = alice_bits.copy() # Start with perfect copy
        
        # 3. EVE INTERVENTION
        if self.is_eve_active:
            # Eve chooses random bases to measure
            eve_bases = np.random.randint(0, 2, FRAME_SIZE)
            
            # Logic: If Eve picks wrong basis, she randomizes the bit
            # Mask where Eve matches Alice (No error introduced to bit yet)
            eve_mismatch = (eve_bases != alice_bases)
            
            # Where mismatch, Eve sends random bit (50% chance of flipping Alice's bit)
            # This results in ~25% total QBER on Bob's side
            noise = np.random.randint(0, 2, FRAME_SIZE)
            bob_bits[eve_mismatch] = noise[eve_mismatch]
            
            # Eve adds Delay (Time of Flight / Processing)
            delay = np.random.normal(EVE_DELAY_MEAN, EVE_DELAY_JITTER, FRAME_SIZE)
            toa += delay

        # 4. Bob Measures
        # We only keep data where Alice and Bob bases matched (Sifting)
        sift_mask = (alice_bases == bob_bases)
        valid_indices = np.where(sift_mask)[0]
        
        if len(valid_indices) == 0:
            return 0, []

        # Calculate Errors (QBER)
        # Compare Alice's sent bits vs Bob's measured bits (after Eve potentially messed them up)
        errors = np.sum(alice_bits[valid_indices] != bob_bits[valid_indices])
        total_sifted = len(valid_indices)
        current_qber = errors / total_sifted if total_sifted > 0 else 0
        
        # Calculate Timing Compliance
        # Count how many photons arrived within the strict time window
        # Using absolute value of ToA
        timing_compliant = np.sum(np.abs(toa[valid_indices]) <= SAFE_TIME_WINDOW)
        timing_score = timing_compliant / total_sifted if total_sifted > 0 else 0

        # Auth Decision (Requires Low QBER AND Good Timing)
        is_auth_success = (current_qber < SAFE_QBER) and (timing_score > 0.90)
        
        return current_qber, toa[valid_indices], 1.0 if is_auth_success else 0.0

    def update(self, frame):
        """Animation Loop"""
        qber, toas, score = self.run_physics_step()
        
        # Update Buffers
        self.qber_history.append(qber)
        self.toa_history.extend(toas)
        self.score_history.append(score)
        
        # --- UPDATE PLOTS ---
        
        # 1. QBER Line
        line_qber.set_data(range(WINDOW_SIZE), self.qber_history)
        if self.is_eve_active:
            ax1.set_title(f"QBER Monitor (ATTACK DETECTED) | Current: {qber:.2%}", color='red', weight='bold')
        else:
            ax1.set_title(f"QBER Monitor (Secure) | Current: {qber:.2%}", color='green')

        # 2. Timing Histogram
        ax2.cla()
        ax2.set_title("Time-of-Arrival (ToA) Distribution")
        ax2.set_xlabel("Offset (ps)")
        ax2.set_ylabel("Count")
        ax2.set_xlim(-100, 150) # View range
        
        # Draw Safety Window
        ax2.axvspan(-SAFE_TIME_WINDOW, SAFE_TIME_WINDOW, color='green', alpha=0.2, label="Valid Window")
        
        # Draw Histogram
        if len(self.toa_history) > 0:
            n, bins, patches = ax2.hist(self.toa_history, bins=30, color='blue', alpha=0.7)
            
            # Highlight delayed photons in red
            # This is a visual trick: usually hard to change color of hist bars individually in efficient loops
            # but useful for static analysis. Here we stick to blue for speed.
            
        # 3. Auth Score
        ax3.cla()
        ax3.set_title("Authentication Health")
        ax3.set_ylim(-0.1, 1.1)
        ax3.plot(self.score_history, color='purple', lw=2)
        ax3.fill_between(range(WINDOW_SIZE), self.score_history, color='purple', alpha=0.3)
        ax3.text(5, 0.1, "Press 'E' to toggle Eve Attack", fontsize=8, style='italic')

        return line_qber,

# --- SETUP PLOT ---
sim = QTASimulator()

fig = plt.figure(figsize=(12, 8))
fig.canvas.mpl_connect('key_press_event', sim.toggle_eve)
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# Top: QBER (Spans width)
ax1 = plt.subplot(gs[0, :])
ax1.set_ylim(0, 0.5)
ax1.set_ylabel("Error Rate")
ax1.axhline(y=SAFE_QBER, color='r', linestyle='--', label="Max Threshold")
line_qber, = ax1.plot([], [], lw=2)
ax1.legend(loc="upper right")

# Bottom Left: ToA
ax2 = plt.subplot(gs[1, 0])

# Bottom Right: Score
ax3 = plt.subplot(gs[1, 1])

plt.tight_layout()
ani = FuncAnimation(fig, sim.update, interval=50, blit=False)

print("--- QTA SIMULATION STARTED ---")
print("1. Watch the Green Window (Valid Timing)")
print("2. Watch the QBER (Should be < 5%)")
print("3. PRESS 'e' TO LAUNCH INTERCEPT ATTACK")
plt.show()

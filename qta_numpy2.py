#!/usr/bin/env python3
# qta_sim_refactored.py
#
# Refactored Quantum Temporal Authentication (QTA) simulator:
# - Single window with 3 subplots
# - Runs continuously until user exits
# - Interactive Eve toggle (button + keyboard 'e')
# - Fixed GUI layout to prevent text overlap
#
# Dependencies: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random
import math


# ======================================
# Configuration
# ======================================

class QTAConfig:
    def __init__(self):
        self.pulses_per_frame = 64
        self.frame_duration_ns = 1e6

        self.mean_photons_mu = 0.1
        self.channel_loss_prob = 0.2
        self.detector_efficiency = 0.9
        self.detector_jitter_ps = 50.0
        self.timing_window_ps = 100.0

        self.qber_threshold = 0.05

        self.eve_active = True
        self.eve_intercept_prob = 0.2
        self.eve_delay_ps = 500.0

        self.decoy_fraction = 0.1
        self.seed = 12345

        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def ps_to_ns(ps):
        return ps * 1e-3


# ======================================
# Entities: Alice, Eve, Bob
# ======================================

class Alice:
    def __init__(self, cfg):
        self.cfg = cfg
        self.P = cfg.pulses_per_frame
        self.frame_times = np.linspace(0, cfg.frame_duration_ns, self.P, endpoint=False)

    def emit(self):
        bases = np.random.choice([0, 1], size=self.P)
        bits = np.random.choice([0, 1], size=self.P)
        photon_present = np.random.rand(self.P) < (1 - np.exp(-self.cfg.mean_photons_mu))
        return bases, bits, photon_present, self.frame_times.copy()


class Eve:
    def __init__(self, cfg):
        self.cfg = cfg

    def intercept_resend(self, bases, bits, photon_present, survived_channel, emitted_times_ns):
        P = len(bases)
        eve_intercept = np.zeros(P, dtype=bool)
        eve_resend_times = np.full(P, np.nan)
        eve_error = np.zeros(P, dtype=bool)

        if not self.cfg.eve_active:
            return bits, photon_present, survived_channel, eve_intercept, eve_resend_times, eve_error

        eve_intercept = (np.random.rand(P) < self.cfg.eve_intercept_prob) & survived_channel
        eve_bases = np.random.choice([0, 1], size=P)

        for i in np.where(eve_intercept)[0]:
            if eve_bases[i] == bases[i]:
                resend_bit = bits[i]
            else:
                resend_bit = np.random.choice([0, 1])
                if resend_bit != bits[i]:
                    eve_error[i] = True

            eve_resend_times[i] = emitted_times_ns[i] + self.cfg.ps_to_ns(self.cfg.eve_delay_ps)
            survived_channel[i] = np.random.rand() > self.cfg.channel_loss_prob
            photon_present[i] = survived_channel[i]
            bits[i] = resend_bit

        return bits, photon_present, survived_channel, eve_intercept, eve_resend_times, eve_error


class Bob:
    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, bases, bits, photon_present, emitted_times_ns, eve_intercept, eve_resend_times):
        P = len(bases)
        detected = photon_present & (np.random.rand(P) < self.cfg.detector_efficiency)
        detection_times = np.full(P, np.nan)

        for i in range(P):
            if not detected[i]:
                continue
            base_time = emitted_times_ns[i]
            if eve_intercept[i] and not math.isnan(eve_resend_times[i]):
                base_time = eve_resend_times[i]
            jitter_ns = np.random.normal(0, self.cfg.detector_jitter_ps * 1e-3)
            detection_times[i] = base_time + jitter_ns

        bob_bases = np.random.choice([0, 1], size=P)
        bob_outcomes = np.full(P, -1, dtype=int)
        measurement_error_rate = 0.005

        for i in range(P):
            if not detected[i]:
                continue
            
            if bob_bases[i] == bases[i]:
                if np.random.rand() < (1 - measurement_error_rate):
                    bob_outcomes[i] = bits[i]
                else:
                    bob_outcomes[i] = 1 - bits[i]
            else:
                bob_outcomes[i] = np.random.choice([0, 1])

        matched_basis = detected & (bob_bases == bases)
        matched_count = np.sum(matched_basis)
        
        if matched_count > 0:
            errors = np.sum(bob_outcomes[matched_basis] != bits[matched_basis])
            qber = errors / matched_count
        else:
            qber = float('nan')
            errors = 0

        valid = np.where(detected)[0]
        if valid.size > 0:
            delays_ps = (detection_times[valid] - emitted_times_ns[valid]) * 1e3
            mean_delay_ps = float(np.mean(delays_ps))
        else:
            mean_delay_ps = float('nan')

        return detected, detection_times, qber, mean_delay_ps, matched_count, int(errors)


# ======================================
# QTA Simulation
# ======================================

class QTASimulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.alice = Alice(cfg)
        self.eve = Eve(cfg)
        self.bob = Bob(cfg)

        self.qber_hist = []
        self.delay_hist = []
        self.idx_hist = []
        self.detect_hist = []
        self.matched_hist = []
        self.error_hist = []
        self.frame_counter = 0
        
        self.max_history = 100
        self.setup_visualization()
        
        self.running = True
        self.paused = False

    def setup_visualization(self):
        """Setup single window with optimized layout"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('QTA Simulator - Press E to toggle Eve | Q to Quit')
        
        # Optimized grid layout with more spacing
        gs = self.fig.add_gridspec(4, 2, 
                                   height_ratios=[2, 1.5, 1.5, 1.2],
                                   hspace=0.45, wspace=0.35,
                                   left=0.08, right=0.96, 
                                   top=0.94, bottom=0.10)
        
        # Main timeline plot (top row, spans both columns)
        self.ax_timeline = self.fig.add_subplot(gs[0, :])
        
        # QBER plot (second row, left)
        self.ax_qber = self.fig.add_subplot(gs[1, 0])
        
        # Delay plot (second row, right)
        self.ax_delay = self.fig.add_subplot(gs[1, 1])
        
        # Statistics panels (third row)
        self.ax_stats_left = self.fig.add_subplot(gs[2, 0])
        self.ax_stats_left.axis('off')
        
        self.ax_stats_right = self.fig.add_subplot(gs[2, 1])
        self.ax_stats_right.axis('off')
        
        # Control info (fourth row, spans both)
        self.ax_controls = self.fig.add_subplot(gs[3, :])
        self.ax_controls.axis('off')
        
        # Eve toggle button - positioned at bottom
        ax_button = plt.axes([0.44, 0.02, 0.12, 0.035])
        self.btn_eve = Button(ax_button, 'Eve: ON', color='red', hovercolor='darkred')
        self.btn_eve.on_clicked(self.toggle_eve)
        
        # Keyboard event handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # Initialize text elements
        self.stats_text_left = self.ax_stats_left.text(
            0.05, 0.5, '', 
            transform=self.ax_stats_left.transAxes,
            fontsize=10, verticalalignment='center',
            family='monospace'
        )
        
        self.stats_text_right = self.ax_stats_right.text(
            0.05, 0.5, '', 
            transform=self.ax_stats_right.transAxes,
            fontsize=10, verticalalignment='center',
            family='monospace'
        )
        
        self.controls_text = self.ax_controls.text(
            0.5, 0.5, '', 
            transform=self.ax_controls.transAxes,
            fontsize=9, verticalalignment='center',
            horizontalalignment='center',
            family='monospace'
        )

    def toggle_eve(self, event=None):
        """Toggle Eve's attack on/off"""
        self.cfg.eve_active = not self.cfg.eve_active
        color = 'red' if self.cfg.eve_active else 'green'
        text = 'Eve: ON' if self.cfg.eve_active else 'Eve: OFF'
        self.btn_eve.label.set_text(text)
        self.btn_eve.color = color
        self.btn_eve.hovercolor = 'darkred' if self.cfg.eve_active else 'darkgreen'
        print(f"\n{'='*50}")
        print(f"EVE ATTACK: {'ACTIVATED' if self.cfg.eve_active else 'DEACTIVATED'}")
        print(f"{'='*50}\n")

    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'e':
            self.toggle_eve()
        elif event.key == 'q':
            print("\nUser requested exit. Shutting down...")
            self.running = False
            plt.close(self.fig)
        elif event.key == ' ':
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RESUMED"
            print(f"\nSimulation {status}")

    def on_close(self, event):
        """Handle window close event"""
        print("\nWindow closed. Shutting down...")
        self.running = False

    def simulate_frame(self):
        """Simulate one frame"""
        bases, bits, photon_present, emit_t = self.alice.emit()
        survived = photon_present & (np.random.rand(self.cfg.pulses_per_frame) >
                                     self.cfg.channel_loss_prob)
        bits, photon_present, survived, eve_int, eve_times, eve_err = \
            self.eve.intercept_resend(bases, bits, photon_present, survived, emit_t)
        detected, det_t, qber, mean_delay, matched_count, error_count = \
            self.bob.detect(bases, bits, photon_present, emit_t, eve_int, eve_times)

        return {
            "frame": self.frame_counter,
            "emit": emit_t,
            "detected": detected,
            "det_times": det_t,
            "eve": eve_int,
            "qber": qber,
            "mean_delay": mean_delay,
            "matched_count": matched_count,
            "error_count": error_count
        }

    def update_plots(self, data):
        """Update all visualizations"""
        emit = data["emit"]
        detected = data["detected"]
        det_t = data["det_times"]
        eve = data["eve"]
        
        P = self.cfg.pulses_per_frame
        frame_ns = self.cfg.frame_duration_ns

        # --- Timeline Plot ---
        self.ax_timeline.clear()
        self.ax_timeline.set_xlim(0, frame_ns)
        self.ax_timeline.set_ylim(-1, P + 1)
        
        eve_status = "[EVE ACTIVE]" if self.cfg.eve_active else "[EVE INACTIVE]"
        qber_str = f"{data['qber']:.4f}" if not math.isnan(data['qber']) else "N/A"
        title = (f"Frame {self.frame_counter} | {eve_status} | "
                f"Detected: {np.sum(detected)}/{P} | "
                f"Matched: {data['matched_count']} | "
                f"Errors: {data['error_count']} | "
                f"QBER: {qber_str}")
        self.ax_timeline.set_title(title, fontsize=11, fontweight='bold', pad=10)
        self.ax_timeline.set_xlabel("Time within frame (ns)", fontsize=9)
        self.ax_timeline.set_ylabel("Pulse index", fontsize=9)
        self.ax_timeline.grid(alpha=0.2)

        self.ax_timeline.scatter(emit, np.arange(P), s=15, alpha=0.5, 
                                label='Alice emissions', color='blue')
        
        idx_det = np.where(detected)[0]
        if idx_det.size:
            self.ax_timeline.scatter(det_t[idx_det], idx_det, s=35, alpha=0.7,
                                   label='Bob detections', color='green', 
                                   edgecolors='black', linewidths=0.5)

        idx_eve = np.where(eve & detected)[0]
        if idx_eve.size:
            self.ax_timeline.scatter(det_t[idx_eve], idx_eve, marker='X', s=80,
                                   label='Eve intercepts', color='red', 
                                   linewidths=2, zorder=5)

        center = frame_ns / 2
        tw = self.cfg.ps_to_ns(self.cfg.timing_window_ps)
        self.ax_timeline.axvline(center - tw, alpha=0.15, linestyle='--', 
                               color='gray', label='Timing window')
        self.ax_timeline.axvline(center + tw, alpha=0.15, linestyle='--', color='gray')
        
        self.ax_timeline.legend(loc='upper right', fontsize=8, framealpha=0.9)

        # --- QBER Plot ---
        self.ax_qber.clear()
        
        display_idx = self.idx_hist[-self.max_history:]
        display_qber = self.qber_hist[-self.max_history:]
        
        valid_indices = [(i, q) for i, q in zip(display_idx, display_qber) if not math.isnan(q)]
        if valid_indices:
            valid_idx, valid_qber = zip(*valid_indices)
            
            self.ax_qber.plot(valid_idx, valid_qber, marker='o', 
                            markersize=4, alpha=0.7, linewidth=1.5, color='#2E86DE')
            self.ax_qber.axhline(self.cfg.qber_threshold, alpha=0.5, 
                               linestyle='--', color='red', linewidth=2,
                               label=f'Threshold: {self.cfg.qber_threshold}')
            
            if valid_qber[-1] > self.cfg.qber_threshold:
                self.ax_qber.set_facecolor('#330000')
            else:
                self.ax_qber.set_facecolor('#003300')
            
            self.ax_qber.set_ylim(0, max(0.2, max(valid_qber) * 1.2))
        
        self.ax_qber.set_title("QBER History", fontsize=10, fontweight='bold', pad=8)
        self.ax_qber.set_xlabel("Frame", fontsize=9)
        self.ax_qber.set_ylabel("QBER", fontsize=9)
        self.ax_qber.grid(alpha=0.2)
        self.ax_qber.legend(fontsize=7, framealpha=0.9)

        # --- Delay Plot ---
        self.ax_delay.clear()
        
        display_delay = self.delay_hist[-self.max_history:]
        valid_delay_data = [(i, d) for i, d in zip(display_idx, display_delay) if not math.isnan(d)]
        
        if valid_delay_data:
            valid_idx_d, valid_delay = zip(*valid_delay_data)
            
            self.ax_delay.plot(valid_idx_d, valid_delay, marker='o', 
                             markersize=4, alpha=0.7, linewidth=1.5, color='#EE5A24')
            self.ax_delay.axhline(0, alpha=0.3, linestyle='-', color='gray')
            
            if abs(valid_delay[-1]) > self.cfg.timing_window_ps:
                self.ax_delay.set_facecolor('#330000')
            else:
                self.ax_delay.set_facecolor('#003300')
            
            max_abs = max(abs(min(valid_delay)), abs(max(valid_delay)))
            self.ax_delay.set_ylim(-max_abs * 1.2, max_abs * 1.2)
        
        self.ax_delay.set_title("Timing Delays", fontsize=10, fontweight='bold', pad=8)
        self.ax_delay.set_xlabel("Frame", fontsize=9)
        self.ax_delay.set_ylabel("Mean delay (ps)", fontsize=9)
        self.ax_delay.grid(alpha=0.2)

        # --- Statistics Display (Split into two panels) ---
        total_frames = len(self.idx_hist)
        valid_qber_values = [q for q in self.qber_hist if not math.isnan(q)]
        avg_qber = np.mean(valid_qber_values) if valid_qber_values else float('nan')
        valid_delay_values = [d for d in self.delay_hist if not math.isnan(d)]
        avg_delay = np.mean(valid_delay_values) if valid_delay_values else float('nan')
        total_detected = sum(self.detect_hist)
        avg_detected = total_detected / total_frames if total_frames else 0
        total_matched = sum(self.matched_hist)
        avg_matched = total_matched / total_frames if total_frames else 0
        total_errors = sum(self.error_hist)
        auth_success = sum(1 for q in valid_qber_values if q <= self.cfg.qber_threshold)
        auth_rate = (auth_success / len(valid_qber_values) * 100) if valid_qber_values else 0
        
        eve_indicator = "[ACTIVE]" if self.cfg.eve_active else "[INACTIVE]"
        current_qber_str = f"{data['qber']:.4f}" if not math.isnan(data['qber']) else "N/A"
        avg_qber_str = f"{avg_qber:.4f}" if not math.isnan(avg_qber) else "N/A"
        current_delay_str = f"{data['mean_delay']:.2f}" if not math.isnan(data['mean_delay']) else "N/A"
        avg_delay_str = f"{avg_delay:.2f}" if not math.isnan(avg_delay) else "N/A"
        
        # Left statistics panel
        stats_left = f"""Frame: {self.frame_counter:,}
Eve: {eve_indicator}

Detection:
  Detected: {total_detected:,}
  Avg/frame: {avg_detected:.1f}/{P}
  Matched: {total_matched:,}
  Avg matched: {avg_matched:.1f}
  Total errors: {total_errors:,}"""
        
        # Right statistics panel
        stats_right = f"""QBER (matched bases):
  Current: {current_qber_str}
  Average: {avg_qber_str}
  Threshold: {self.cfg.qber_threshold}
  Auth rate: {auth_rate:.1f}%

Timing:
  Cur delay: {current_delay_str} ps
  Avg delay: {avg_delay_str} ps
  Window: +/-{self.cfg.timing_window_ps} ps"""
        
        self.stats_text_left.set_text(stats_left)
        self.stats_text_right.set_text(stats_right)
        
        # Color based on security
        if not math.isnan(data['qber']) and data['qber'] > self.cfg.qber_threshold:
            self.stats_text_left.set_color('red')
            self.stats_text_right.set_color('red')
        else:
            self.stats_text_left.set_color('lime')
            self.stats_text_right.set_color('lime')
        
        # Controls info
        controls_info = "Controls: [E] Toggle Eve | [Space] Pause/Resume | [Q] Quit"
        self.controls_text.set_text(controls_info)
        self.controls_text.set_color('cyan')

    def run(self):
        """Main simulation loop"""
        print("\n" + "="*80)
        print("QUANTUM TEMPORAL AUTHENTICATION SIMULATOR")
        print("="*80)
        print("\nStarting continuous simulation...")
        print("\nControls:")
        print("  • Press [E] or click button to toggle Eve attack")
        print("  • Press [Space] to pause/resume")
        print("  • Press [Q] or close window to quit")
        print("\n" + "="*80 + "\n")
        
        plt.ion()
        plt.show(block=False)
        
        try:
            while self.running:
                if not self.paused:
                    data = self.simulate_frame()
                    
                    self.idx_hist.append(self.frame_counter)
                    self.qber_hist.append(data["qber"])
                    self.delay_hist.append(data["mean_delay"])
                    self.detect_hist.append(np.sum(data["detected"]))
                    self.matched_hist.append(data["matched_count"])
                    self.error_hist.append(data["error_count"])
                    
                    self.update_plots(data)
                    self.frame_counter += 1
                    
                    if self.frame_counter % 100 == 0:
                        recent_qber = [q for q in self.qber_hist[-100:] if not math.isnan(q)]
                        avg_recent_qber = np.mean(recent_qber) if recent_qber else float('nan')
                        avg_recent_detect = np.mean(self.detect_hist[-100:])
                        avg_recent_matched = np.mean(self.matched_hist[-100:])
                        
                        qber_str = f"{avg_recent_qber:.4f}" if not math.isnan(avg_recent_qber) else "N/A"
                        print(f"[Frame {self.frame_counter}] "
                              f"QBER: {qber_str} | "
                              f"Det: {avg_recent_detect:.1f}/{self.cfg.pulses_per_frame} | "
                              f"Match: {avg_recent_matched:.1f}")
                
                plt.pause(0.05)
                
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected. Shutting down...")
        except Exception as e:
            print(f"\n\nError occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup and print final summary"""
        print("\n" + "="*80)
        print("SIMULATION COMPLETED")
        print("="*80)
        print(f"\nTotal Frames: {self.frame_counter:,}")
        print(f"Total Detected: {sum(self.detect_hist):,}")
        print(f"Total Matched: {sum(self.matched_hist):,}")
        print(f"Total Errors: {sum(self.error_hist):,}")
        
        valid_qber = [q for q in self.qber_hist if not math.isnan(q)]
        if valid_qber:
            print(f"\nQBER: Avg={np.mean(valid_qber):.4f}, "
                  f"Min={min(valid_qber):.4f}, Max={max(valid_qber):.4f}")
        
        valid_delays = [d for d in self.delay_hist if not math.isnan(d)]
        if valid_delays:
            print(f"Delay: Avg={np.mean(valid_delays):.2f} ps")
        
        auth_success = sum(1 for q in valid_qber if q <= self.cfg.qber_threshold)
        if valid_qber:
            print(f"Auth Success: {auth_success/len(valid_qber)*100:.1f}%")
        
        print("\n" + "="*80 + "\n")
        plt.ioff()
        plt.close('all')


# ======================================
# Main
# ======================================

if __name__ == "__main__":
    cfg = QTAConfig()
    sim = QTASimulator(cfg)
    sim.run()
    
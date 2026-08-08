import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_core import (K_PLANT, TAU_PLANT, PIDController, run_scenario,
                       compute_metrics, moving_average_filter, low_pass_filter)

K = K_PLANT
TAU = TAU_PLANT
Kp, Ki = 0.24667, 0.010974

# ---------------------------------------------------------------------------
# (a) PI vs PID  (scenario 1 conditions: step 25 -> 40)
#
# NOTE on design: for the PID's closed loop, the characteristic equation is
#   (tau + K*Kd) s^2 + (1 + K*Kp) s + K*Ki = 0
# i.e. adding Kd increases the *effective* plant inertia (tau -> tau+K*Kd).
# If Kp, Ki are left unchanged, the damping ratio silently drops and the
# response gets WORSE (more overshoot) -- not because derivative action is
# bad, but because it was not retuned. To make a fair comparison we retune
# Kp, Ki for the PID so both controllers target the *same* closed-loop
# poles (zeta=0.9, wn as before); this isolates what Kd actually buys us.
# ---------------------------------------------------------------------------
zeta = 0.9
wn = 4.0 / (zeta * 60.0)
Kd = 3.0
tau_eff = TAU + K * Kd
Ki_pid = wn**2 * tau_eff / K
Kp_pid = (2 * zeta * wn * tau_eff - 1) / K
print(f"Retuned PID gains for same closed-loop poles: Kp={Kp_pid:.5f}, "
      f"Ki={Ki_pid:.6f}, Kd={Kd}")

ctrl_pi = PIDController(Kp=Kp, Ki=Ki, Kd=0.0, anti_windup=True)
res_pi = run_scenario(lambda t: 40.0, lambda t: 25.0, lambda t: TAU,
                       ctrl_pi, T0=25.0, K=K)

ctrl_pid = PIDController(Kp=Kp_pid, Ki=Ki_pid, Kd=Kd, anti_windup=True)
res_pid = run_scenario(lambda t: 40.0, lambda t: 25.0, lambda t: TAU,
                        ctrl_pid, T0=25.0, K=K)

m_pi = compute_metrics(res_pi["t"], res_pi["T"], 40.0, 25.0, 0.0)
m_pid = compute_metrics(res_pid["t"], res_pid["T"], 40.0, 25.0, 0.0)
print("PI  metrics:", m_pi)
print("PID metrics (retuned, same poles):", m_pid)

# Noise sensitivity: same noise realization through PI vs (retuned) PID --
# the derivative term amplifies high-frequency measurement noise.
ctrl_pi_n = PIDController(Kp=Kp, Ki=Ki, Kd=0.0, anti_windup=True)
res_pi_n = run_scenario(lambda t: 40.0, lambda t: 25.0, lambda t: TAU,
                         ctrl_pi_n, T0=25.0, K=K, noise_std=0.15, seed=7)
ctrl_pid_n = PIDController(Kp=Kp_pid, Ki=Ki_pid, Kd=Kd, anti_windup=True)
res_pid_n = run_scenario(lambda t: 40.0, lambda t: 25.0, lambda t: TAU,
                          ctrl_pid_n, T0=25.0, K=K, noise_std=0.15, seed=7)
print(f"std(du) PI  with noise: {np.std(np.diff(res_pi_n['u'])):.5f}")
print(f"std(du) PID with noise: {np.std(np.diff(res_pid_n['u'])):.5f}")

fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axs[0].plot(res_pi["t"], res_pi["T"], label="PI", color="tab:red")
axs[0].plot(res_pid["t"], res_pid["T"], label=f"PID (Kd={Kd})", color="tab:green")
axs[0].plot(res_pi["t"], res_pi["sp"], "--", color="black", label="Setpoint")
axs[0].set_ylabel("دما (°C)")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

axs[1].plot(res_pi["t"], res_pi["u"] * 100, label="PI", color="tab:red")
axs[1].plot(res_pid["t"], res_pid["u"] * 100, label=f"PID (Kd={Kd})", color="tab:green")
axs[1].set_ylabel("u (%)")
axs[1].set_xlabel("زمان (s)")
axs[1].legend()
axs[1].grid(True, alpha=0.3)
fig.suptitle("مقایسه‌ی کنترل‌کننده PI و PID (سناریوی ۱)")
fig.tight_layout()
fig.savefig("bonus_pi_vs_pid.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# (b) Anti-windup ON vs OFF  (scenario 4 conditions: bigger step -> more
#     saturation -> windup becomes visible)
# ---------------------------------------------------------------------------
def sp_fn_step(t):
    return 40.0 if t < 150.0 else 60.0

ctrl_aw_on = PIDController(Kp=Kp, Ki=Ki, Kd=0.0, anti_windup=True)
res_aw_on = run_scenario(sp_fn_step, lambda t: 25.0, lambda t: TAU,
                          ctrl_aw_on, T0=25.0, K=K)

ctrl_aw_off = PIDController(Kp=Kp, Ki=Ki, Kd=0.0, anti_windup=False)
res_aw_off = run_scenario(sp_fn_step, lambda t: 25.0, lambda t: TAU,
                           ctrl_aw_off, T0=25.0, K=K)

m_aw_on = compute_metrics(res_aw_on["t"], res_aw_on["T"], 60.0, 40.0, 150.0)
m_aw_off = compute_metrics(res_aw_off["t"], res_aw_off["T"], 60.0, 40.0, 150.0)
print("Anti-windup ON  metrics (2nd step):", m_aw_on)
print("Anti-windup OFF metrics (2nd step):", m_aw_off)

fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axs[0].plot(res_aw_on["t"], res_aw_on["T"], label="Anti-Windup فعال", color="tab:blue")
axs[0].plot(res_aw_off["t"], res_aw_off["T"], label="Anti-Windup غیرفعال", color="tab:red")
axs[0].plot(res_aw_on["t"], res_aw_on["sp"], "--", color="black", label="Setpoint")
axs[0].set_ylabel("دما (°C)")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

axs[1].plot(res_aw_on["t"], res_aw_on["u"] * 100, label="Anti-Windup فعال", color="tab:blue")
axs[1].plot(res_aw_off["t"], res_aw_off["u"] * 100, label="Anti-Windup غیرفعال", color="tab:red")
axs[1].set_ylabel("u (%)")
axs[1].set_xlabel("زمان (s)")
axs[1].legend()
axs[1].grid(True, alpha=0.3)
fig.suptitle("اثر Anti-Windup هنگام اشباع طولانی کنترل‌کننده (سناریوی ۴)")
fig.tight_layout()
fig.savefig("bonus_anti_windup.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# (c) Measurement noise + simple filtering (moving average)
# ---------------------------------------------------------------------------
noise_std = 0.3  # deg C

ctrl_noisy_raw = PIDController(Kp=Kp, Ki=Ki, Kd=0.0, anti_windup=True)
res_noisy_raw = run_scenario(lambda t: 40.0, lambda t: 25.0, lambda t: TAU,
                              ctrl_noisy_raw, T0=25.0, K=K,
                              noise_std=noise_std, filt=None, seed=1)

ctrl_noisy_filt = PIDController(Kp=Kp, Ki=Ki, Kd=0.0, anti_windup=True)
res_noisy_filt = run_scenario(lambda t: 40.0, lambda t: 25.0, lambda t: TAU,
                               ctrl_noisy_filt, T0=25.0, K=K,
                               noise_std=noise_std,
                               filt=moving_average_filter(10), seed=1)

u_std_raw = np.std(np.diff(res_noisy_raw["u"]))
u_std_filt = np.std(np.diff(res_noisy_filt["u"]))
print(f"std(du) without filter: {u_std_raw:.5f}")
print(f"std(du) with moving-average(10) filter: {u_std_filt:.5f}")

fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axs[0].plot(res_noisy_raw["t"], res_noisy_raw["Tmeas"], color="lightcoral",
            alpha=0.6, label="اندازه‌گیری خام (نویزی)")
axs[0].plot(res_noisy_filt["t"], res_noisy_filt["Tmeas"], color="tab:blue",
            label="اندازه‌گیری فیلتر شده (Moving Average, N=10)")
axs[0].plot(res_noisy_raw["t"], res_noisy_raw["sp"], "--", color="black", label="Setpoint")
axs[0].set_ylabel("دما (°C)")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

axs[1].plot(res_noisy_raw["t"], res_noisy_raw["u"] * 100, color="lightcoral",
            alpha=0.7, label="u بدون فیلتر")
axs[1].plot(res_noisy_filt["t"], res_noisy_filt["u"] * 100, color="tab:blue",
            label="u با فیلتر")
axs[1].set_ylabel("u (%)")
axs[1].set_xlabel("زمان (s)")
axs[1].legend()
axs[1].grid(True, alpha=0.3)
fig.suptitle("اثر نویز اندازه‌گیری و فیلتر Moving Average بر سیگنال کنترلی")
fig.tight_layout()
fig.savefig("bonus_noise_filter.png", dpi=140)
plt.close(fig)

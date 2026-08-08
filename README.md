# Digital Temperature Control System for an Industrial Incubator

This repository contains the simulation and conceptual hardware design for a digital temperature control system tailored for an industrial incubator. The project was developed as the final project for the **Digital Control Systems Lab** (University of Tehran).

## 📌 Project Overview
Before deploying the software onto real hardware (STM32 microcontroller), a complete digital controller was designed, simulated, and evaluated to precisely regulate the internal chamber temperature under various environmental disturbances.

### Key Features
* **Thermal System Modeling:** First-order thermal dynamics modeled with configurable parameters ($K$, $\tau$, $T_{\text{ambient}}$).
* **Digital PI / PID Controller:** Discrete-time controller implementation with anti-windup mechanism and PWM output saturation (0–100%).
* **Disturbance Rejection Simulation:** Evaluates response against chamber door openings (transient thermal loss), ambient temperature drops, and setpoint step changes.
* **Noise Filtering:** Digital low-pass / moving average filter integration for noisy sensor signal handling.
* **Hardware Conceptual Design:** Complete hardware architecture outline including industrial sensors, signal conditioning, STM32 peripheral configurations, and firmware pseudocode.

---

## 📐 System Model
The thermal dynamics of the incubator chamber are described by the differential equation:

$$\frac{dT}{dt} = -\frac{T - T_{\text{ambient}}}{\tau} + \frac{K}{\tau} u$$

Where:
* $T$: Chamber temperature ($^\circ\text{C}$)
* $T_{\text{ambient}}$: Ambient temperature ($^\circ\text{C}$)
* $u$: Heater power input ($u \in [0, 1]$, corresponding to $0 - 100\%$ PWM)
* $K$: System gain ($20 - 80\ ^\circ\text{C}$)
* $\tau$: Thermal time constant ($30 - 180\ \text{s}$)
* **Sampling Period ($T_s$):** $100\ \text{ms}$

---

## 🧪 Test Scenarios & Simulation
The system is simulated for **300 seconds** with $T_s = 100\text{ms}$ across the following scenarios:
1. **Step Response:** Initial temperature at $25^\circ\text{C}$, setpoint set to $40^\circ\text{C}$.
2. **Thermal Disturbance:** Chamber door opened for 10 seconds at $t = 120\text{s}$ (sudden increase in heat loss).
3. **Ambient Drop:** Ambient temperature drops from $25^\circ\text{C}$ to $15^\circ\text{C}$.
4. **Setpoint Change:** Setpoint steps from $40^\circ\text{C}$ to $60^\circ\text{C}$.

### Performance Metrics Evaluated
* Rise Time ($t_r$)
* Settling Time ($t_s$)
* Percentage Overshoot ($\%OS$)
* Steady-State Error ($e_{ss}$)
* System Stability & Anti-Windup Performance

---

## 🛠️ Tech Stack & Dependencies
* **Programming Languages:** Python / MATLAB / GNU Octave
* **Libraries:** `matplotlib`, `numpy`, `scipy` (if running Python implementation)
* **Target Hardware Architecture:** STM32 Microcontroller (ADC, TIM PWM)

---

## 📊 Results Summary
The Discrete PI/PID controller with Anti-Windup successfully stabilizes the incubator temperature, maintains zero steady-state error under ambient variations, and quickly recovers from thermal disturbances such as temporary door openings[cite: 1].

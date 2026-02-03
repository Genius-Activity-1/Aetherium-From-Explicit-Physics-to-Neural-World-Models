# Aetherium – World‑Only V5 (Phase‑Field World Model)

> **Experimental world‑model prototype inspired by phase‑field dynamics and the Aetherium framework**  
> Deterministic, agent‑compatible, and designed for emergent structures (zones, stress, events).

---

## ✨ Overview

**Aetherium World‑Only V5** is a lightweight **world simulation core** based on a 2D grid of zones governed by coupled scalar fields:

- **ρ (rho)** – density / resource / matter proxy  
- **θ (theta)** – phase‑like variable (order / alignment / orientation)  
- **Φg (phi_g)** – gravity‑like potential derived from density

From these fields, the model derives **observables** such as coherence, flow, and stress, and injects **field events** that generate long‑term structure without scripting.

The model is:
- deterministic (seeded RNG),
- quasi‑conservative (mass mostly conserved),
- suitable as a **world model** for simulations, games, or cognitive architectures.

---

## 🧠 Why this project exists

This repository explores a key question:

> *How can a world evolve coherently without explicit rules, factions, or narratives—only fields and local interactions?*

The answer here is a **phase‑field inspired world**, where:
- structure emerges from gradients,
- crises emerge from stress,
- stability is temporary and local.

This makes the model suitable as:
- a **foundation layer** for AI / PNJ simulations,
- a **sandbox world model** for research,
- a **procedural substrate** for games or cognitive agents.

---

## 🧩 Core Concepts

### Zones
The world is a rectangular grid of zones. Each zone holds:

| Variable | Meaning |
|--------|--------|
| `rho` | Density / resource / mass proxy |
| `theta` | Phase (wrapped in \[-π, π\]) |
| `phi_g` | Gravity‑like potential |
| `psi_coherence` | Local phase coherence |
| `omg_flow` | Phase gradient intensity |
| `phi_stress` | Stress indicator |
| `event_count` | Number of field events |

---

### Derived Observables

- **Coherence** → stability / order
- **Flow** → turbulence / instability
- **Stress** → likelihood of events

These observables are *not inputs* but consequences of the fields.

---

### Field Events

Events are **field‑level disturbances**, not scripted narrative actions:

- `PHASE_SHOCK` – sudden phase displacement
- `VORTEX_DEFECT` – topological‑like rotation
- `DENSITY_DROP` – mostly conservative redistribution of density

Events depend on local stress and flow, not on global triggers.

---

## ⚖️ Mass Philosophy (ρ)

The V5.4.x series uses **Option 1: quasi‑conservative mass**:

- density mostly moves by diffusion,
- small sources/sinks exist (recovery, gravity coupling),
- a **mass regulator** gently nudges total mass back toward its initial value.

This avoids:
- runaway collapse,
- runaway inflation,
- frozen dead zones.

The goal is **tension without explosion**.

---

## 🧪 Determinism & Tests

The model is **fully deterministic**:

- seeded RNG per world instance,
- same seed ⇒ same evolution.

Included tests verify:
- determinism,
- mass conservation (diffusion‑only),
- regulator ability to add *and* remove mass,
- bounds on all observables.

---

## 🚀 How to run

### Requirements

- Python ≥ 3.10
- NumPy

```bash
pip install numpy
```

### Run the demo

```bash
python aetherium_world_only_v5_phase_field.py
```

You should see:
- test validation,
- total mass drift,
- final zone state summary.

---

## 🔧 Tuning knobs (important)

Key parameters you may want to adjust:

```python
self.rho_diffuse_eps          # diffusion strength
self.rho_grav_feed            # gravity → density coupling
self.rho_evap                 # density evaporation
self.mass_regulator_k         # mass correction strength
self.mass_regulator_max_step  # correction cap per tick
```

These let you move between:
- calm continental worlds,
- turbulent collapse scenarios,
- slow‑burn instability.

---

## 🧭 Intended Extensions (not included)

This repo intentionally focuses on **world‑only** dynamics.

Planned or external extensions include:
- PNJ / agent layers reacting to stress & resources,
- faction emergence,
- memory or history layers,
- coupling to cognitive or decision agents.

---

## ⚠️ Status & Disclaimer

This is:
- **experimental research code**,
- not optimized,
- not a physics simulator,
- not production‑ready.

Its value is **conceptual and exploratory**.

---

## 📜 License

Open‑source (choose MIT / Apache‑2.0 / GPL according to your repo policy).

---

## ✍️ Author & Context

Created as part of the **Aetherium** research line exploring:

- world models,
- emergent structure,
- phase‑based cognition substrates,
- AI‑compatible environments.

The project is deliberately minimal, readable, and hackable.

---

> *A world should not be scripted. It should unfold.*


# 🌌 Aetherium

## From Explicit Physics to Neural World Models

**Aetherium** est un projet de recherche explorant la convergence entre les **simulations de champs de phase (Phase‑Field Physics)** et les **modèles de monde neuronaux (Neural World Models)**.

L’objectif est de démontrer comment une architecture d’IA peut :
- apprendre les **lois implicites d’un système complexe**,
- extraire des **structures latentes stables**,
- prédire des **dynamiques physiques et sociales émergentes**.

Le projet combine une **simulation déterministe explicite** et un **modèle neuronal prédictif** entraîné dessus.

---

# 🎯 Vision du projet

Les dynamiques complexes — qu’elles soient **physiques** (fluides, gravité, champs de phase) ou **sociales** (agents, factions, comportements émergents) — peuvent être décrites comme des **interactions de champs couplés**.

Aetherium explore deux approches complémentaires :

1. **Simulation déterministe** → règles physiques explicites
2. **World Model neuronal** → apprentissage implicite de ces règles via un espace latent

👉 Idée clé : *apprendre la physique plutôt que la coder entièrement à la main.*

---

# 🛠 Architecture du système

## 1️⃣ Moteur de simulation (NumPy)

Fichier : `aetherium_world_only_v5_4_full.py`

Simulation d’un environnement 2D où chaque zone possède :

### Variables d’état
- **Densité** ρ
- **Phase** θ
- **Potentiel gravitationnel** φ_g

### Dynamique
- Stabilisation des gradients via **tanh saturation**
- Diffusion + couplage local
- Évolution déterministe stable sur long terme

### Agents émergents
- PNJ avec profils psychologiques (Stoïque, Leader, etc.)
- Formation de **micro‑factions**
- Rétroaction comportement ↔ champ physique

### Régulateur de masse
- Correction douce vers `initial_mass`
- Évite la dérive numérique
- Assure stabilité sur simulations longues

---

## 2️⃣ Neural World Model (PyTorch)

Fichier : `aetherium_world_model_phy_head_v_2.py`

Architecture prédictive factorisée suivant la décomposition :

Φ – Ψ – Ω

### Modules

### Φ — Flux global (lent)
- LSTM
- Capture tendances macro
- Variables lentes / globales

### Ψ & Ω — Flux locaux (rapides)
- ConvLSTM
- Préservation structure spatiale
- Capture turbulences / dynamiques fines

### Physics Head
- Lecture interprétable
- Extraction directe de paramètres physiques :
  - cohérence fluide (C_t)
  - déphasage (Δφ_t)

### Contraintes spectrales
- FFT 2D
- `loss_spectral_coherence`
- Maintien des structures fréquentielles

---

# 🔬 Physics‑Informed Loss

Le modèle ne prédit pas uniquement des pixels :

Il est **guidé par des contraintes physiques explicites**.

### Fonctions de perte

### Reconstruction
- L1 / L2 sur les frames

### Contrainte de phase
- `loss_phase_threshold`
- Stabilise la variance vers σ ≈ 0.10

### Cohérence spectrale
- Analyse FFT
- Pénalisation du bruit chaotique

👉 Objectif : prédictions **structurellement plausibles**, pas seulement visuellement proches.

---

# 🚀 Installation

## Prérequis

```bash
pip install numpy torch matplotlib
```

---

# ▶️ Utilisation

## Lancer la simulation

```bash
python aetherium_world_only_v5_4_full.py
```

## Entraîner le world model

```bash
python aetherium_world_model_phy_head_v_2.py
```

---

# 📊 Cas d’usage

- Recherche sur les **world models physics‑informed**
- Simulation multi‑agents émergente
- Jeux vidéo / mondes persistants
- Modélisation socio‑physique
- Prototypage AGI incarnée légère

---

# 🧠 Concepts clés

- Phase‑Field Simulation
- Physics‑Informed Learning
- World Models
- Latent Factorization (Φ‑Ψ‑Ω)
- Emergent Behavior

---

# 🗺 Roadmap

- [ ] Optimisation GPU
- [ ] Port Rust haute performance
- [ ] Multi‑échelle 3D
- [ ] Entraînement auto‑supervisé
- [ ] Couplage agents cognitifs (AGI losange)

---

# 📄 Licence

Ce projet est distribué sous licence **Apache License 2.0**.

### Pourquoi Apache 2.0 ?

- ✅ Utilisation commerciale autorisée
- ✅ Modification et redistribution autorisées
- ✅ Protection explicite des brevets
- ✅ Compatible open‑source & industrie
- ✅ Adaptée aux projets DeepTech / IA / recherche appliquée

Cela permet :
- la réutilisation académique libre
- l’intégration en entreprise
- la contribution communautaire
- tout en protégeant la propriété intellectuelle des contributeurs

👉 Voir le fichier `LICENSE` pour le texte complet.

---

# 🤝 Citation

Si vous utilisez Aetherium dans vos travaux :

```
Morin, R. — Aetherium: From Explicit Physics to Neural World Models
```

---

# ⭐ Philosophie

> Construire d’abord un monde explicite.
> Puis apprendre à une IA à en découvrir les lois.
> Et enfin supprimer progressivement les règles codées.

Aetherium est une étape vers des **agents capables de comprendre leur environnement plutôt que de simplement le reproduire**.


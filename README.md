# SeedPro: Place Brachytherapy Seeds like Expert Clinicians via Hierarchical Reinforcement Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-In--Submission-blue)](https://github.com/Haitao-Lee/SeedPro)

**SeedPro** is an automatic and fine-grained preoperative planning framework for brachytherapy driven by hierarchical reinforcement learning agents.  
It efficiently produces expert-level treatment plans with reduced computational costs, ensuring high reproducibility and precise dose optimization.

---

## 📖 Introduction

Accurate preoperative planning plays a pivotal role in brachytherapy. However, manual planning is labor-intensive and often suffers from limited reproducibility. 

Although AI-driven approaches have recently emerged, many generate mechanically rigid seed arrangements that lack clinical practicality — such as excessive puncture trajectories or inefficient seed distributions.

**SeedPro** addresses these limitations by:  

- Emulating expert clinical decision-making through a multi-level state space generation algorithm  
- Prioritizing minimization of insertion needles ($n_p$) over total seeds ($n_s$), following real-world planning principles  
- Achieving state-of-the-art (SOTA) dosimetric performance with improved healthy tissue protection  

---

## 🛠️ Methodology

The SeedPro framework consists of three core components:

### 1️⃣ Multi-level State Space Construction  
A generation algorithm that mimics hierarchical clinical reasoning and enables global fine-grained optimization.

### 2️⃣ Hierarchical Reinforcement Learning-based Sampling  

- **High-level Agent ($\Omega_h$):**  
  Performs coarse-grained selection of trajectory combinations from the candidate state space.  

- **Low-level Agent ($\Omega_l$):**  
  Conducts fine-grained seed placement within selected trajectories until dosimetric objectives are satisfied.

### 3️⃣ Segmented Adaptive Objective Function  
A multi-objective reward mechanism that:

- Ensures sufficient tumoricidal dose coverage  
- Penalizes excessive dose to Organs at Risk (OARs)  
- Encourages reduced puncture trajectories and seed redundancy  

---

## 📊 Experimental Results

Experiments were conducted on an in-house clinical dataset of **17 patients** from Ruijin Hospital, including pancreatic, head & neck, and other tumor cases.

### 🔬 Performance Highlights

- **100% Success Rate (SR)** in achieving:
  - $V_{100} > 90\%$  
  - $D_{90} \ge 120$ Gy  

- Reduced number of:
  - Puncture needles ($n_p$)  
  - Implanted seeds ($n_s$)  

- Comparable or improved dosimetric quality versus SOTA methods

### 📈 Quantitative Comparison

| Method | Success Rate | $n_p$ (Needles) | $n_s$ (Seeds) |
|:------:|:------------:|:---------------:|:-------------:|
| **SeedPro (Ours)** | **100.00%** | **Lower** | **Fewer** |
| BrachyPlan | 100.00% | Higher | More |

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Haitao-Lee/SeedPro.git
cd SeedPro
```

### 2️⃣ Install Dependencies

```bash
pip install torch numpy scipy
```

### 3️⃣ Training / Inference

The optimization follows the coarse-to-fine two-stage hierarchical learning strategy described in Algorithm 1 of the paper.

```bash
python main.py --episodes 400 --goal_d90 120
```

---

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{li2026seedpro,
  title={SeedPro: Place Brachytherapy Seeds like Expert Clinicians via Hierarchical Reinforcement Agents},
  author={Li, Haitao and Liu, Jiaxuan and Huang, Wei and Wang, Zhongmin and Chen, Xiaojun},
  journal={School of Mechanical Engineering, Shanghai Jiao Tong University},
  year={2026},
  url={https://github.com/Haitao-Lee/SeedPro}
}
```

---

## 📫 Contact

For questions, please contact the corresponding author:

**Prof. Xiaojun Chen**  
Email: xiaojunchen@sjtu.edu.cn  

---

## 📄 License

This project is released under the MIT License.

# 🆓 Free Tier Deployment Guide for RSNA Knee AI

Deploy **RSNA Knee AI** using **Hugging Face Medical AI Spaces (16GB RAM Free)**, **Kaggle GPU Kernels**, and **Cloudflare Tunnels**.

---

## 1. Free Cloud GPU Training & Evaluation: Kaggle Kernels
Kaggle provides **30 hours per week of free NVIDIA T4/P100 GPUs**:
```bash
kaggle kernels push -p deploy/free/
```

---

## 2. Free Diagnostic Console: Hugging Face Spaces
1. Create a Space with Docker SDK at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Push your `Dockerfile`, `src/`, `package.json`, and `README_HF.md`.

---

## 3. Remote Diagnostic Demo: Cloudflare Tunnel
```powershell
# Windows
.\deploy\free\tunnel.ps1 -Port 3008

# Linux / macOS
./deploy/free/tunnel.sh 3008
```

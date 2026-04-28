# SoW + DoD — MLOps Churn Simulation

**Tanggal:** 2026-04-28
**Pemilik:** Gos Shafadonia (BNI ML Engineer)
**Status:** APPROVED v2 (binding)

---

## 1. Tujuan

Membangun simulasi MLOps end-to-end untuk:

1. **Pemahaman pribadi** user tentang cara kerja MLOps secara hands-on.
2. **Demo visual** untuk PM yang membutuhkan visibility ke ML operations (menjawab pain point: *"Saya tidak bisa memonitor perkembangan sama sekali"*).

## 2. Use Case

| Aspek | Keputusan |
|---|---|
| Domain | Banking — Customer Churn Prediction |
| Dataset | Churn Modelling (Kaggle), 10.000 baris, 14 kolom |
| Tipe ML | Klasifikasi biner (`Exited`: 1/0) |
| Cerita bisnis | "BNI ingin prediksi nasabah mana yang akan tutup rekening, agar tim retention bisa proaktif kontak mereka." |

## 3. Tech Stack (Versi Tervalidasi via Context7 — 2026-04-28)

| Komponen | Versi | Status |
|---|---|---|
| Python | 3.11 | IN |
| **uv** (Astral) | terbaru | IN — package & project manager (pyproject.toml + uv.lock) |
| **ruff** | terbaru | IN — linter + formatter (replace black/isort/flake8). Config di pyproject.toml |
| MLflow | v3.1.4 (latest) | IN |
| Gradio | v6.0.1 (latest) | IN |
| scikit-learn | terbaru kompatibel | IN |
| XGBoost | terbaru kompatibel | IN |
| pandas, numpy, matplotlib | terbaru kompatibel | IN |
| scipy | terbaru kompatibel — `scipy.stats.ks_2samp` untuk drift | IN |
| pytest | untuk testing | IN |
| MLflow backend | SQLite lokal | IN |
| Storage artifacts | Local filesystem | IN |

## 4. In Scope (20 item, semua wajib diimplementasi)

### Tier A — Core MLOps
1. Data ingestion & preprocessing pipeline (reproducible)
2. Training pipeline dengan 3 algoritma: Logistic Regression, Random Forest, XGBoost
3. Hyperparameter variations manual: minimum 3 run per algoritma (total ≥ 9 run)
4. MLflow Tracking server lokal
5. MLflow Model Registry dengan **aliases** (`staging`, `production`, `archived`) — mengikuti best practice MLflow 3.x; stages deprecated sejak 2.9.0
6. Skrip alias management via `MlflowClient.set_registered_model_alias()` (promote/demote winning model)

### Tier B — Production Realism
7. Gradio app untuk inference (input fitur nasabah → prediksi churn + probability)
8. Skrip simulasi inference traffic (men-generate prediction logs)
9. Production monitoring dilihat di **MLflow UI native** (1 long-running run bernama `production-monitoring` dengan time-series metrics: prediction count, latency p50/p95, drift score) — TIDAK ada tab monitoring di Gradio
10. Skrip simulasi data drift (input dengan distribusi shifted)
11. Drift detection via `scipy.stats.ks_2samp` (Kolmogorov-Smirnov test) — pakai library existing, no custom math. Logged sebagai metric `drift_score` ke MLflow, visual di MLflow UI native (chart over step)
12. Skrip retraining trigger (manual command)

### Tier C — Advanced
13. A/B testing simulasi: 2 versi model, traffic split 50/50, comparison view
14. Alerting via MLflow tag `alert=true` pada run yang drift score-nya melewati threshold (filter via `tags.alert = 'true'` di MLflow UI). Tidak ada custom alert dashboard
15. Audit trail menggunakan **MLflow native registry history** per model version (timestamp + transition history). **Honest framing untuk PM**: lokal demo = single OS user, multi-user audit memerlukan MLflow auth setup (Out of Scope)

### Supporting
16. EDA notebook (1 file, untuk explore awal — bukan bagian pipeline)
17. README dengan setup + demo flow
18. `pyproject.toml` + `uv.lock` (di-manage via uv) — TIDAK pakai requirements.txt
19. Test cases dengan pytest:
    - Test preprocessing function (input/output schema)
    - Test training pipeline (run completes, model artifact created)
    - Test inference function (valid input → valid output)
    - Test drift detection function (known distributions → expected score)
20. Design document & implementation plan di `docs/superpowers/specs/`

## 5. Out of Scope (BINDING — TIDAK akan diimplementasi)

| Item | Alasan |
|---|---|
| Docker / containerization | Lokal eksekusi cukup untuk demo PM |
| Kubernetes / orchestration | Overkill untuk simulasi |
| Cloud deployment (AWS/GCP/Azure) | Lokal saja |
| CI/CD pipeline | Bukan fokus belajar MLOps core |
| Authentication / RBAC | Demo lokal, tidak butuh auth |
| Real-time streaming (Kafka, dll) | Batch + simulasi traffic cukup |
| Database production (Postgres/MySQL) | SQLite cukup |
| Feature store (Feast, Tecton) | Adds tooling tanpa nilai untuk simulasi |
| Optuna / automated hyperparameter tuning | Manual variations cukup |
| Custom MLflow plugins / custom flavor | Pakai built-in flavors |
| Custom model explainability (SHAP, LIME) | Bukan fokus MLOps |
| Frontend kustom selain Gradio | Gradio cukup |
| API serving via FastAPI/Flask | MLflow built-in serving + Gradio cukup |
| Multi-user collaboration features | Single-user demo |
| Production-grade error handling & retry logic | Simulasi, bukan produksi |

## 6. Tool Boundary & UI Constraint

### Tool boundary (no overlap)
- **MLflow native UI** (port 5000) = surface OBSERVASI: experiment view, run comparison, model registry, monitoring, drift, audit trail
- **Gradio app** (port 7860) = surface AKSI: trigger training (Training Lab), live inference, A/B test toggle
- Tidak boleh duplikasi capability (mis. tidak ada tab "Monitoring" di Gradio karena MLflow sudah punya)

### Gradio constraint
Gradio harus **kohesif dalam SATU app** dengan `gr.Tabs` (3 tab saja: Training Lab / Inference / A/B Test). Tidak boleh:
- Multiple Gradio app pada port berbeda
- UI yang memaksa user navigasi keluar untuk lookup info
- Modal stack dalam atau hidden expansion yang menyembunyikan konteks

### Process topology
- 2 proses lokal (MLflow tracking server + Gradio app), dilauncher 1 command
- MLflow UI = pintu observasi; Gradio = pintu aksi

## 7. Deliverables

1. Source code di `/Users/ghawsshafadonia/Documents/Pekerjaan/BNI/test/`
2. Design doc: `docs/superpowers/specs/2026-04-28-mlops-churn-simulation-design.md`
3. Implementation plan: `docs/superpowers/specs/2026-04-28-mlops-churn-simulation-plan.md`
4. SoW + DoD: dokumen ini (`2026-04-28-mlops-churn-simulation-sow-dod.md`)
5. README di `test/`
6. Sample artefak hasil run (model versions, run history, prediction logs)

## 8. Definition of Done

### 8.1 Per-Komponen DoD (semua wajib ✓)
- [ ] Komponen berjalan tanpa error di lingkungan lokal user
- [ ] Bisa di-run ulang dengan satu command tanpa manual prep
- [ ] Mengikuti API terbaru MLflow 3.x / Gradio 6.x — divalidasi via fetch docs sebelum koding
- [ ] Test case relevan pass (untuk komponen yang ada di list test #19)
- [ ] Tidak ada hardcoded secret atau path absolute
- [ ] README di-update dengan cara pakai komponen tersebut
- [ ] Komentar kode hanya untuk hal non-obvious

### 8.2 Project-Level DoD
- [ ] User bisa demo end-to-end flow dalam ≤ 15 menit dengan ≤ 5 command
- [ ] Setiap pain point PM ("tidak bisa monitor perkembangan") punya komponen yang menjawabnya
- [ ] Semua 20 item di "In Scope" punya bukti implementasi (file/test/screenshot)
- [ ] Spec & implementation plan di-review & approved sebelum koding dimulai
- [ ] Setiap library yang digunakan sudah dicek docs terbarunya via Context7/WebFetch
- [ ] `uv run ruff check .` PASS (no lint errors)
- [ ] `uv run ruff format --check .` PASS (formatted consistently)
- [ ] `uv run pytest` PASS (semua 13 test green)

## 9. Aturan Operasional

1. **Decisions are binding.** Kalau muncul kebutuhan di luar SoW selama implementasi, itu adalah scope-change conversation eksplisit, bukan sneak-add.
2. **Validate before code.** Setiap library API divalidasi ke docs terbaru sebelum ditulis ke kode.
3. **No hedging.** Bahasa di plan dan kode harus tegas — tidak ada "kalau perlu", "tergantung", "mungkin".

## 10. Estimasi Effort

5–7 sesi kerja, di-breakdown lebih detail di implementation plan (next phase setelah design approved).

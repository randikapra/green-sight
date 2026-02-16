# 🌿 GreenSight v2.0

**Deep Learning-Based Urban Green Space Monitoring Using Sentinel-2 Satellite Imagery**

A multi-city study of urban green space dynamics in Sri Lanka using transfer learning on multi-spectral satellite imagery. Features climate-adaptive thresholds, confidence-based validation, and comprehensive quality assurance.

---

## 🎯 Key Features (v2.0)

✅ **Climate-Adaptive Classification** — Separate NDVI thresholds for Wet/Intermediate/Dry/Arid zones  
✅ **Confidence Filtering** — 70% minimum confidence for predictions, 75% for change detection  
✅ **Automatic Validation** — Flag/reject unrealistic changes (±30%/±100% thresholds)  
✅ **Visual Quality Assessment** — RGB + NDVI + Predictions side-by-side validation grids  
✅ **Multi-City Support** — Colombo, Kandy, Hambantota, Jaffna across all climate zones

---

## 🚀 Quick Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt
earthengine authenticate

# 2. Run full pipeline
python scripts/collect_data.py       # Download Sentinel-2 from GEE
python scripts/preprocess.py         # Generate labeled patches
python scripts/train.py              # Train all models
python scripts/analyze.py            # Temporal change detection

# 3. VALIDATE RESULTS (CRITICAL!)
python scripts/validate_results.py   # Generate visual validation grids
```

**Review:** Check `results/*/temporal_analysis/validation_grid.png` to verify predictions

---

## 📊 Research Configuration

### Study Areas (Multi-Climate)
- **Colombo** (Wet Zone): 79.82-79.92°E, 6.85-6.98°N — Highly urbanized capital
- **Kandy** (Intermediate): 80.59-80.69°E, 7.26-7.33°N — Hill capital, cultural center
- **Hambantota** (Dry Zone): 81.08-81.18°E, 6.10-6.20°N — Rapid infrastructure development
- **Jaffna** (Arid Zone): 79.98-80.08°E, 9.63-9.73°N — Post-conflict reconstruction

### Time Periods
- **2019:** Jan-Mar (baseline)
- **2024:** Jan-Mar (comparison) — 5-year change

### Models Evaluated
| Model | Type | Params | Notes |
|-------|------|--------|-------|
| ResNet-50 | CNN | 25.6M | Strong baseline |
| EfficientNet-B0 | CNN | 5.3M | Lightweight |
| ViT-Small | Transformer | 22.1M | Patch attention |
| Swin-Tiny | Transformer | 28.3M | Shifted windows |
| ConvNeXt-Tiny | Modern CNN | 28.6M | ViT-competitive |

---

## 🔬 Methodology Highlights

### Climate-Adaptive NDVI Thresholds
```
Zone          Green Space Threshold    Rationale
──────────────────────────────────────────────────
Wet           NDVI ≥ 0.45             Abundant vegetation baseline
Intermediate  NDVI ≥ 0.40             Moderate vegetation
Dry           NDVI ≥ 0.35             Sparse but active vegetation
Arid          NDVI ≥ 0.30             Minimal vegetation present
```

### Quality Assurance Pipeline
1. **NDVI bounds:** 0.95 max (excludes clouds/artifacts)
2. **Otsu validation:** Thresholds must be in 0.20-0.65 range
3. **Confidence filtering:** 70% min for predictions
4. **Change validation:** Flag >±30%, reject >±100%

### Output Validation
Each city gets:
- `temporal_results.json` — Statistics + validation status
- `prediction_maps.npz` — Predictions + confidence maps
- `validation_grid.png` — Visual QA (RGB, NDVI, predictions, confidence)
- `validation_change_map.png` — Spatial change patterns

---

## 📁 Project Structure

```
green-sight/
├── configs/default.yaml          # All settings (climate zones, models, thresholds)
├── scripts/
│   ├── collect_data.py           # GEE download
│   ├── preprocess.py             # Patch generation + Otsu labeling
│   ├── train.py                  # Multi-model training
│   ├── analyze.py                # Temporal analysis + validation
│   └── validate_results.py       # Visual QA generation (NEW!)
├── src/
│   ├── data/                     # Collection + preprocessing modules
│   ├── models/                   # Model factory + trainer
│   ├── analysis/                 # Temporal analysis + visualization
│   └── utils/                    # Config + logging
├── data/raw/                     # Sentinel-2 GeoTIFFs
├── data/processed/               # Labeled patches
├── results/                      # Model outputs + figures
└── CHANGELOG.md                  # Complete list of fixes (NEW!)
```

---

## ⚠️ Critical Changes from v1.0

See `CHANGELOG.md` for full details. Major fixes:

1. ✅ **NDVI upper bound** added (0.95) — eliminates cloud false positives
2. ✅ **Climate-adaptive thresholds** — respects vegetation zones
3. ✅ **Confidence filtering** — only robust predictions used
4. ✅ **Change validation** — auto-flags/rejects implausible results
5. ✅ **Visual validation** — new script for manual QA

**Migration:** Rerun preprocessing and temporal analysis with new thresholds.

---

## 🎓 For Researchers

### Reproducing Results
```bash
# Use exact config
python scripts/train.py --config configs/default.yaml

# Single city analysis
python scripts/analyze.py --city colombo
python scripts/validate_results.py --city colombo
```

### Customization
Edit `configs/default.yaml`:
- Change study area bounds
- Adjust confidence thresholds
- Enable/disable models
- Modify time periods

### Adding New Cities
1. Add to `study_areas` in config with appropriate climate zone
2. Set climate-adaptive green space threshold
3. Run full pipeline

---

## 📊 Expected Results (Post-Fix)

Realistic change ranges (5-year period):
- **Wet Zone:** ±5-15% (dense baseline, slow change)
- **Dry/Arid:** ±10-25% (sparse baseline, higher variability)
- **Rapid Development Areas:** Up to ±30% (infrastructure projects)

**Red Flags:**
- Change >±30% → Manual verification required
- Change >±100% → Likely data/model error
- Very low confidence (<50% coverage) → Poor data quality

---

## 🐛 Troubleshooting

**Unrealistic changes detected:**
- Check `validation_grid.png` for cloud contamination
- Verify Otsu thresholds in preprocessing logs
- Increase `min_prediction_confidence` in config

**Low confidence coverage:**
- May indicate poor model performance
- Try ensemble approach (average multiple models)
- Check for seasonal mismatch or cloud cover

**Class imbalance:**
- Review per-class metrics in `test_results.json`
- Consider focal loss or class weights
- Verify training data distribution

---

## 📧 Support

For questions related to:
- **GEE authentication:** https://developers.google.com/earth-engine/guides/auth
- **Model selection:** Review ablation study in `scripts/ablation.py`
- **Configuration:** See `configs/default.yaml` comments
- **Validation:** See `CHANGELOG.md` for methodology

---

## 📝 License

Research code for academic use. Please cite if used in publications.

---

**Version:** 2.0  
**Last Updated:** 2025  
**Status:** Production-ready for A-tier conference submission


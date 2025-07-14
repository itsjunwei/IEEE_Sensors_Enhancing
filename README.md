# Enhancing 3D Sound Event Localization and Detection with Distance Estimation Using Reverberation and Spatial Coherence Features

This repository contains code needed for feature extraction for the paper "Enhancing 3-D Sound Event Localization and Detection with Distance Estimation Using Reverberation and Spatial Coherence Features" available in [this link](https://ieeexplore.ieee.org/abstract/document/11062503).

Official implementation of the Coherence and Direct-Path Dominance (CDPD) Feature proposed for 3-D Sound Event Localization and Detection with Distance Estimation.

If you have any questions, please feel free to reach out to us at: `junwei004@e.ntu.edu.sg`

## Repository Structure

```text
.
├── README.md
├── extract_salsa_feats.py      # Core extraction functions
```

Within `extract_salsa_feats.py`, we provide code to extract all SALSA variants (SALSA, SALSA-Lite, SALSA-D, SALSA-DLite). 

## Requirements

This code has been tested for Python 3.11.13, NumPy v1.26.4, Librosa v0.11.0.

## Citation

Please consider citing our papers if you find this code useful for your research, thank you!

```
@article{yeow2025enhancing,
  title={Enhancing 3D Sound Event Localization and Detection with Distance Estimation Using Reverberation and Spatial Coherence Features},
  author={Yeow, Jun-Wei and Tan, Ee-Leng and Bai, Jisheng and Peksi, Santi and Gan, Woon-Seng},
  journal={IEEE Sensors Journal},
  year={2025},
  publisher={IEEE}
}
```
# Bearing Fault Early Detection with Custom DSP Features

## Overview
This repository presents a comprehensive approach to early detection of bearing faults using custom digital signal processing (DSP) features derived from vibration data. The methodology leverages advanced signal analysis techniques to identify fault precursors in rotating machinery, enabling predictive maintenance. The project includes two primary Jupyter notebooks that implement and evaluate these methods on benchmark datasets.

Key objectives:
- Extract and analyze time-domain and frequency-domain features for fault detection.
- Compare traditional and adaptive transform-based approaches for improved sensitivity and accuracy.
- Provide reproducible code for researchers and practitioners in mechanical engineering and machine learning.


## Démo Vidéo – Détection Précoce (130 Mo)

### Détection ultra-précoce du défaut – Vidéo complète avec son

<video width="100%" controls autoplay loop muted>
  <source src="https://github.com/DrStef/Bearing-Fault-Early-Detection-with-Custom-DSP-Features_v01/raw/main/bearing_videos/EM_bearing1_f3_surframes_v04_lowdef.mp4" type="video/mp4">
  Votre navigateur ne supporte pas la vidéo.
</video>

**Le défaut devient visible à l’œil nu dès la super-frame ~460** (vers 23 minutes dans la vidéo).  
Kalman confirme l’alarme persistante à la super-frame ~465–470.

→ Aucun faux positif. Aucune configuration complexe. Juste un modèle physique + un Kalman.



## Methods

The analysis is divided into two notebooks, each focusing on distinct DSP techniques.

### Notebook I: Time Series Methods

This notebook explores classical time series analysis methods for feature extraction from bearing vibration signals. It includes:
- Computation of statistical features such as root mean square (RMS), kurtosis, skewness, and crest factor.
- Time-domain signal processing, including envelope detection and trend analysis.
- Application of these features to machine learning models like LSTM and Prophet for binary classification of healthy vs. faulty states.
- Evaluation on datasets like the PRONOSTIA (NASA) bearing dataset, with Kalman filtering for denoising.

The notebook demonstrates how these methods can detect early fault signatures with minimal computational overhead.

### Notebook II: CWT and aT-CWT

This notebook implements wavelet-based transforms for enhanced fault detection in non-stationary signals. Key components include:
- Continuous Wavelet Transform (CWT) for multi-resolution analysis of vibration spectra, using complex Morlet wavelets to capture transient fault impulses.
- Adaptive Time-Coherent Continuous Wavelet Transform (aT-CWT), a custom extension that adjusts wavelet parameters dynamically based on signal characteristics to improve resolution in varying speed conditions.
- Feature selection from scalograms, followed by integration with classifiers like CNN autoencoders for anomaly detection.
- Comparative analysis against baseline methods to highlight improvements in early detection rates.

These techniques are particularly effective for identifying subtle frequency modulations indicative of incipient faults.

## Results

Preliminary results indicate that aT-CWT achieves a higher accuracy in early fault detection compared to standard time series methods, with a false positive rate below 5% on test datasets. Detailed metrics, including precision, recall, and ROC curves, are visualized in the notebooks. 

Further validation on real-world industrial data is recommended.



<video width="900" controls>
  <source src="https://github.com/DrStef/Bearing-Fault-Early-Detection-with-Custom-DSP-Features/edit/main/bearing_defaut_detection.mp4" type="video/mp4">
</video>



## Notebooks



## Installation and Usage

To run the notebooks:
1. Clone the repository: `git clone https://github.com/DrStef/Bearing-Fault-Early-Detection-with-Custom-DSP-Features.git`
2. Install dependencies: `pip install -r requirements.txt` (includes numpy, scipy, pywt, scikit-learn, tensorflow, and matplotlib).
3. Launch Jupyter: `jupyter notebook` and open the respective notebooks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work is inspired by datasets from the PRONOSTIA (NASA) Bearing Data Center and builds on open-source DSP libraries.



## Dataset: NASA Bearing Dataset (Focus on Set 2)
The NASA Prognostics Data Repository provides run-to-failure vibration data from Rexnord ZA-2115 double-row bearings under constant conditions (2000 RPM, 6000 lbs radial load, force-lubricated).

- **Set 2 Details** (Selected for Simplicity):
  - Recording Duration: February 12, 2004 10:32:39 to February 19, 2004 06:22:39.
  - No. of Files: 984 (1-second excerpts every 10 minutes, ASCII format).
  - No. of Channels: 4 (one accelerometer per bearing: Bearing 1 Ch1, Bearing 2 Ch2, Bearing 3 Ch3, Bearing 4 Ch4).
  - Sampling Rate: 20 kHz.
  - Fault: Outer race failure in Bearing 1 (progressive degradation over ~16 minutes total runtime).
  - Structure: Files named by timestamp (e.g., "2004.02.12.10.32.39"); early files healthy, later show increasing impulses.

Data is loaded as a NumPy array [984 files, 4 channels, 20,000 samples] for analysis. Theoretical fault frequencies (at 2000 RPM):
| Frequency | Value (Hz) | Interpretation |
|-----------|------------|----------------|
| FTF (Cage) | ≈ 0.40 | Cage rotation (multiples ~50-60 Hz noise). |
| BPFO (Outer Race) | ≈ 236.4 | Dominant for Set 2 fault; sidebands ±33 Hz. |
| BPFI (Inner Race) | ≈ 803.6 | Secondary; harmonics ~1607 Hz. |
| BSF (Ball Spin) | ≈ 141.2 | Roller faults; subharmonics ~70 Hz. |

## Strategy
1. **Auditory Validation**: Concatenate 1s clips per channel into full audio files (WAV at 20 kHz or downsampled to 8 kHz for listenability). Listen for perceptual changes (e.g., "rasps" after 8 min, "metal screams" near end) to ground-truth "human ear" anomaly timestamps.
2. **Preprocessing**: High-pass filter (>100 Hz) to remove 60 Hz noise; Kalman smoothing for state estimation (constant-velocity model).
3. **Feature Extraction**: 
   - CWT (Morlet complex) for time-frequency scalograms.
   - Synchrosqueezing to sharpen harmonic ridges and denoise (reassign energy to true instantaneous frequencies).
   - Custom transforms: Magnitude-phase correlations across scales for anomaly scoring (e.g., high corr >0.8 indicates fault coupling).
4. **Detection & ML**: Threshold energy/correlation maps for early alerts; feed to CNN autoencoder (reconstruction error for unsupervised) or supervised CNN (classification of degradation stages: healthy, mild, strong, failure).
5. **Evaluation**: Compare features with auditory timestamps; ROC AUC for detection; extend to drilling telemetry (vibration harmonics as proxies for bit wear/stick-slip).

This pipeline prioritizes early detection (e.g., BPFO at 236 Hz emerging in mild phase), bridging human intuition with automated DSP/ML.

## Theoretical Fault Frequencies for Rexnord ZA-2115 Bearing (at 2000 RPM)

Based on standard formulas (from the Stack Exchange discussion) and bearing parameters: n=16 (rollers per row), D=2.815 in (pitch diameter), d=0.331 in (roller diameter), φ=15.171° (contact angle). Shaft speed R = 2000/60 = 33.33 rev/s.

### Table of Fault Frequencies

| Frequency       | Formula                          | Theoretical Value (Hz) | Interpretation                                                                 |
|-----------------|----------------------------------|------------------------|-------------------------------------------------------------------------------|
| FTF (Cage)     | (R/2) (1 - (d/D) cos φ)         | ≈ 0.40                | Fundamental train frequency (cage rotation). Close to your 50-60 Hz peaks? (multiples possible). |
| BPFO (Outer Race) | (n R/2) (1 - (d/D) cos φ) | ≈ 236.4               | Ball pass outer – outer race fault, with sidebands at ±33.33 Hz. |
| BPFI (Inner Race) | (n R/2) (1 + (d/D) cos φ) | ≈ 803.6               | Ball pass inner – inner race fault, harmonics ~2x = 1607 Hz. |
| BSF (Ball Spin) | (D R / 2d) [1 - ((d/D) cos φ)^2] | ≈ 141.2               | Ball spin – roller element fault, subharmonics ~70 Hz.         |

These values confirm your peaks: 986 Hz ~4x BSF (564 Hz) or 2x BPFI sideband; 50-60 Hz ~ multiples FTF or BSF/2. For 9000 Hz, it's outside faults (shaft resonance or noise).






## Installation & Usage
1. Clone repo: `git clone https://github.com/DrStef/Bearing-Fault-Detection.git`
2. Install deps: `conda install -c conda-forge pywt numpy scipy matplotlib pandas -y` (or pip equivalent).
3. Run notebook: `jupyter notebook main_analysis.ipynb`
   - Loads data from `archiveNASA/2nd_test`.
   - Generates plots, audio, and features.
   

## Extensions to Drilling Telemetry
The methods scale to oil & gas telemetry: Custom CWT/synchrosqueezing on multi-sensor vibrations (torque/pressure fusion) for real-time fault prediction (e.g., harmonic anomalies indicating bit fatigue 30s ahead). Aligns with streaming ML pipelines (Kafka/Flink) for 10k ft deep operations.

## License
MIT License – feel free to fork and collaborate!

*Contact: DrStef on GitHub | Open to DSP/ML discussions for industrial prognostics.*





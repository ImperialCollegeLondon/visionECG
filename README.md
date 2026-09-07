# visionECG: Reconstructing synthetic hearts from ECG using flow matching

## Overview

**visionECG** is a conditional latent flow matching model that generates patient-specific left ventricular geometry and motion throughout the cardiac cycle from 12-lead ECG waveforms and demographic information. Its structured 4D reconstructions enable flexible measurement of cardiac structure and function, supporting assessment of structural abnormalities and cardiovascular risk stratification.

<p align="center">
  <img src="assets/example_case_video.gif" width="800" alt="ECG waveforms and generated left ventricular mesh sequences"/>
</p>
<p align="center">
  <em>ECG waveforms and corresponding patient-specific cardiac meshes.</em>
</p>

<p align="center">
  <img src="assets/measurement_video.gif" width="800" alt="Dynamic measurements of generated cardiac meshes"/>
</p>
<p align="center">
  <em>Dynamic measurements of cardiac structure and function.</em>
</p>

## Features

- ECG-to-4D left ventricular mesh generation
- Conditional flow matching with anatomical priors
- Patient-specific cardiac geometry and motion
- Global and regional cardiac measurements
- Interpretable structural heart disease assessment
- Cardiovascular risk stratification

## Reference
**[medRxiv 2026 preprint](https://www.medrxiv.org/content/10.64898/2026.09.01.26360987v1)**  

<!-- ## Installation

```bash
git clone https://github.com/JZCambridge/visionECG.git
cd visionECG
conda env create -f environment.yml
conda activate visionecg
``` -->

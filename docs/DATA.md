# Data access and provenance

## Source dataset

The study uses **Leipzig Heart Center ECG-Database: Arrhythmias in Children and Patients with Congenital Heart Disease, version 1.0.0**, published on PhysioNet in 2025.

- Dataset DOI: `10.13026/7a4j-vn37`
- PhysioNet project: `leipzig-heart-center-ecg/1.0.0`
- Upstream file license reported by PhysioNet: **Open Data Commons Attribution License v1.0**
- The dataset includes 12-lead ECG, intracardiac electrograms, and rhythm annotations including AVNRT and AVRT.

Raw waveform files are not included in this GitHub repository. Download them directly from PhysioNet and comply with the upstream terms.

Example download command from the PhysioNet dataset page:

```bash
wget -r -N -c -np https://physionet.org/files/leipzig-heart-center-ecg/1.0.0/
```

The supplied notebook defaults to:

```python
cfg.data_dir = './leipzig-heart-centerwith-congenital-heart-disease-1.0.0'
```

Either rename/place your local dataset accordingly or edit `cfg.data_dir` to the downloaded PhysioNet folder.

## Dataset citation

Klehs S, Franke D, Alhamad B, Gebauer R, Teich L, Teich T, Paech C. *Leipzig Heart Center ECG-Database: Arrhythmias in Children and Patients with Congenital Heart Disease*. Version 1.0.0. PhysioNet. 2025. DOI: 10.13026/7a4j-vn37.

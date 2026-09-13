# AI model card: MechSSL-Window Transformer

## Intended use
Internal research evaluation of AVNRT-vs-AVRT mechanism classification using ECG and intracardiac electrograms.

## Not intended use
Not for autonomous diagnosis, treatment selection, or deployment without external validation and prospective clinical evaluation.

## Input data
Windowed beat-token representations from 12-lead ECG, intracardiac electrograms, channel masks, mechanism tokens, and handcrafted summaries.

## Output
Probability of AVRT versus AVNRT at window and record level.

## Training and evaluation data
Internal patient-wise grouped cross-validation on the available supervised AVNRT/AVRT records.

## Limitations
Small sample size, possible dataset-specific acquisition patterns, missing channels, and no external validation.

## Ethical and safety notes
Predictions should be interpreted as research signals only. Human electrophysiologist review remains required.

## Deployment warning
Do not deploy clinically unless independently validated, monitored, calibrated, and approved through appropriate regulatory and institutional pathways.

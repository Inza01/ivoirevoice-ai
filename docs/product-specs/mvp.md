# IvoireVoice AI MVP contract

## Product intent

IvoireVoice AI is a local research demonstrator for comparing French and
Dioula automatic speech recognition. It makes a bounded Whisper Tiny
adaptation inspectable without presenting it as an industrial model.

## Supported journeys

1. Upload or record a supported audio file in the local Gradio interface.
2. Compare Whisper Tiny, Whisper Small and the local Tiny Dioula pilot.
3. Optionally provide a reference to calculate WER and CER.
4. Inspect latency, RTF and edit counts.
5. Export privacy-safe comparison results.
6. View two explicitly separate aggregate benchmark experiments.

## Acceptance criteria

- model failures are isolated;
- the two public baselines remain usable when the private checkpoint is absent;
- model execution is sequential and releases resources;
- WER, CER and RTF identify lower values as better;
- the 600-audio validation experiment is never merged with the historical
  150-audio pilot;
- public exports contain no personal local paths;
- automated tests do not require corpus access or model downloads.

## ML evidence

Only Whisper Tiny was adapted. The pilot used 2,250 train audios and 600
validation audios with speaker-disjoint splits. Reported values are validation
evidence for a pilot, not final-holdout or production performance.

## Non-goals

- public checkpoint or corpus distribution;
- final-holdout evaluation;
- full-corpus training;
- authenticated production deployment;
- claims about Wav2Vec2, XLS-R or MMS implementation;
- universal robustness across accents, noise and recording conditions.

## Human approval boundaries

Human approval is required before:

- using final-holdout data;
- launching full training;
- publishing private artifacts or a checkpoint;
- changing corpus licensing or consent assumptions;
- enabling public Gradio sharing;
- pushing or deploying when the current task does not explicitly authorize it.

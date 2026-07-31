# Local data contract

The real IvoireVoice AI corpus is not included in this repository. Audio,
transcriptions, participant metadata and full manifests remain local to
protect speakers and because redistribution rights and consent have not been
confirmed.

## Expected layout

Authorized users may prepare their own corpus outside the repository:

```text
voices_data/
├── group_or_speaker_001/
│   ├── clips.json
│   └── wav/
│       └── sample.wav
└── group_or_speaker_002/
    ├── clips.json
    └── wav/
        └── sample.wav
```

Set its location with `IVOIREVOICE_DIOULA_DATA_DIR`. Generated manifests and
private experiment outputs must be placed under `IVOIREVOICE_ARTIFACTS_DIR`.

## Manifest

The minimal public schema is illustrated by
[`examples/sample_manifest.csv`](../examples/sample_manifest.csv). The local
training manifest additionally preserves raw text, Unicode-normalized text,
the no-tone variant, `target_text_mvp`, provenance and governance fields.

Recommended audio properties:

- mono PCM WAV;
- 16 kHz sampling rate;
- finite, non-empty samples;
- short utterances compatible with the configured Whisper duration limit.

## Split policy

Train, validation and test must be separated by speaker. A `speaker_id` or an
identical `audio_sha256` must never occur in more than one split. The pilot
training workflow uses train and validation only; the historical test pilot
and final holdout remain excluded.

## Preparing an authorized corpus

1. place the source corpus outside Git;
2. configure the data and artifact environment variables;
3. run `make manifest-dioula`, `make audit-dioula` and
   `make curate-dioula`;
4. inspect quarantine and split reports;
5. freeze a new local dataset version only after human approval;
6. keep full manifests, references and predictions outside Git.

The fixture and example files in this repository are synthetic. They are not
copied from the real corpus and cannot be used to reconstruct it.

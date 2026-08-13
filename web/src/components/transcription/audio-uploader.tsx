"use client";

import { useRef, useState } from "react";

import { PrimaryButton, SecondaryButton } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";

export type AudioUploaderLabels = {
  chooseFile: string;
  description: string;
  selectedFile: string;
  formats: string;
  microphoneDescription: string;
  microphoneTitle: string;
  title: string;
  microphoneUnavailable: string;
  invalidFile: string;
  fileTooLarge: string;
  or: string;
};

type AudioUploaderProps = {
  labels: AudioUploaderLabels;
};

export function AudioUploader({ labels }: AudioUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  return (
    <section className="audio-uploader" aria-labelledby="audio-upload-title">
      <div className="upload-zone">
        <span className="upload-icon">
          <Icon name="upload" />
        </span>
        <h2 id="audio-upload-title">{labels.title}</h2>
        <p>{fileName ? `${labels.selectedFile} : ${fileName}` : labels.description}</p>
        <span className="upload-formats">{labels.formats}</span>
        <input
          accept=".wav,.mp3,.m4a,.ogg,audio/wav,audio/mpeg,audio/mp4,audio/ogg"
          aria-describedby={error ? "audio-file-error" : undefined}
          aria-label={labels.chooseFile}
          className="visually-hidden"
          id="audio-file"
          onChange={(event) => {
            const file = event.currentTarget.files?.[0];
            setFileName(null);
            setError(null);
            if (!file) return;
            if (file.size > 25 * 1024 * 1024) {
              setError(labels.fileTooLarge);
              event.currentTarget.value = "";
              return;
            }
            if (!file.type.startsWith("audio/")) {
              setError(labels.invalidFile);
              event.currentTarget.value = "";
              return;
            }
            setFileName(file.name);
          }}
          ref={inputRef}
          type="file"
        />
        {error ? (
          <p className="field-error" id="audio-file-error" role="alert">
            {error}
          </p>
        ) : null}
        <PrimaryButton onClick={() => inputRef.current?.click()}>
          <Icon name="upload" />
          {labels.chooseFile}
        </PrimaryButton>
      </div>
      <div className="upload-divider" aria-hidden="true">
        <span>{labels.or}</span>
      </div>
      <div className="microphone-option">
        <span className="microphone-icon">
          <Icon name="microphone" />
        </span>
        <div>
          <h3>{labels.microphoneTitle}</h3>
          <p>{labels.microphoneDescription}</p>
          <p className="capability-note">{labels.microphoneUnavailable}</p>
        </div>
        <SecondaryButton disabled>
          <Icon name="microphone" />
          {labels.microphoneTitle}
        </SecondaryButton>
      </div>
    </section>
  );
}

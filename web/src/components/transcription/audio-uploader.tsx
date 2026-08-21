"use client";

import { useEffect, useId, useRef, useState } from "react";

import { PrimaryButton, SecondaryButton } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";

export const AUDIO_MAX_BYTES = 25 * 1024 * 1024;
export const AUDIO_MAX_SECONDS = 30;
export const AUDIO_EXTENSIONS = [".wav", ".flac", ".ogg", ".mp3"] as const;

const MIME_TYPES: Readonly<Record<(typeof AUDIO_EXTENSIONS)[number], readonly string[]>> = {
  ".wav": ["audio/wav", "audio/x-wav", "audio/wave"],
  ".flac": ["audio/flac", "audio/x-flac"],
  ".ogg": ["audio/ogg", "application/ogg"],
  ".mp3": ["audio/mpeg", "audio/mp3"],
};

const AUDIO_ACCEPT = [...AUDIO_EXTENSIONS, ...Object.values(MIME_TYPES).flat()].join(",");

export type AudioUploaderLabels = {
  chooseFile: string;
  description: string;
  dropActive: string;
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
  disabled?: boolean;
  file: File | null;
  labels: AudioUploaderLabels;
  onFileChange: (file: File | null) => void;
};

function fileExtension(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
}

export function validateAudioFile(file: File): "file_too_large" | "invalid_file" | null {
  if (file.size > AUDIO_MAX_BYTES) return "file_too_large";
  const extension = fileExtension(file.name);
  if (!AUDIO_EXTENSIONS.some((candidate) => candidate === extension)) return "invalid_file";
  const allowedMimes = MIME_TYPES[extension as (typeof AUDIO_EXTENSIONS)[number]];
  const mime = file.type.trim().toLowerCase();
  if (mime && !allowedMimes.includes(mime)) return "invalid_file";
  return null;
}

export function safeAudioUploadFilename(file: File): string {
  const extension = fileExtension(file.name);
  return AUDIO_EXTENSIONS.some((candidate) => candidate === extension)
    ? `audio-upload${extension}`
    : "audio-upload.bin";
}

export function AudioUploader({
  disabled = false,
  file,
  labels,
  onFileChange,
}: AudioUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);
  const inputId = useId();
  const formatsId = `${inputId}-formats`;
  const errorId = `${inputId}-error`;
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file && inputRef.current) inputRef.current.value = "";
  }, [file]);

  const acceptFile = (candidate: File | undefined) => {
    if (!candidate || disabled) return;
    const validation = validateAudioFile(candidate);
    if (validation) {
      setError(validation === "file_too_large" ? labels.fileTooLarge : labels.invalidFile);
      onFileChange(null);
      if (inputRef.current) inputRef.current.value = "";
      return;
    }
    setError(null);
    onFileChange(candidate);
  };

  return (
    <section className="audio-uploader" aria-labelledby={`${inputId}-title`}>
      <div
        aria-busy={disabled}
        className={`upload-zone${dragActive ? " is-drag-active" : ""}${file ? " is-ready" : ""}`}
        onDragEnter={(event) => {
          event.preventDefault();
          if (disabled) return;
          dragDepth.current += 1;
          setDragActive(true);
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          if (disabled) return;
          dragDepth.current = Math.max(0, dragDepth.current - 1);
          if (dragDepth.current === 0) setDragActive(false);
        }}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) event.dataTransfer.dropEffect = "copy";
        }}
        onDrop={(event) => {
          event.preventDefault();
          dragDepth.current = 0;
          setDragActive(false);
          acceptFile(event.dataTransfer.files[0]);
        }}
      >
        <span className="upload-icon">
          <Icon name="upload" />
        </span>
        <h2 id={`${inputId}-title`}>{labels.title}</h2>
        <p>
          {dragActive
            ? labels.dropActive
            : file
              ? `${labels.selectedFile} : ${file.name}`
              : labels.description}
        </p>
        <span className="upload-formats" id={formatsId}>
          {labels.formats}
        </span>
        <input
          accept={AUDIO_ACCEPT}
          aria-describedby={`${formatsId}${error ? ` ${errorId}` : ""}`}
          aria-invalid={error ? "true" : undefined}
          aria-label={labels.chooseFile}
          className="visually-hidden"
          disabled={disabled}
          id={inputId}
          onChange={(event) => acceptFile(event.currentTarget.files?.[0])}
          ref={inputRef}
          type="file"
        />
        {error ? (
          <p className="field-error" id={errorId} role="alert">
            {error}
          </p>
        ) : null}
        <PrimaryButton disabled={disabled} onClick={() => inputRef.current?.click()}>
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

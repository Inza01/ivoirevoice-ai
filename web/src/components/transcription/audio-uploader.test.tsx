import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, vi } from "vitest";

import {
  AUDIO_MAX_BYTES,
  AudioUploader,
  safeAudioUploadFilename,
  type AudioUploaderLabels,
} from "@/components/transcription/audio-uploader";

const labels: AudioUploaderLabels = {
  chooseFile: "Choose",
  description: "Drop audio",
  dropActive: "Drop now",
  fileTooLarge: "Too large",
  formats: "Audio formats",
  invalidFile: "Invalid audio",
  microphoneDescription: "Record audio",
  microphoneTitle: "Microphone",
  microphoneUnavailable: "Unavailable",
  or: "or",
  selectedFile: "Selected",
  title: "Upload",
};

afterEach(() => vi.restoreAllMocks());

function Harness({ disabled = false }: { disabled?: boolean }) {
  const [file, setFile] = useState<File | null>(null);
  return <AudioUploader disabled={disabled} file={file} labels={labels} onFileChange={setFile} />;
}

describe("AudioUploader", () => {
  it("selects a supported audio file without reading its contents", () => {
    const fileReader = vi.spyOn(FileReader.prototype, "readAsArrayBuffer");
    const { container } = render(<Harness />);
    const input = container.querySelector("input[type='file']");
    const file = new File(["synthetic"], "sample.wav", { type: "audio/wav" });

    fireEvent.change(input!, { target: { files: [file] } });

    expect(screen.getByText("Selected : sample.wav")).toBeInTheDocument();
    expect(fileReader).not.toHaveBeenCalled();
  });

  it("accepts drag and drop with a keyboard-equivalent file button", () => {
    const { container } = render(<Harness />);
    const zone = container.querySelector(".upload-zone")!;
    const file = new File(["synthetic"], "sample.flac", { type: "audio/flac" });

    fireEvent.dragEnter(zone, { dataTransfer: { files: [file] } });
    expect(screen.getByText("Drop now")).toBeInTheDocument();
    fireEvent.drop(zone, { dataTransfer: { files: [file] } });

    expect(screen.getByText("Selected : sample.flac")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Choose" })).toBeEnabled();
  });

  it.each([
    ["notes.txt", "audio/wav"],
    ["sample.wav", "text/plain"],
    ["sample.m4a", "audio/mp4"],
  ])("rejects an invalid extension or MIME combination: %s", (filename, mime) => {
    const { container } = render(<Harness />);
    const input = container.querySelector("input[type='file']");
    const file = new File(["synthetic"], filename, { type: mime });

    fireEvent.change(input!, { target: { files: [file] } });

    expect(screen.getByRole("alert")).toHaveTextContent("Invalid audio");
    expect(screen.queryByText(new RegExp(filename.replace(".", "\\.")))).not.toBeInTheDocument();
  });

  it("rejects an audio file above 25 MiB", () => {
    const { container } = render(<Harness />);
    const input = container.querySelector("input[type='file']");
    const file = new File(["small"], "sample.mp3", { type: "audio/mpeg" });
    Object.defineProperty(file, "size", { value: AUDIO_MAX_BYTES + 1 });

    fireEvent.change(input!, { target: { files: [file] } });

    expect(screen.getByRole("alert")).toHaveTextContent("Too large");
  });

  it("accepts a missing browser MIME for a supported extension and uses a generic upload name", () => {
    const { container } = render(<Harness />);
    const input = container.querySelector("input[type='file']");
    const file = new File(["synthetic"], "personal recording.OGG", { type: "" });

    fireEvent.change(input!, { target: { files: [file] } });

    expect(screen.getByText(/personal recording\.OGG/)).toBeInTheDocument();
    expect(safeAudioUploadFilename(file)).toBe("audio-upload.ogg");
  });

  it("keeps upload and microphone controls disabled while processing", () => {
    render(<Harness disabled />);

    expect(screen.getByRole("button", { name: "Choose" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Microphone" })).toBeDisabled();
  });
});

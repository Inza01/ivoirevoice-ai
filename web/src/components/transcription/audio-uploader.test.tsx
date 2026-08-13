import { fireEvent, render, screen } from "@testing-library/react";

import { AudioUploader, type AudioUploaderLabels } from "@/components/transcription/audio-uploader";

const labels: AudioUploaderLabels = {
  chooseFile: "Choose",
  description: "Drop audio",
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

describe("AudioUploader", () => {
  it("previews an audio filename without sending or reading its contents", () => {
    const { container } = render(<AudioUploader labels={labels} />);
    const input = container.querySelector("input[type='file']");
    const file = new File(["synthetic"], "sample.wav", { type: "audio/wav" });

    fireEvent.change(input!, { target: { files: [file] } });

    expect(screen.getByText("Selected : sample.wav")).toBeInTheDocument();
  });

  it("rejects a non-audio file locally", () => {
    const { container } = render(<AudioUploader labels={labels} />);
    const input = container.querySelector("input[type='file']");
    const file = new File(["synthetic"], "notes.txt", { type: "text/plain" });

    fireEvent.change(input!, { target: { files: [file] } });

    expect(screen.getByRole("alert")).toHaveTextContent("Invalid audio");
    expect(screen.queryByText(/notes\.txt/)).not.toBeInTheDocument();
  });
});

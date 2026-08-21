"""Bounded, ephemeral staging for untrusted transcription uploads."""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from ivoirevoice.services.transcription_service import AUDIO_MIME_TYPES

UPLOAD_CHUNK_BYTES = 1024 * 1024


class UploadValidationError(Exception):
    """A privacy-safe validation failure suitable for HTTP translation."""

    def __init__(self, *, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class StagedAudio:
    """One temporary upload whose path is never part of a public response."""

    path: Path
    extension: str
    size_bytes: int


def _unsupported(message: str = "Format audio non supporté.") -> UploadValidationError:
    return UploadValidationError(
        status_code=415,
        code="unsupported_audio",
        message=message,
    )


def _signature_matches(path: Path, extension: str) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(12)
    except OSError:
        return False
    if extension == ".wav":
        return len(header) >= 12 and header[:4] in {b"RIFF", b"RF64"} and header[8:12] == b"WAVE"
    if extension == ".flac":
        return header.startswith(b"fLaC")
    if extension == ".ogg":
        return header.startswith(b"OggS")
    if extension == ".mp3":
        return header.startswith(b"ID3") or (
            len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
        )
    return False


class SecureUploadStager:
    """Write an UploadFile once under a random name and delete it unconditionally."""

    def __init__(self, *, max_size_bytes: int, temporary_root: Path | None = None) -> None:
        if max_size_bytes <= 0:
            raise ValueError("La taille maximale doit être strictement positive.")
        self.max_size_bytes = max_size_bytes
        self.temporary_root = temporary_root

    @asynccontextmanager
    async def stage(self, upload: UploadFile) -> AsyncIterator[StagedAudio]:
        """Validate metadata/signature and yield a private temporary file."""

        try:
            extension = Path(upload.filename or "").suffix.lower()
            allowed_mimes = AUDIO_MIME_TYPES.get(extension)
            if allowed_mimes is None:
                raise _unsupported()
            content_type = (upload.content_type or "").partition(";")[0].strip().lower()
            if content_type not in allowed_mimes:
                raise _unsupported()

            if self.temporary_root is not None:
                self.temporary_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="ivoirevoice-upload-",
                dir=self.temporary_root,
            ) as directory:
                destination = Path(directory) / f"{uuid4().hex}{extension}"
                size_bytes = 0
                try:
                    with destination.open("xb") as stream:
                        while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                            size_bytes += len(chunk)
                            if size_bytes > self.max_size_bytes:
                                raise UploadValidationError(
                                    status_code=413,
                                    code="payload_too_large",
                                    message="Audio trop volumineux.",
                                )
                            stream.write(chunk)
                except UploadValidationError:
                    raise
                except OSError as exc:
                    raise UploadValidationError(
                        status_code=503,
                        code="service_unavailable",
                        message="Serveur momentanément indisponible.",
                    ) from exc

                if size_bytes == 0:
                    raise _unsupported("Le fichier audio est vide.")
                if not _signature_matches(destination, extension):
                    raise _unsupported()
                yield StagedAudio(
                    path=destination,
                    extension=extension,
                    size_bytes=size_bytes,
                )
        finally:
            # Starlette may have spooled multipart data before the endpoint is
            # called. Closing it is required on every success and failure path.
            with suppress(Exception):
                await upload.close()

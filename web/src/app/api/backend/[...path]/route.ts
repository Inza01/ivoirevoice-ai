const AUDIO_LIMIT_BYTES = 25 * 1024 * 1024;
const MULTIPART_OVERHEAD_BYTES = 256 * 1024;
const TRANSCRIPTION_REQUEST_LIMIT_BYTES = AUDIO_LIMIT_BYTES + MULTIPART_OVERHEAD_BYTES;
const TRANSLATION_REQUEST_LIMIT_BYTES = 64 * 1024;
const BACKEND_RESPONSE_LIMIT_BYTES = 2 * 1024 * 1024;
const BACKEND_TIMEOUT_MS = 120_000;
const SAFE_TRANSCRIPTION_ID = /^[A-Za-z0-9_-]{1,128}$/;

type ProxyContext = {
  readonly params: Promise<{ readonly path: readonly string[] }>;
};

type AllowedMethod = "GET" | "POST";

type Contract = {
  readonly backendPath: string;
  readonly method: AllowedMethod;
  readonly requestLimitBytes?: number;
  readonly requestMediaType?: "application/json" | "multipart/form-data";
};

type PublicErrorCode =
  | "invalid_request"
  | "method_not_allowed"
  | "not_found"
  | "payload_too_large"
  | "service_unavailable"
  | "unsupported_media_type"
  | "upstream_response_too_large";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

function publicError(status: number, code: PublicErrorCode, allow?: string): Response {
  const headers = new Headers({
    "Cache-Control": "no-store",
    "Content-Type": "application/json; charset=utf-8",
  });
  if (allow) headers.set("Allow", allow);
  return new Response(JSON.stringify({ error: { code } }), { status, headers });
}

function contractFor(path: readonly string[]): Contract | null {
  const joined = path.join("/");
  if (joined === "api/health") return { backendPath: "/api/health", method: "GET" };
  if (joined === "api/v1/models") return { backendPath: "/api/v1/models", method: "GET" };
  if (joined === "api/v1/languages") return { backendPath: "/api/v1/languages", method: "GET" };
  if (joined === "api/v1/transcriptions") {
    return {
      backendPath: "/api/v1/transcriptions",
      method: "POST",
      requestLimitBytes: TRANSCRIPTION_REQUEST_LIMIT_BYTES,
      requestMediaType: "multipart/form-data",
    };
  }
  if (joined === "api/v1/translations") {
    return {
      backendPath: "/api/v1/translations",
      method: "POST",
      requestLimitBytes: TRANSLATION_REQUEST_LIMIT_BYTES,
      requestMediaType: "application/json",
    };
  }
  if (
    path.length === 4 &&
    path[0] === "api" &&
    path[1] === "v1" &&
    path[2] === "transcriptions" &&
    SAFE_TRANSCRIPTION_ID.test(path[3] ?? "")
  ) {
    return { backendPath: `/api/v1/transcriptions/${path[3]}`, method: "GET" };
  }
  return null;
}

function backendOrigin(): URL | null {
  const rawOrigin = process.env.IVOIREVOICE_API_INTERNAL_URL?.trim();
  if (!rawOrigin) return null;

  try {
    const origin = new URL(rawOrigin);
    const isHttp = origin.protocol === "http:" || origin.protocol === "https:";
    const isRootOrigin = origin.pathname === "/" && !origin.search && !origin.hash;
    if (!isHttp || !origin.hostname || origin.username || origin.password || !isRootOrigin) {
      return null;
    }
    return origin;
  } catch {
    return null;
  }
}

function parseContentLength(request: Request, maximum: number): number | Response {
  const rawLength = request.headers.get("Content-Length");
  if (!rawLength || !/^\d+$/.test(rawLength)) {
    return publicError(411, "invalid_request");
  }

  const length = Number(rawLength);
  if (!Number.isSafeInteger(length) || length <= 0) {
    return publicError(400, "invalid_request");
  }
  if (length > maximum) {
    return publicError(413, "payload_too_large");
  }
  return length;
}

async function readBoundedBody(
  body: ReadableStream<Uint8Array> | null,
  maximum: number,
): Promise<ArrayBuffer | null> {
  if (!body) return null;

  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maximum) {
        await reader.cancel();
        return null;
      }
      chunks.push(value);
    }
  } catch {
    try {
      await reader.cancel();
    } catch {
      // A failed stream is handled as a bounded, privacy-safe request error.
    }
    return null;
  } finally {
    reader.releaseLock();
  }

  const combined = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return combined.buffer;
}

async function discardBody(response: Response): Promise<void> {
  try {
    await response.body?.cancel();
  } catch {
    // Error payloads are intentionally not consumed or exposed.
  }
}

function contentTypeMatches(request: Request, expected: NonNullable<Contract["requestMediaType"]>) {
  const contentType = request.headers.get("Content-Type")?.toLowerCase() ?? "";
  if (expected === "multipart/form-data") {
    return contentType.startsWith("multipart/form-data;") && contentType.includes("boundary=");
  }
  return contentType === "application/json" || contentType.startsWith("application/json;");
}

function isSameOrigin(request: Request): boolean {
  const origin = request.headers.get("Origin");
  if (!origin) return false;
  try {
    return new URL(origin).origin === new URL(request.url).origin;
  } catch {
    return false;
  }
}

function sanitizedBackendError(status: number): Response {
  if (status === 400 || status === 422) return publicError(status, "invalid_request");
  if (status === 404) return publicError(404, "not_found");
  if (status === 413) return publicError(413, "payload_too_large");
  if (status === 415) return publicError(415, "unsupported_media_type");
  return publicError(status >= 500 ? 502 : status, "service_unavailable");
}

async function proxy(request: Request, context: ProxyContext): Promise<Response> {
  const { path } = await context.params;
  const contract = contractFor(path);
  if (!contract || new URL(request.url).search) {
    return publicError(404, "not_found");
  }
  if (request.method !== contract.method) {
    return publicError(405, "method_not_allowed", contract.method);
  }

  const origin = backendOrigin();
  if (!origin) return publicError(503, "service_unavailable");

  let body: ArrayBuffer | undefined;
  if (contract.method === "POST") {
    if (!isSameOrigin(request)) return publicError(403, "invalid_request");
    if (!contract.requestLimitBytes || !contract.requestMediaType) {
      return publicError(500, "service_unavailable");
    }
    if (!contentTypeMatches(request, contract.requestMediaType)) {
      return publicError(415, "unsupported_media_type");
    }
    const declaredLength = parseContentLength(request, contract.requestLimitBytes);
    if (declaredLength instanceof Response) return declaredLength;
    const requestBody = await readBoundedBody(request.body, contract.requestLimitBytes);
    if (!requestBody || requestBody.byteLength !== declaredLength) {
      return publicError(
        requestBody && requestBody.byteLength < declaredLength ? 400 : 413,
        requestBody && requestBody.byteLength < declaredLength
          ? "invalid_request"
          : "payload_too_large",
      );
    }
    body = requestBody;
  }

  const headers = new Headers({ Accept: "application/json" });
  const requestContentType = request.headers.get("Content-Type");
  if (body && requestContentType) headers.set("Content-Type", requestContentType);

  let backendResponse: Response;
  try {
    backendResponse = await fetch(new URL(contract.backendPath, origin), {
      body,
      cache: "no-store",
      headers,
      method: contract.method,
      redirect: "error",
      signal: AbortSignal.timeout(BACKEND_TIMEOUT_MS),
    });
  } catch {
    return publicError(502, "service_unavailable");
  }

  if (!backendResponse.ok) {
    await discardBody(backendResponse);
    return sanitizedBackendError(backendResponse.status);
  }

  const declaredResponseLength = backendResponse.headers.get("Content-Length");
  if (
    declaredResponseLength &&
    (!/^\d+$/.test(declaredResponseLength) ||
      Number(declaredResponseLength) > BACKEND_RESPONSE_LIMIT_BYTES)
  ) {
    await discardBody(backendResponse);
    return publicError(502, "upstream_response_too_large");
  }
  const responseContentType = backendResponse.headers.get("Content-Type");
  if (!responseContentType?.toLowerCase().startsWith("application/json")) {
    await discardBody(backendResponse);
    return publicError(502, "service_unavailable");
  }
  const responseBody = await readBoundedBody(backendResponse.body, BACKEND_RESPONSE_LIMIT_BYTES);
  if (!responseBody) return publicError(502, "upstream_response_too_large");

  const responseHeaders = new Headers({
    "Cache-Control": "no-store",
    "Content-Type": responseContentType,
  });
  return new Response(responseBody, { status: backendResponse.status, headers: responseHeaders });
}

export async function GET(request: Request, context: ProxyContext): Promise<Response> {
  return proxy(request, context);
}

export async function POST(request: Request, context: ProxyContext): Promise<Response> {
  return proxy(request, context);
}

export async function DELETE(request: Request, context: ProxyContext): Promise<Response> {
  return proxy(request, context);
}

export async function HEAD(request: Request, context: ProxyContext): Promise<Response> {
  return proxy(request, context);
}

export async function OPTIONS(request: Request, context: ProxyContext): Promise<Response> {
  return proxy(request, context);
}

export async function PATCH(request: Request, context: ProxyContext): Promise<Response> {
  return proxy(request, context);
}

export async function PUT(request: Request, context: ProxyContext): Promise<Response> {
  return proxy(request, context);
}

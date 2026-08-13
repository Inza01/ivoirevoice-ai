import { afterEach, describe, expect, it, vi } from "vitest";

import { GET, POST, PUT } from "./route";

const BACKEND_ORIGIN = "http://127.0.0.1:8000";
const FRONTEND_ORIGIN = "http://127.0.0.1:3000";

function context(...path: string[]) {
  return { params: Promise.resolve({ path }) };
}

function request(path: string, init?: RequestInit): Request {
  return new Request(`${FRONTEND_ORIGIN}/api/backend/${path}`, init);
}

function postRequest(path: string, body: string, contentType: string): Request {
  return request(path, {
    body,
    headers: {
      "Content-Length": String(new TextEncoder().encode(body).byteLength),
      "Content-Type": contentType,
      Cookie: "private=session",
      Origin: FRONTEND_ORIGIN,
      Authorization: "Bearer private",
    },
    method: "POST",
  });
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("allowlisted backend route", () => {
  it("forwards an allowlisted GET to the configured server-only origin", async () => {
    vi.stubEnv("IVOIREVOICE_API_INTERNAL_URL", BACKEND_ORIGIN);
    const backendFetch = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json({ service: "ivoirevoice", status: "ok", version: "1" }));
    vi.stubGlobal("fetch", backendFetch);

    const response = await GET(request("api/health"), context("api", "health"));

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ service: "ivoirevoice", status: "ok", version: "1" });
    expect(backendFetch).toHaveBeenCalledOnce();
    const [url, init] = backendFetch.mock.calls[0] ?? [];
    expect(String(url)).toBe(`${BACKEND_ORIGIN}/api/health`);
    expect(init?.method).toBe("GET");
    expect(init?.cache).toBe("no-store");
  });

  it("returns a privacy-safe 404 without calling the backend for an unknown path", async () => {
    vi.stubEnv("IVOIREVOICE_API_INTERNAL_URL", BACKEND_ORIGIN);
    const backendFetch = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", backendFetch);

    const response = await GET(request("private/admin"), context("private", "admin"));

    expect(response.status).toBe(404);
    expect(await response.json()).toEqual({ error: { code: "not_found" } });
    expect(backendFetch).not.toHaveBeenCalled();
  });

  it("returns 405 and the contract Allow value for a known path with a wrong method", async () => {
    vi.stubEnv("IVOIREVOICE_API_INTERNAL_URL", BACKEND_ORIGIN);
    const backendFetch = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", backendFetch);

    const response = await POST(
      postRequest("api/health", "{}", "application/json"),
      context("api", "health"),
    );

    expect(response.status).toBe(405);
    expect(response.headers.get("Allow")).toBe("GET");
    expect(await response.json()).toEqual({ error: { code: "method_not_allowed" } });
    expect(backendFetch).not.toHaveBeenCalled();
  });

  it("handles other registered HTTP methods without opening the allowlist", async () => {
    vi.stubEnv("IVOIREVOICE_API_INTERNAL_URL", BACKEND_ORIGIN);
    const backendFetch = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", backendFetch);

    const known = await PUT(
      request("api/v1/models", { method: "PUT" }),
      context("api", "v1", "models"),
    );
    const unknown = await PUT(request("private", { method: "PUT" }), context("private"));

    expect(known.status).toBe(405);
    expect(known.headers.get("Allow")).toBe("GET");
    expect(unknown.status).toBe(404);
    expect(backendFetch).not.toHaveBeenCalled();
  });

  it("rejects unsafe transcription identifiers and query variants", async () => {
    vi.stubEnv("IVOIREVOICE_API_INTERNAL_URL", BACKEND_ORIGIN);
    const backendFetch = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", backendFetch);

    const unsafeId = await GET(
      request("api/v1/transcriptions/bad.id"),
      context("api", "v1", "transcriptions", "bad.id"),
    );
    const query = await GET(request("api/v1/models?private=true"), context("api", "v1", "models"));

    expect(unsafeId.status).toBe(404);
    expect(query.status).toBe(404);
    expect(backendFetch).not.toHaveBeenCalled();
  });

  it("rejects cross-origin mutations before reading or forwarding their body", async () => {
    vi.stubEnv("IVOIREVOICE_API_INTERNAL_URL", BACKEND_ORIGIN);
    const backendFetch = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", backendFetch);
    const body = JSON.stringify({ source_language: "fr", target_language: "dyu", text: "test" });
    const crossOrigin = request("api/v1/translations", {
      body,
      headers: {
        "Content-Length": String(body.length),
        "Content-Type": "application/json",
        Origin: "https://attacker.invalid",
      },
      method: "POST",
    });

    const response = await POST(crossOrigin, context("api", "v1", "translations"));

    expect(response.status).toBe(403);
    expect(await response.json()).toEqual({ error: { code: "invalid_request" } });
    expect(backendFetch).not.toHaveBeenCalled();
  });

  it("rejects an upload declared above 25 MiB plus multipart overhead", async () => {
    vi.stubEnv("IVOIREVOICE_API_INTERNAL_URL", BACKEND_ORIGIN);
    const backendFetch = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", backendFetch);
    const upload = request("api/v1/transcriptions", {
      body: "small",
      headers: {
        "Content-Length": String(25 * 1024 * 1024 + 256 * 1024 + 1),
        "Content-Type": "multipart/form-data; boundary=safe",
        Origin: FRONTEND_ORIGIN,
      },
      method: "POST",
    });

    const response = await POST(upload, context("api", "v1", "transcriptions"));

    expect(response.status).toBe(413);
    expect(await response.json()).toEqual({ error: { code: "payload_too_large" } });
    expect(backendFetch).not.toHaveBeenCalled();
  });

  it("forwards only the safe request headers on an allowed mutation", async () => {
    vi.stubEnv("IVOIREVOICE_API_INTERNAL_URL", BACKEND_ORIGIN);
    const backendFetch = vi.fn<typeof fetch>().mockResolvedValue(
      Response.json({
        id: "translation_1",
        source_language: "fr",
        status: "queued",
        target_language: "dyu",
      }),
    );
    vi.stubGlobal("fetch", backendFetch);
    const body = JSON.stringify({ source_language: "fr", target_language: "dyu", text: "test" });

    const response = await POST(
      postRequest("api/v1/translations", body, "application/json"),
      context("api", "v1", "translations"),
    );

    expect(response.status).toBe(200);
    const [, init] = backendFetch.mock.calls[0] ?? [];
    const forwardedHeaders = new Headers(init?.headers);
    expect(forwardedHeaders.get("Content-Type")).toBe("application/json");
    expect(forwardedHeaders.get("Cookie")).toBeNull();
    expect(forwardedHeaders.get("Authorization")).toBeNull();
  });

  it("does not expose upstream messages, stack traces, or private paths", async () => {
    vi.stubEnv("IVOIREVOICE_API_INTERNAL_URL", BACKEND_ORIGIN);
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>().mockResolvedValue(
        Response.json(
          {
            message: "failed at /home/private/checkpoint-000001",
            stack: "secret stack",
          },
          { status: 500 },
        ),
      ),
    );

    const response = await GET(request("api/v1/models"), context("api", "v1", "models"));
    const publicBody = await response.text();

    expect(response.status).toBe(502);
    expect(JSON.parse(publicBody)).toEqual({ error: { code: "service_unavailable" } });
    expect(publicBody).not.toContain("/home/");
    expect(publicBody).not.toContain("stack");
  });

  it.each([
    "ftp://127.0.0.1:8000",
    "http://user:password@127.0.0.1:8000",
    "http://127.0.0.1:8000/private",
    "http://127.0.0.1:8000?token=secret",
  ])("rejects an invalid server-only backend origin: %s", async (backendOrigin) => {
    vi.stubEnv("IVOIREVOICE_API_INTERNAL_URL", backendOrigin);
    const backendFetch = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", backendFetch);

    const response = await GET(request("api/health"), context("api", "health"));

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ error: { code: "service_unavailable" } });
    expect(backendFetch).not.toHaveBeenCalled();
  });
});

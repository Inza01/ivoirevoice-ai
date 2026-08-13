import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Next 16 marks the API/CLI selector experimental. Explicit `false` keeps
  // the stable TypeScript compiler API and avoids detached child processes.
  experimental: {
    // Keep build-time type checking in-process. This preserves the full Next
    // type gate in restricted local/CI workers that disallow detached children.
    useTypeScriptCli: false,
  },
  poweredByHeader: false,
  reactStrictMode: true,
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "X-Frame-Options", value: "DENY" },
          {
            key: "Permissions-Policy",
            value: "camera=(), geolocation=(), microphone=(self)",
          },
        ],
      },
    ];
  },
};

export default nextConfig;

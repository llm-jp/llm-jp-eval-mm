import type { NextConfig } from "next";

const isStaticExport = process.env.STATIC_EXPORT === "1";
const staticBasePath = "/llm-jp-eval-mm";

const staticConfig: NextConfig = {
  output: "export",
  basePath: staticBasePath,
  images: { unoptimized: true },
  trailingSlash: true,
  env: {
    NEXT_PUBLIC_BASE_PATH: staticBasePath,
    NEXT_PUBLIC_STATIC_EXPORT: "1",
  },
};

const devConfig: NextConfig = {
  async rewrites() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },
};

export default isStaticExport ? staticConfig : devConfig;

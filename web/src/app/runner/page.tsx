"use client";

import dynamic from "next/dynamic";
import { IS_STATIC_EXPORT } from "@/lib/base-path";
import { BackendOnlyNotice } from "@/components/backend-only-notice";

const RunnerContent = dynamic(() => import("@/components/runner-content"), {
  ssr: false,
});

export default function RunnerPage() {
  if (IS_STATIC_EXPORT) return <BackendOnlyNotice page="Runner" />;
  return <RunnerContent />;
}

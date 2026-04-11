"use client";

import dynamic from "next/dynamic";

const RunnerContent = dynamic(() => import("@/components/runner-content"), {
  ssr: false,
});

export default function RunnerPage() {
  return <RunnerContent />;
}

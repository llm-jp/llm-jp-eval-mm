import Image from "next/image";
import { withBasePath } from "@/lib/base-path";

export function Introduction() {
  return (
    <section className="flex flex-col gap-6">
      <h2 className="text-2xl font-semibold tracking-tight">Introduction</h2>
      <p className="text-base leading-relaxed">
        We introduce <b>llm-jp-eval-mm</b>, a toolkit for evaluating multiple
        multimodal tasks related to Japanese language performance in a unified
        environment. The toolkit is a benchmarking platform that integrates
        existing Japanese multimodal tasks and consistently evaluates model
        outputs across multiple metrics. This paper outlines the design of
        llm-jp-eval-mm for its construction and ongoing development, reports
        the results of evaluating publicly available Japanese and multilingual
        VLMs, and discusses the findings in the light of existing research.
      </p>
      <figure className="flex flex-col items-center gap-2">
        <Image
          src={withBasePath("/teaser.png")}
          alt="Overview of llm-jp-eval-mm"
          width={1200}
          height={675}
          className="h-auto w-full max-w-4xl rounded-md border"
          unoptimized
        />
        <figcaption className="text-sm text-muted-foreground">
          Figure 1: <b>Overview of llm-jp-eval-mm.</b>
        </figcaption>
      </figure>
    </section>
  );
}

import { PaperMeta } from "@/components/landing/paper-meta";
import { Introduction } from "@/components/landing/introduction";
import { StaticLeaderboard } from "@/components/landing/static-leaderboard";
import { BibTex } from "@/components/landing/bibtex";

export default function Home() {
  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-16 px-4 py-16 sm:px-6 lg:px-8">
      <PaperMeta />
      <Introduction />
      <StaticLeaderboard />
      <BibTex />
    </div>
  );
}

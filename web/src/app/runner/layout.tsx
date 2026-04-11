export default function RunnerLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-[calc(100vh-3.5rem)] bg-runner-bg">{children}</div>
  );
}

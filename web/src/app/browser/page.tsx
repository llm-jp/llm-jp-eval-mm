export default function BrowserPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
      <h1 className="text-3xl font-bold tracking-tight">Prediction Browser</h1>
      <p className="mt-2 text-muted-foreground">
        Browse individual model predictions and ground-truth comparisons.
      </p>

      <div className="mt-8 rounded-lg border bg-card p-6 text-card-foreground">
        <p className="text-sm text-muted-foreground">
          Prediction browser coming soon. This page will let you inspect
          per-sample inputs, model outputs, and scoring details.
        </p>
      </div>
    </div>
  );
}

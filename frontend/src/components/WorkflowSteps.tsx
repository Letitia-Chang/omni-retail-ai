import { Link } from "@tanstack/react-router";
import { FileText, Sparkles, Table2, Users } from "lucide-react";

const STEPS = [
  {
    to: "/segments",
    icon: Users,
    title: "Pick a segment",
    description: "See AI-ranked product picks and why each was chosen.",
  },
  {
    to: "/campaigns",
    icon: Table2,
    title: "Browse all picks",
    description: "Every curated recommendation, searchable and filterable.",
  },
  {
    to: "/generator",
    icon: Sparkles,
    title: "Generate ad copy",
    description: "Live, RAG-grounded copy for any product, on demand.",
  },
  {
    to: "/ad-copies",
    icon: FileText,
    title: "Review & export",
    description: "Compare generated copy and export a finished CSV.",
  },
] as const;

// Deliberately not a numbered "step 0" here: this app has no upload/import
// screen anywhere in the UI. Data prep (cleaning, segmentation, purchase-
// intent scoring, campaign ranking) is a separate offline batch pipeline —
// see scripts/run_full_pipeline.py — not something a user of this dashboard
// ever does by hand. This line credits that foundation without implying
// the 4 steps below include it.
export function WorkflowSteps() {
  return (
    <div className="mb-6 rounded-xl border border-primary/15 bg-gradient-to-br from-primary/[0.06] to-transparent p-5">
      <p className="mb-5 text-xs text-muted-foreground">
        Catalog and customer data is cleaned, segmented, and scored offline by a batch
        pipeline — this is what you do with the results:
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-x-4 gap-y-5">
        {STEPS.map((step, i) => {
          const Icon = step.icon;
          return (
            <Link key={step.to} to={step.to} className="group relative flex gap-3">
              <div className="relative z-10 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-semibold text-primary-foreground shadow-sm">
                {i + 1}
              </div>
              <div className="min-w-0 pt-1">
                <div className="flex items-center gap-1.5">
                  <Icon className="h-3.5 w-3.5 shrink-0 text-primary" />
                  <div className="text-sm font-medium transition-colors group-hover:text-primary">
                    {step.title}
                  </div>
                </div>
                <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
                  {step.description}
                </p>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}

import { useEffect, useState } from "react";
import { Sparkles, X } from "lucide-react";

const STORAGE_PREFIX = "omniretail-how-this-works-dismissed:";

/** One-line "How this works" explainer, dismissible independently per page
 * (each `id` gets its own localStorage key) so dismissing it on one page
 * doesn't silently hide it everywhere else. */
export function HowThisWorks({ id, children }: { id: string; children: React.ReactNode }) {
  const storageKey = STORAGE_PREFIX + id;
  const [dismissed, setDismissed] = useState(true);

  useEffect(() => {
    setDismissed(window.localStorage.getItem(storageKey) === "1");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey]);

  if (dismissed) return null;

  const dismiss = () => {
    window.localStorage.setItem(storageKey, "1");
    setDismissed(true);
  };

  return (
    <div className="mb-6 flex items-start gap-3 rounded-lg border border-primary/20 bg-primary/5 px-4 py-3">
      <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
      <p className="flex-1 text-sm text-foreground/85">
        <span className="font-medium text-foreground">How this works: </span>
        {children}
      </p>
      <button
        type="button"
        onClick={dismiss}
        aria-label="Dismiss"
        className="shrink-0 rounded-md p-1 text-muted-foreground hover:bg-primary/10 hover:text-foreground"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}

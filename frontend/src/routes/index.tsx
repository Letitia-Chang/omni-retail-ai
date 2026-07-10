import { createFileRoute, redirect } from "@tanstack/react-router";

// Overview leads the sidebar and carries the 4-step workflow guide, so
// it's the landing experience — a first-time visitor sees "here's how this
// works" before anything else. Segments (the hands-on segment-by-segment
// explorer) lives at /segments, one click away via the workflow guide or
// the nav.
export const Route = createFileRoute("/")({
  beforeLoad: () => {
    throw redirect({ to: "/overview" });
  },
});

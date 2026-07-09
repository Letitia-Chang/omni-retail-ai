import { createFileRoute, redirect } from "@tanstack/react-router";

// Segments is the more useful first screen — pick a segment, see why each
// product was picked, generate copy — so it's the landing experience.
// Overview (the summary dashboard) lives at /overview for once someone
// already understands the system and wants the bird's-eye view.
export const Route = createFileRoute("/")({
  beforeLoad: () => {
    throw redirect({ to: "/segments" });
  },
});

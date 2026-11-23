import { QueryClient } from "@tanstack/react-query";
import { createRouter } from "@tanstack/react-router";
import { routeTree } from "./routeTree.gen";

export const getRouter = () => {
  // The demo backend runs on a free tier that sleeps when idle and takes
  // about a minute to wake. Keep retrying for ~75s instead of failing fast.
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: 6,
        retryDelay: (attempt) => Math.min(2000 * 2 ** attempt, 20000),
      },
    },
  });

  const router = createRouter({
    routeTree,
    context: { queryClient },
    scrollRestoration: true,
    defaultPreloadStaleTime: 0,
  });

  return router;
};

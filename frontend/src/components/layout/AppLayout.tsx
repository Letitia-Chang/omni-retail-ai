import { Link, Outlet, useRouterState } from "@tanstack/react-router";
import { LayoutDashboard, Users, Table2, Sparkles, FileText, BarChart3, Bot } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/segments", label: "Segments", icon: Users },
  { to: "/campaigns", label: "Campaigns", icon: Table2 },
  { to: "/generator", label: "AI Generator", icon: Sparkles },
  { to: "/ad-copies", label: "Ad Copies", icon: FileText },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/overview", label: "Overview", icon: LayoutDashboard },
] as const;

export function AppLayout() {
  const path = useRouterState({ select: (s) => s.location.pathname });

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <aside className="hidden md:flex w-64 shrink-0 flex-col border-r border-border bg-sidebar">
        <div className="flex items-center gap-2 px-6 py-5 border-b border-sidebar-border">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent text-primary-foreground shadow-sm">
            <Bot className="h-5 w-5" />
          </div>
          <div>
            <div className="text-sm font-semibold tracking-tight">OmniRetail</div>
            <div className="text-xs text-muted-foreground">AI Marketing</div>
          </div>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {NAV.map((item) => {
            const active = path.startsWith(item.to);
            const Icon = item.icon;
            return (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  active
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                )}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="p-4 border-t border-sidebar-border text-xs text-muted-foreground">
          Connected to FastAPI
        </div>
      </aside>

      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile branding + nav */}
        <div className="md:hidden flex items-center gap-2 px-4 py-3 border-b border-border bg-sidebar">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Bot className="h-4 w-4" />
          </div>
          <span className="font-semibold text-sm">OmniRetail</span>
        </div>
        <nav className="md:hidden flex gap-1 overflow-x-auto px-3 py-2 border-b border-border bg-sidebar">
          {NAV.map((item) => {
            const active = path.startsWith(item.to);
            return (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  "px-3 py-1.5 rounded-md text-xs font-medium whitespace-nowrap",
                  active ? "bg-primary text-primary-foreground" : "text-muted-foreground"
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <main className="flex-1 p-4 md:p-8 max-w-[1400px] w-full mx-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { clsx } from "clsx";

const links = [
  { href: "/", label: "Overview" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/history", label: "History" },
  { href: "/docs", label: "Docs" },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 bg-surface/80 backdrop-blur-md border-b border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 flex-shrink-0 group">
          <div className="relative">
            <div className="w-2 h-2 rounded-full bg-profit animate-pulse-slow" />
            <div className="absolute inset-0 w-2 h-2 rounded-full bg-profit opacity-30 scale-150" />
          </div>
          <span className="font-display font-bold text-base tracking-tight text-primary">
            PL<span className="text-accent">Bot</span>
          </span>
        </Link>

        {/* Nav links — always visible, shrink gracefully on small screens */}
        <nav className="flex items-center gap-0.5 sm:gap-1">
          {links.map(({ href, label }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={clsx(
                  // Base: tighter on mobile, normal on sm+
                  "px-2.5 py-1.5 sm:px-3.5 rounded-lg font-medium transition-all duration-150",
                  // Font: smaller on mobile so all four fit without wrapping
                  "text-xs sm:text-sm",
                  active
                    ? "bg-subtle text-primary"
                    : "text-muted hover:text-primary hover:bg-subtle/60",
                )}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}

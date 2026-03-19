'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { clsx } from 'clsx'

const links = [
  { href: '/',           label: 'Overview'  },
  { href: '/dashboard',  label: 'Dashboard' },
  { href: '/history',    label: 'History'   },
]

export default function Navbar() {
  const pathname = usePathname()

  return (
    <header className="sticky top-0 z-50 bg-surface/80 backdrop-blur-md border-b border-border">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">

        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5 group">
          {/* Animated dot — green when profit, red when loss (static green placeholder) */}
          <div className="relative">
            <div className="w-2 h-2 rounded-full bg-profit animate-pulse-slow" />
            <div className="absolute inset-0 w-2 h-2 rounded-full bg-profit opacity-30 scale-150" />
          </div>
          <span className="font-display font-bold text-base tracking-tight text-primary">
            PL<span className="text-accent">Bot</span>
          </span>
        </Link>

        {/* Nav links */}
        <nav className="flex items-center gap-1">
          {links.map(({ href, label }) => {
            const active = pathname === href
            return (
              <Link
                key={href}
                href={href}
                className={clsx(
                  'px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all duration-150',
                  active
                    ? 'bg-subtle text-primary'
                    : 'text-muted hover:text-primary hover:bg-subtle/60'
                )}
              >
                {label}
              </Link>
            )
          })}
        </nav>

        {/* Live indicator */}
        <div className="flex items-center gap-2 text-xs text-muted">
          <span className="w-1.5 h-1.5 rounded-full bg-profit inline-block" />
          Live
        </div>

      </div>
    </header>
  )
}

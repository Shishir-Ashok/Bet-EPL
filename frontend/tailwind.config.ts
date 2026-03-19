import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Primary palette
        primary: {
          DEFAULT: '#0B1120',
          50:  '#F0F4FF',
          100: '#E0E8FF',
          500: '#3B5BDB',
          900: '#0B1120',
        },
        // Accent — interactive, CTAs
        accent: {
          DEFAULT: '#3B82F6',
          hover:   '#2563EB',
          light:   '#EFF6FF',
        },
        // Semantic
        profit: {
          DEFAULT: '#10B981',
          bg:      '#ECFDF5',
          border:  '#6EE7B7',
        },
        loss: {
          DEFAULT: '#EF4444',
          bg:      '#FEF2F2',
          border:  '#FCA5A5',
        },
        draw: {
          DEFAULT: '#F59E0B',
          bg:      '#FFFBEB',
          border:  '#FCD34D',
        },
        pending: {
          DEFAULT: '#8B5CF6',
          bg:      '#F5F3FF',
        },
        // Surfaces
        surface:    '#FFFFFF',
        background: '#F8F9FB',
        border:     '#E2E8F0',
        muted:      '#94A3B8',
        subtle:     '#F1F5F9',
      },
      fontFamily: {
        display: ['var(--font-bricolage)', 'sans-serif'],
        body:    ['var(--font-jakarta)', 'sans-serif'],
        mono:    ['var(--font-mono)', 'monospace'],
      },
      fontSize: {
        'display-2xl': ['4.5rem',  { lineHeight: '1.05', letterSpacing: '-0.03em', fontWeight: '700' }],
        'display-xl':  ['3.75rem', { lineHeight: '1.1',  letterSpacing: '-0.025em', fontWeight: '700' }],
        'display-lg':  ['3rem',    { lineHeight: '1.15', letterSpacing: '-0.02em', fontWeight: '700' }],
        'display-md':  ['2.25rem', { lineHeight: '1.2',  letterSpacing: '-0.015em', fontWeight: '600' }],
        'display-sm':  ['1.875rem',{ lineHeight: '1.25', letterSpacing: '-0.01em', fontWeight: '600' }],
      },
      boxShadow: {
        'card':    '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
        'card-md': '0 4px 12px rgba(0,0,0,0.08), 0 1px 3px rgba(0,0,0,0.05)',
        'card-lg': '0 10px 30px rgba(0,0,0,0.1), 0 2px 8px rgba(0,0,0,0.06)',
      },
      borderRadius: {
        '2.5xl': '1.25rem',
      },
      animation: {
        'fade-up': 'fadeUp 0.5s ease forwards',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
      },
      keyframes: {
        fadeUp: {
          '0%':   { opacity: '0', transform: 'translateY(16px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}

export default config

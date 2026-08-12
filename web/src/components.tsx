/** Shared presentation pieces. */

import type { ReactNode } from 'react'
import type { RunStatus } from './api'

export function Panel({
  title,
  action,
  children,
}: {
  title?: string
  action?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="rounded-lg border border-neutral-800 bg-neutral-900/40">
      {title && (
        <header className="flex items-center justify-between border-b border-neutral-800 px-4 py-2.5">
          <h2 className="text-sm font-semibold text-neutral-300">{title}</h2>
          {action}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  )
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex gap-3 py-1 text-sm">
      <dt className="w-36 shrink-0 text-neutral-500">{label}</dt>
      <dd className="min-w-0 flex-1 text-neutral-200">{children}</dd>
    </div>
  )
}

export function Button({
  children,
  onClick,
  variant = 'default',
  disabled,
  type = 'button',
}: {
  children: ReactNode
  onClick?: () => void
  variant?: 'default' | 'primary' | 'danger' | 'ghost'
  disabled?: boolean
  type?: 'button' | 'submit'
}) {
  const styles = {
    primary: 'bg-emerald-600 text-white hover:bg-emerald-500 disabled:bg-emerald-900',
    danger: 'bg-red-900/60 text-red-200 hover:bg-red-800/60 border border-red-900',
    ghost: 'text-neutral-400 hover:text-neutral-100 hover:bg-neutral-800',
    default: 'bg-neutral-800 text-neutral-200 hover:bg-neutral-700 border border-neutral-700',
  }[variant]

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${styles}`}
    >
      {children}
    </button>
  )
}

const STATUS_STYLES: Record<RunStatus, string> = {
  completed: 'bg-emerald-950 text-emerald-300 border-emerald-900',
  running: 'bg-sky-950 text-sky-300 border-sky-900',
  failed: 'bg-red-950 text-red-300 border-red-900',
  cancelled: 'bg-amber-950 text-amber-300 border-amber-900',
  pending: 'bg-neutral-800 text-neutral-400 border-neutral-700',
}

export function StatusBadge({ status, alive }: { status: RunStatus; alive?: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-xs font-medium ${STATUS_STYLES[status]}`}
    >
      {status === 'running' && alive && (
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-400" />
      )}
      {status}
    </span>
  )
}

export function Progress({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-neutral-800">
      <div
        className="h-full rounded-full bg-emerald-500 transition-all duration-500"
        style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }}
      />
    </div>
  )
}

/** Planner and ingest warnings. These are the honest part of the product — they say
 *  what the run will and will not achieve — so they are styled to be read, not dismissed. */
export function Notes({ items, tone = 'warn' }: { items: string[]; tone?: 'warn' | 'info' }) {
  if (!items.length) return null
  const styles =
    tone === 'warn'
      ? 'border-amber-900/60 bg-amber-950/30 text-amber-200/90'
      : 'border-neutral-800 bg-neutral-900/40 text-neutral-400'

  return (
    <ul className={`space-y-2 rounded-md border p-3 text-sm ${styles}`}>
      {items.map((item, i) => (
        <li key={i} className="flex gap-2">
          <span aria-hidden className="select-none opacity-60">
            !
          </span>
          <span>{item}</span>
        </li>
      ))}
    </ul>
  )
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-neutral-400">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-neutral-600 border-t-neutral-300" />
      {label}
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-8 text-center text-sm text-neutral-500">{children}</p>
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-red-900 bg-red-950/40 p-3 text-sm text-red-200">
      {message}
    </div>
  )
}

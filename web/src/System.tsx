/**
 * What this machine can do, and whether anything is wrong with it.
 *
 * The same preflight `llmforge doctor` runs. Worth having in the interface because a
 * degraded machine — no fused attention, no compiler — silently produces slower runs
 * rather than errors, and the planner's estimates are built on these measurements.
 */

import { useEffect, useState } from 'react'
import { api, type DoctorReport, type Hardware } from './api'
import { Button, ErrorBox, Field, Panel, Spinner } from './components'

const STATUS_STYLE: Record<string, string> = {
  pass: 'text-emerald-400',
  warn: 'text-amber-400',
  fail: 'text-red-400',
  skip: 'text-neutral-600',
}

const STATUS_LABEL: Record<string, string> = {
  pass: 'ok',
  warn: 'warn',
  fail: 'fail',
  skip: '—',
}

export function System() {
  const [hardware, setHardware] = useState<Hardware | null>(null)
  const [report, setReport] = useState<DoctorReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.hardware().then(setHardware).catch(() => undefined)
  }, [])

  async function check() {
    setBusy(true)
    setError(null)
    try {
      setReport(await api.doctor())
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <Panel title="This machine">
        {hardware ? (
          <dl>
            <Field label="gpu">
              {hardware.n_gpus > 1 ? `${hardware.n_gpus}x ` : ''}
              {hardware.gpu} <span className="text-neutral-500">sm_{hardware.capability}</span>
            </Field>
            <Field label="memory">
              {hardware.memory_gb.toFixed(0)} GB per device
              {hardware.n_gpus > 1 && (
                <span className="text-neutral-500">
                  {' '}
                  · {(hardware.memory_gb * hardware.n_gpus).toFixed(0)} GB total
                </span>
              )}
              {hardware.unified_memory && (
                <span className="text-neutral-500"> · unified with system RAM</span>
              )}
            </Field>
            <Field label="throughput">
              {hardware.bf16_tflops.toFixed(0)} TFLOP/s bf16, measured
            </Field>
            <Field label="bandwidth">{hardware.bandwidth_gbps.toFixed(0)} GB/s, measured</Field>
            <Field label="attention">
              {hardware.flash_attn
                ? 'flash-attn'
                : hardware.flash_sdpa
                  ? 'fused SDPA'
                  : 'unfused — long context will be memory-hungry'}
            </Field>
            <Field label="compiler">
              {hardware.compile_ok ? 'torch.compile available' : 'unavailable — training runs eagerly'}
            </Field>
          </dl>
        ) : (
          <Spinner label="reading hardware" />
        )}
        <p className="mt-3 text-xs text-neutral-500">
          These are measurements, not specifications. Every time estimate the planner
          gives you is derived from them, and they are taken again automatically if the
          machine changes.
        </p>
      </Panel>

      <Panel
        title="Preflight"
        action={
          <Button onClick={check} disabled={busy}>
            {busy ? 'Checking…' : report ? 'Re-check' : 'Run checks'}
          </Button>
        }
      >
        {!report && !busy && !error && (
          <p className="text-sm text-neutral-500">
            Verifies the parts of the stack that break quietly — the CUDA build, bf16
            tensor cores, a fused attention backend, and the compiler.
          </p>
        )}
        {busy && <Spinner label="probing — this benchmarks the GPU" />}
        {error && <ErrorBox message={error} />}

        {report && (
          <div className="space-y-1">
            {report.checks.map((c) => (
              <div key={c.name} className="flex gap-3 py-1 text-sm">
                <span className={`w-10 shrink-0 ${STATUS_STYLE[c.status]}`}>
                  {STATUS_LABEL[c.status]}
                </span>
                <span className="w-36 shrink-0 text-neutral-300">{c.name}</span>
                <span className="min-w-0 flex-1 text-neutral-500">{c.detail}</span>
              </div>
            ))}
            <p className={`pt-2 text-sm ${report.ok ? 'text-emerald-400' : 'text-red-400'}`}>
              {report.ok ? 'Ready to train.' : 'Blocking problems — training will not work.'}
            </p>
          </div>
        )}
      </Panel>
    </div>
  )
}

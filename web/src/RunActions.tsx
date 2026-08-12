/**
 * What you do with a model once it exists: measure whether it changed anything, and
 * take it somewhere else.
 */

import { useEffect, useState } from 'react'
import {
  api,
  type EvalReport,
  type ExportLevels,
  type ExportResult,
  type QuantLevel,
} from './api'
import { Button, ErrorBox, Notes, Panel, Spinner } from './components'

export function Evaluate({ runId, mode }: { runId: string; mode: string }) {
  const [report, setReport] = useState<EvalReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function run() {
    setBusy(true)
    setError(null)
    try {
      setReport(await api.evalRun(runId))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const baselineLabel = mode === 'distill' ? 'teacher' : 'before'

  return (
    <Panel
      title="Did it work?"
      action={
        <Button onClick={run} disabled={busy}>
          {busy ? 'Measuring…' : report ? 'Re-run' : 'Evaluate'}
        </Button>
      }
    >
      {!report && !busy && !error && (
        <p className="text-sm text-neutral-500">
          Scores this model on held-out data from your folder
          {mode === 'finetune' && ', alongside the model it started from'}
          {mode === 'distill' && ', alongside its teacher'}.
        </p>
      )}

      {busy && <Spinner label="loading models and scoring — this takes a minute" />}
      {error && <ErrorBox message={error} />}

      {report && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-end gap-6">
            {report.before_ppl !== null && (
              <Metric label={`perplexity ${baselineLabel}`} value={report.before_ppl.toFixed(2)} />
            )}
            <Metric
              label="perplexity yours"
              value={report.after_ppl !== null ? report.after_ppl.toFixed(2) : '—'}
              strong
            />
            {report.improvement !== null && (
              <Metric
                label="change"
                value={`${report.improvement > 0 ? '−' : '+'}${Math.abs(report.improvement * 100).toFixed(0)}%`}
                tone={
                  report.improvement > 0.05
                    ? 'good'
                    : report.improvement > -0.05
                      ? 'flat'
                      : 'bad'
                }
              />
            )}
            <Metric label="held-out examples" value={report.n_examples.toLocaleString()} />
          </div>

          <Notes items={report.notes} />

          {report.comparisons.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-neutral-300">Generations</h3>
              {report.comparisons.map((c, i) => (
                <div key={i} className="rounded border border-neutral-800 bg-neutral-950 p-3">
                  <p className="mb-2 text-xs text-neutral-500">{c.prompt}</p>
                  {c.before && (
                    <p className="mb-2 border-l-2 border-neutral-700 pl-3 text-sm text-neutral-500">
                      <span className="mr-2 text-xs uppercase tracking-wide">{baselineLabel}</span>
                      {c.before}
                    </p>
                  )}
                  <p className="border-l-2 border-emerald-700 pl-3 text-sm text-neutral-200">
                    <span className="mr-2 text-xs uppercase tracking-wide text-emerald-500">
                      yours
                    </span>
                    {c.after}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Panel>
  )
}

function Metric({
  label,
  value,
  strong,
  tone,
}: {
  label: string
  value: string
  strong?: boolean
  tone?: 'good' | 'flat' | 'bad'
}) {
  const colour =
    tone === 'good'
      ? 'text-emerald-400'
      : tone === 'bad'
        ? 'text-red-400'
        : tone === 'flat'
          ? 'text-amber-400'
          : strong
            ? 'text-neutral-100'
            : 'text-neutral-300'

  return (
    <div>
      <div className="text-xs text-neutral-500">{label}</div>
      <div className={`text-xl tabular-nums ${colour}`}>{value}</div>
    </div>
  )
}

export function Export({ runId, nParams }: { runId: string; nParams?: number }) {
  const [levels, setLevels] = useState<ExportLevels | null>(null)
  const [format, setFormat] = useState<'gguf' | 'safetensors'>('gguf')
  const [quant, setQuant] = useState('')
  const [result, setResult] = useState<ExportResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .exportLevels()
      .then((l) => {
        setLevels(l)
        setQuant(l.defaults[format] ?? '')
      })
      .catch(() => setLevels(null))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (levels) setQuant(levels.defaults[format] ?? '')
  }, [format, levels])

  const options: QuantLevel[] = (levels?.levels ?? []).filter((l) => l.format === format)
  const selected = options.find((l) => l.name === quant)

  async function run() {
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await api.exportRun(runId, format, quant))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel title="Export">
      <div className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="text-sm">
            <span className="mb-1 block text-neutral-400">Format</span>
            <select
              value={format}
              onChange={(e) => setFormat(e.target.value as 'gguf' | 'safetensors')}
              className="w-full rounded border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none focus:border-neutral-500"
            >
              <option value="gguf">GGUF — for Ollama / llama.cpp</option>
              <option value="safetensors">safetensors — for transformers</option>
            </select>
          </label>

          <label className="text-sm">
            <span className="mb-1 block text-neutral-400">Quantization</span>
            <select
              value={quant}
              onChange={(e) => setQuant(e.target.value)}
              className="w-full rounded border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none focus:border-neutral-500"
            >
              {options.map((level) => (
                <option key={level.name} value={level.name} disabled={!level.available}>
                  {level.name}
                  {!level.available ? ' (needs llama.cpp)' : ''}
                </option>
              ))}
            </select>
          </label>
        </div>

        {selected && (
          <p className="text-xs text-neutral-500">
            {selected.summary}
            {nParams ? (
              <>
                {' '}
                Roughly{' '}
                <strong className="text-neutral-400">
                  {((nParams * selected.bits) / 8 / 1e6).toFixed(0)} MB
                </strong>
                .
              </>
            ) : null}
          </p>
        )}

        <div className="flex items-center gap-3">
          <Button variant="primary" onClick={run} disabled={busy || !quant || !selected?.available}>
            {busy ? 'Exporting…' : 'Export'}
          </Button>
          {busy && <Spinner label="converting weights" />}
        </div>

        {error && <ErrorBox message={error} />}

        {result && (
          <div className="rounded-md border border-emerald-900 bg-emerald-950/30 p-3 text-sm">
            <p className="text-emerald-200">
              Exported {result.megabytes.toFixed(1)} MB as {result.quantization}.
            </p>
            <p className="scroll-x mt-1 font-mono text-xs text-neutral-400">{result.path}</p>
          </div>
        )}
      </div>
    </Panel>
  )
}

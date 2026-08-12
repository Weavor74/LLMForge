/**
 * A single run: live loss curves, throughput, samples, and a chat box once it is done.
 *
 * The stream reconstructs state from disk on the server side, so opening this page on
 * a run already in progress replays its whole history and then follows along.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  api,
  compact,
  duration,
  streamRun,
  type MetricRecord,
  type Run,
  type RunDetail,
} from './api'
import { Button, Empty, ErrorBox, Field, Notes, Panel, Progress, Spinner, StatusBadge } from './components'
import { Evaluate, Export } from './RunActions'

export function RunView({ runId, onBack }: { runId: string; onBack: () => void }) {
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [run, setRun] = useState<Run | null>(null)
  const [metrics, setMetrics] = useState<MetricRecord[]>([])
  const [samples, setSamples] = useState<{ step: number; text: string }[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setDetail(null)
    setMetrics([])

    api
      .run(runId)
      .then((d) => {
        if (cancelled) return
        setDetail(d)
        setRun(d)
        setMetrics(d.metrics)
        setSamples(d.samples)
      })
      .catch((e) => !cancelled && setError((e as Error).message))

    // The socket sends the full history first, so replace rather than append —
    // otherwise a reconnect would double every point.
    let seenAny = false
    const close = streamRun(runId, (message) => {
      if (cancelled) return
      if (message.type === 'metrics') {
        setMetrics((current) => {
          if (!seenAny) {
            seenAny = true
            return message.records.length >= current.length ? message.records : current
          }
          return [...current, ...message.records]
        })
      } else if (message.type === 'status' || message.type === 'progress') {
        setRun(message.run)
      } else if (message.type === 'done') {
        setSamples(message.samples)
      }
    })

    return () => {
      cancelled = true
      close()
    }
  }, [runId])

  // A worker derives its plan after the run row exists, so the first fetch can land
  // before there is a plan to show. Pick it up once the run reports a step count.
  const planMissing = !detail?.plan?.mode
  const hasPlanNow = (run?.total_steps ?? 0) > 0
  useEffect(() => {
    if (!planMissing || !hasPlanNow) return
    let cancelled = false
    api
      .run(runId)
      .then((d) => !cancelled && setDetail(d))
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [runId, planMissing, hasPlanNow])

  const current = run ?? detail
  if (error) return <ErrorBox message={error} />
  if (!current) return <Spinner label="loading run" />

  const active = current.status === 'running'

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <button onClick={onBack} className="text-xs text-neutral-500 hover:text-neutral-300">
            ← all runs
          </button>
          <h1 className="truncate font-mono text-lg">{current.id}</h1>
          {current.name && <p className="text-sm text-neutral-400">{current.name}</p>}
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={current.status} alive={current.alive} />
          {active && (
            <Button
              variant="danger"
              onClick={() => api.cancel(current.id).catch((e) => setError(e.message))}
            >
              Stop
            </Button>
          )}
          {(current.status === 'cancelled' || current.status === 'failed') && (
            <Button
              onClick={() =>
                api
                  .resume(current.id)
                  .catch((e) => setError((e as Error).message))
              }
            >
              Resume
            </Button>
          )}
        </div>
      </div>

      {current.error && <ErrorBox message={current.error} />}

      <Panel>
        <div className="mb-3 flex items-baseline justify-between text-sm">
          <span className="text-neutral-400">
            step {current.step.toLocaleString()} / {current.total_steps.toLocaleString()}
          </span>
          <span className="text-neutral-500">
            {compact(current.tokens_seen)} tokens · {duration(current.elapsed_s)}
            {active && metrics.length > 0 && <Eta metrics={metrics} />}
          </span>
        </div>
        <Progress value={current.progress} />

        <dl className="mt-4 grid gap-x-8 sm:grid-cols-2">
          <Field label="best val loss">
            {current.best_val_loss !== null ? current.best_val_loss.toFixed(4) : '—'}
          </Field>
          <Field label="train loss">
            {current.train_loss !== null ? current.train_loss.toFixed(4) : '—'}
          </Field>
          {detail?.plan?.mode && <PlanFields plan={detail.plan} />}
        </dl>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Loss">
          <LossChart metrics={metrics} />
        </Panel>
        <Panel title="Throughput">
          <ThroughputChart metrics={metrics} />
        </Panel>
      </div>

      {detail?.plan?.notes && detail.plan.notes.length > 0 && (
        <Panel title="What to expect">
          <Notes items={detail.plan.notes} />
        </Panel>
      )}

      {samples.length > 0 && (
        <Panel title="Samples">
          <div className="space-y-3">
            {samples.map((s) => (
              <div key={s.step}>
                <div className="mb-1 text-xs text-neutral-500">step {s.step.toLocaleString()}</div>
                <pre className="scroll-x whitespace-pre-wrap rounded border border-neutral-800 bg-neutral-950 p-3 font-mono text-xs text-neutral-300">
                  {s.text.trim()}
                </pre>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {!active && current.status === 'completed' && (
        <>
          <Chat runId={current.id} />
          <Evaluate runId={current.id} mode={current.mode} />
          <Export runId={current.id} nParams={detail?.plan?.n_params ?? detail?.plan?.base_params} />
        </>
      )}
    </div>
  )
}

function Eta({ metrics }: { metrics: MetricRecord[] }) {
  const latest = [...metrics].reverse().find((m) => m.eta_s !== undefined)
  if (!latest?.eta_s) return null
  return <> · {duration(latest.eta_s)} left</>
}

function PlanFields({ plan }: { plan: RunDetail['plan'] }) {
  if (plan.mode === 'finetune') {
    return (
      <>
        <Field label="base">{plan.base_model}</Field>
        <Field label="method">{plan.method}</Field>
      </>
    )
  }
  return (
    <>
      <Field label="size">
        {plan.tier} — {compact(plan.n_params)} params
      </Field>
      {plan.mode === 'distill' ? (
        <Field label="teacher">
          {plan.teacher} — {plan.teacher_label}
        </Field>
      ) : (
        <Field label="context">{plan.seq_len?.toLocaleString()} tokens</Field>
      )}
    </>
  )
}

const AXIS = { stroke: '#525252', fontSize: 11 }
const TOOLTIP = {
  contentStyle: {
    background: '#0a0a0a',
    border: '1px solid #262626',
    borderRadius: 6,
    fontSize: 12,
  },
}

function LossChart({ metrics }: { metrics: MetricRecord[] }) {
  // Training and validation are logged on different cadences; merging by step keeps
  // both series on one axis without inventing points.
  const data = useMemo(() => {
    const byStep = new Map<number, { step: number; train?: number; val?: number }>()
    for (const m of metrics) {
      const entry = byStep.get(m.step) ?? { step: m.step }
      if (m.loss !== undefined) entry.train = m.loss
      if (m.val_loss !== undefined) entry.val = m.val_loss
      byStep.set(m.step, entry)
    }
    return [...byStep.values()].sort((a, b) => a.step - b.step)
  }, [metrics])

  if (!data.length) return <Empty>No metrics yet.</Empty>

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -12 }}>
        <CartesianGrid stroke="#1f1f1f" />
        <XAxis dataKey="step" {...AXIS} />
        <YAxis {...AXIS} domain={['auto', 'auto']} />
        <Tooltip {...TOOLTIP} />
        <Line
          type="monotone"
          dataKey="train"
          stroke="#38bdf8"
          dot={false}
          strokeWidth={1.5}
          name="train"
          connectNulls
        />
        <Line
          type="monotone"
          dataKey="val"
          stroke="#34d399"
          dot={false}
          strokeWidth={2}
          name="val"
          connectNulls
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

function ThroughputChart({ metrics }: { metrics: MetricRecord[] }) {
  const data = useMemo(
    () =>
      metrics
        .filter((m) => m.tokens_per_sec !== undefined)
        .map((m) => ({ step: m.step, rate: m.tokens_per_sec, mem: m.peak_gb })),
    [metrics],
  )

  if (!data.length) return <Empty>No throughput data yet.</Empty>

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -12 }}>
        <CartesianGrid stroke="#1f1f1f" />
        <XAxis dataKey="step" {...AXIS} />
        <YAxis {...AXIS} tickFormatter={(v) => compact(v as number)} />
        <Tooltip {...TOOLTIP} formatter={(v) => compact(v as number)} />
        <Line
          type="monotone"
          dataKey="rate"
          stroke="#a78bfa"
          dot={false}
          strokeWidth={1.5}
          name="tokens/sec"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

function Chat({ runId }: { runId: string }) {
  const [prompt, setPrompt] = useState('')
  const [output, setOutput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  async function send() {
    setBusy(true)
    setError(null)
    try {
      const { text } = await api.chat(runId, prompt)
      setOutput(text)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel title="Try it">
      <div className="space-y-3">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !busy && send()}
            placeholder="Type a prompt…"
            className="min-w-0 flex-1 rounded-md border border-neutral-700 bg-neutral-950 px-3 py-1.5 text-sm outline-none focus:border-neutral-500"
          />
          <Button variant="primary" onClick={send} disabled={busy}>
            {busy ? 'Generating…' : 'Send'}
          </Button>
        </div>
        {busy && <Spinner label="loading the model — first call is slow" />}
        {error && <ErrorBox message={error} />}
        {output && (
          <pre className="scroll-x whitespace-pre-wrap rounded border border-neutral-800 bg-neutral-950 p-3 font-mono text-xs text-neutral-200">
            {output.trim()}
          </pre>
        )}
      </div>
    </Panel>
  )
}

/** App shell: navigation, the runs list, and the hardware banner. */

import { useCallback, useEffect, useState } from 'react'
import { api, compact, duration, type Hardware, type Run } from './api'
import { Empty, Panel, Progress, Spinner, StatusBadge } from './components'
import { Generate } from './Generate'
import { NewModel } from './NewModel'
import { RunView } from './RunView'
import { System } from './System'

type View =
  | { page: 'new' }
  | { page: 'runs' }
  | { page: 'run'; id: string }
  | { page: 'generate' }
  | { page: 'system' }

export default function App() {
  const [view, setView] = useState<View>({ page: 'new' })
  const [hardware, setHardware] = useState<Hardware | null>(null)

  useEffect(() => {
    api.hardware().then(setHardware).catch(() => setHardware(null))
  }, [])

  return (
    <div className="min-h-screen">
      <header className="border-b border-neutral-800 bg-neutral-900/60 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <span className="font-semibold tracking-tight">LLMForge</span>

          <nav className="flex gap-1">
            {(
              [
                ['new', 'New model'],
                ['generate', 'Generate data'],
                ['runs', 'Runs'],
                ['system', 'System'],
              ] as const
            ).map(([page, label]) => (
              <button
                key={page}
                onClick={() => setView({ page })}
                className={`rounded px-3 py-1 text-sm transition-colors ${
                  view.page === page || (page === 'runs' && view.page === 'run')
                    ? 'bg-neutral-800 text-neutral-100'
                    : 'text-neutral-400 hover:text-neutral-200'
                }`}
              >
                {label}
              </button>
            ))}
          </nav>

          {hardware && (
            <span className="ml-auto font-mono text-xs text-neutral-500">
              {hardware.gpu} · {hardware.memory_gb.toFixed(0)} GB ·{' '}
              {hardware.bf16_tflops.toFixed(0)} TFLOP/s · {hardware.bandwidth_gbps.toFixed(0)} GB/s
            </span>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-6">
        {view.page === 'new' && (
          <NewModel onStarted={(id) => setView({ page: 'run', id })} />
        )}
        {view.page === 'runs' && <RunList onOpen={(id) => setView({ page: 'run', id })} />}
        {view.page === 'run' && (
          <RunView runId={view.id} onBack={() => setView({ page: 'runs' })} />
        )}
        {view.page === 'generate' && (
          <Generate onStarted={(id) => setView({ page: 'run', id })} />
        )}
        {view.page === 'system' && <System />}
      </main>
    </div>
  )
}

function RunList({ onOpen }: { onOpen: (id: string) => void }) {
  const [runs, setRuns] = useState<Run[] | null>(null)

  const load = useCallback(() => {
    api.runs().then(setRuns).catch(() => setRuns([]))
  }, [])

  useEffect(() => {
    load()
    // Cheap enough to poll: the list is a summary query against SQLite.
    const timer = setInterval(load, 3000)
    return () => clearInterval(timer)
  }, [load])

  if (!runs) return <Spinner label="loading runs" />
  if (!runs.length) return <Empty>No runs yet. Start one from “New model”.</Empty>

  return (
    <Panel title={`${runs.length} run${runs.length === 1 ? '' : 's'}`}>
      <div className="space-y-1">
        {runs.map((run) => (
          <button
            key={run.id}
            onClick={() => onOpen(run.id)}
            className="block w-full rounded-md px-3 py-2.5 text-left transition-colors hover:bg-neutral-800/60"
          >
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <span className="font-mono text-sm text-neutral-200">{run.id}</span>
              <StatusBadge status={run.status} alive={run.alive} />
              {run.name && <span className="text-xs text-neutral-500">{run.name}</span>}
              <span className="ml-auto text-xs text-neutral-500">
                {run.mode === 'finetune'
                  ? `${run.base_model ?? ''} · ${run.method ?? ''}`
                  : run.mode === 'distill'
                    ? `distil ${run.base_model ?? ''} → ${compact(run.n_params)}`
                    : `${run.tier ?? ''} · ${compact(run.n_params)} params`}
              </span>
            </div>

            <div className="mt-2 flex items-center gap-3">
              <div className="flex-1">
                <Progress value={run.progress} />
              </div>
              <span className="w-40 shrink-0 text-right text-xs text-neutral-500">
                {run.best_val_loss !== null ? `val ${run.best_val_loss.toFixed(3)} · ` : ''}
                {compact(run.tokens_seen)} tok · {duration(run.elapsed_s)}
              </span>
            </div>
          </button>
        ))}
      </div>
    </Panel>
  )
}

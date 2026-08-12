/**
 * The primary flow: choose a folder, review the derived plan, start.
 *
 * Nothing here decides anything. The backend analyses the corpus and derives the
 * plan; this page renders it, lets the user override, and sends back the spec that
 * produced it. That is what keeps a clicked run identical to a typed one.
 */

import { useState } from 'react'
import { api, compact, duration, type Proposal } from './api'
import { Button, ErrorBox, Field, Notes, Panel, Spinner } from './components'
import { FolderPicker, type PickerMode } from './FolderPicker'

type Mode = 'scratch' | 'finetune' | 'distill'

const MODES: readonly (readonly [Mode, string])[] = [
  ['scratch', 'Train from scratch'],
  ['finetune', 'Fine-tune a model'],
  ['distill', 'Distil a model'],
] as const

const TIERS = ['auto', 'nano', 'micro', 'small', 'medium', 'large']
const METHODS = ['auto', 'full', 'lora', 'qlora']

export function NewModel({ onStarted }: { onStarted: (runId: string) => void }) {
  const [folder, setFolder] = useState('')
  const [model, setModel] = useState('')
  const [mode, setMode] = useState<Mode>('scratch')
  const [tier, setTier] = useState('auto')
  const [method, setMethod] = useState('auto')
  const [name, setName] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  // Which field the picker is currently filling, if any.
  const [picking, setPicking] = useState<null | { field: 'folder' | 'model'; mode: PickerMode }>(
    null,
  )

  const [proposal, setProposal] = useState<Proposal | null>(null)
  const [analysing, setAnalysing] = useState(false)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const finetuning = mode === 'finetune'
  const distilling = mode === 'distill'
  const needsModel = finetuning || distilling
  const canAnalyse = folder.length > 0 && (!needsModel || model.length > 0)

  async function analyse() {
    setAnalysing(true)
    setError(null)
    setProposal(null)
    try {
      setProposal(
        await api.analyze({
          folder,
          base: finetuning ? model : null,
          teacher: distilling ? model : null,
          tier: !finetuning && tier !== 'auto' ? tier : null,
          method: finetuning && method !== 'auto' ? method : null,
        }),
      )
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setAnalysing(false)
    }
  }

  async function start() {
    if (!proposal) return
    setStarting(true)
    setError(null)
    try {
      const { run_id } = await api.start(proposal.spec, name || undefined)
      onStarted(run_id)
    } catch (e) {
      setError((e as Error).message)
      setStarting(false)
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      {picking && (
        <FolderPicker
          value={picking.field === 'folder' ? folder : model}
          mode={picking.mode}
          onChange={picking.field === 'folder' ? setFolder : setModel}
          onClose={() => setPicking(null)}
        />
      )}

      <Panel title="Corpus">
        <div className="space-y-4">
          <div>
            <label className="mb-1.5 block text-sm text-neutral-400">Folder</label>
            <div className="flex gap-2">
              <input
                value={folder}
                onChange={(e) => setFolder(e.target.value)}
                placeholder="/path/to/your/data"
                className="min-w-0 flex-1 rounded-md border border-neutral-700 bg-neutral-950 px-3 py-1.5 font-mono text-sm outline-none focus:border-neutral-500"
              />
              <Button onClick={() => setPicking({ field: 'folder', mode: 'corpus' })}>
                Browse…
              </Button>
            </div>
          </div>

          <div>
            <label className="mb-1.5 block text-sm text-neutral-400">What to build</label>
            <div className="flex gap-2">
              {MODES.map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => {
                    setMode(key)
                    setProposal(null)
                  }}
                  className={`flex-1 rounded-md border px-3 py-2 text-sm transition-colors ${
                    mode === key
                      ? 'border-emerald-700 bg-emerald-950/50 text-emerald-200'
                      : 'border-neutral-700 bg-neutral-900 text-neutral-400 hover:border-neutral-600'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {needsModel && (
            <div>
              <label className="mb-1.5 block text-sm text-neutral-400">
                {distilling ? 'Teacher model' : 'Base model'}
              </label>
              <div className="flex gap-2">
                <input
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder="Qwen/Qwen3-8B  — or a local folder"
                  className="min-w-0 flex-1 rounded-md border border-neutral-700 bg-neutral-950 px-3 py-1.5 font-mono text-sm outline-none focus:border-neutral-500"
                />
                <Button onClick={() => setPicking({ field: 'model', mode: 'model' })}>
                  Browse…
                </Button>
              </div>
              <p className="mt-1 text-xs text-neutral-500">
                {distilling
                  ? 'The student will be built to match this model and learn from its predictions.'
                  : 'A Hugging Face model id, or a folder on this machine.'}
              </p>
            </div>
          )}

          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs text-neutral-500 hover:text-neutral-300"
          >
            {showAdvanced ? '− ' : '+ '}Advanced
          </button>

          {showAdvanced && (
            <div className="grid gap-3 rounded-md border border-neutral-800 bg-neutral-950/50 p-3 sm:grid-cols-2">
              <label className="text-sm">
                <span className="mb-1 block text-neutral-400">Run name</span>
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="optional label"
                  className="w-full rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-sm outline-none focus:border-neutral-500"
                />
              </label>
              {finetuning ? (
                <label className="text-sm">
                  <span className="mb-1 block text-neutral-400">Method</span>
                  <select
                    value={method}
                    onChange={(e) => setMethod(e.target.value)}
                    className="w-full rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-sm outline-none focus:border-neutral-500"
                  >
                    {METHODS.map((m) => (
                      <option key={m}>{m}</option>
                    ))}
                  </select>
                </label>
              ) : (
                <label className="text-sm">
                  <span className="mb-1 block text-neutral-400">Model size</span>
                  <select
                    value={tier}
                    onChange={(e) => setTier(e.target.value)}
                    className="w-full rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-sm outline-none focus:border-neutral-500"
                  >
                    {TIERS.map((t) => (
                      <option key={t}>{t}</option>
                    ))}
                  </select>
                </label>
              )}
            </div>
          )}

          <div className="flex items-center gap-3">
            <Button variant="primary" onClick={analyse} disabled={!canAnalyse || analysing}>
              {analysing ? 'Analysing…' : 'Analyse'}
            </Button>
            {analysing && <Spinner label="reading the corpus — this can take a while" />}
          </div>
        </div>
      </Panel>

      {error && <ErrorBox message={error} />}

      {proposal && <PlanReview proposal={proposal} onStart={start} starting={starting} />}
    </div>
  )
}

function PlanReview({
  proposal,
  onStart,
  starting,
}: {
  proposal: Proposal
  onStart: () => void
  starting: boolean
}) {
  const { analysis, plan, hardware } = proposal
  const tokens = analysis.exact_tokens ?? analysis.est_tokens
  const finetune = plan.mode === 'finetune'
  const distill = plan.mode === 'distill'

  return (
    <>
      <Panel title="What is in the folder">
        <dl>
          <Field label="documents">{analysis.n_documents.toLocaleString()}</Field>
          <Field label="tokens">
            {tokens.toLocaleString()}
            {analysis.exact_tokens === null && (
              <span className="text-neutral-500"> (estimated)</span>
            )}
          </Field>
          <Field label="detected as">
            {analysis.kind === 'instruction' ? 'instruction / chat data' : 'raw text'}
          </Field>
          <Field label="files">
            {analysis.n_files_used.toLocaleString()} used,{' '}
            {analysis.n_files_skipped.toLocaleString()} skipped
          </Field>
          <Field label="dropped">
            {analysis.n_dropped_duplicate.toLocaleString()} duplicate,{' '}
            {analysis.n_dropped_quality.toLocaleString()} low quality
          </Field>
        </dl>

        <div className="scroll-x mt-3">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-neutral-800 text-left text-xs text-neutral-500">
                <th className="py-1 pr-4 font-medium">format</th>
                <th className="py-1 pr-4 text-right font-medium">files</th>
                <th className="py-1 pr-4 text-right font-medium">docs</th>
                <th className="py-1 text-right font-medium">chars</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(analysis.by_extension)
                .sort((a, b) => b[1].chars - a[1].chars)
                .map(([ext, stat]) => (
                  <tr key={ext} className="border-b border-neutral-900">
                    <td className="py-1 pr-4 font-mono text-cyan-400">{ext}</td>
                    <td className="py-1 pr-4 text-right">{stat.files.toLocaleString()}</td>
                    <td className="py-1 pr-4 text-right">{stat.documents.toLocaleString()}</td>
                    <td className="py-1 text-right">{compact(stat.chars)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>

        <div className="mt-3">
          <Notes items={analysis.warnings} />
        </div>
      </Panel>

      <Panel title="The plan">
        <dl>
          {finetune ? (
            <>
              <Field label="base model">
                <span className="font-mono">{plan.base_model}</span> — {plan.base_label} (
                {plan.architecture})
              </Field>
              <Field label="method">
                <strong className="text-emerald-300">{plan.method}</strong>
                {plan.method === 'full'
                  ? ' — every weight updated'
                  : plan.method === 'lora'
                    ? ' — adapter trained, base frozen'
                    : ' — base quantized to 4-bit, then frozen'}
              </Field>
              <Field label="trainable">
                {compact(plan.trainable_params)} of {compact(plan.base_params)} (
                {((plan.trainable_params! / plan.base_params!) * 100).toFixed(2)}%)
              </Field>
              <Field label="data">
                {plan.n_examples?.toLocaleString()} examples,{' '}
                {plan.supervised ? 'supervised conversations' : 'raw text (continued pretraining)'}
              </Field>
              <Field label="training">
                {plan.total_steps.toLocaleString()} steps over {plan.epochs?.toFixed(1)} passes
              </Field>
            </>
          ) : distill ? (
            <>
              <Field label="teacher">
                <span className="font-mono">{plan.teacher}</span> — {plan.teacher_label}
                {plan.teacher_load_4bit && (
                  <span className="ml-2 rounded border border-amber-900 bg-amber-950 px-1.5 py-0.5 text-xs text-amber-300">
                    loaded 4-bit
                  </span>
                )}
              </Field>
              <Field label="student">
                <strong className="text-emerald-300">{plan.tier}</strong> —{' '}
                {compact(plan.n_params)} parameters (
                {((plan.n_params! / plan.teacher_params!) * 100).toFixed(0)}% of the teacher)
              </Field>
              <Field label="architecture">
                {plan.n_layer} layers, {plan.n_head} heads ({plan.n_kv_head} kv), d_model{' '}
                {plan.d_model}, ffn {plan.d_ff}
              </Field>
              <Field label="vocabulary">
                {plan.vocab_size?.toLocaleString()}{' '}
                <span className="text-neutral-500">(the teacher's)</span>
              </Field>
              <Field label="objective">
                {((plan.alpha ?? 0) * 100).toFixed(0)}% teacher KL at T={plan.temperature},{' '}
                {((1 - (plan.alpha ?? 0)) * 100).toFixed(0)}% true-token cross-entropy
              </Field>
              <Field label="training">
                {plan.total_steps.toLocaleString()} steps, {compact(plan.total_tokens)} tokens (
                {plan.epochs?.toFixed(1)} passes)
              </Field>
            </>
          ) : (
            <>
              <Field label="size">
                <strong className="text-emerald-300">{plan.tier}</strong> —{' '}
                {compact(plan.n_params)} parameters
              </Field>
              <Field label="architecture">
                {plan.n_layer} layers, {plan.n_head} heads ({plan.n_kv_head} kv), d_model{' '}
                {plan.d_model}, ffn {plan.d_ff}
              </Field>
              <Field label="vocabulary">{plan.vocab_size?.toLocaleString()}</Field>
              <Field label="training">
                {plan.total_steps.toLocaleString()} steps, {compact(plan.total_tokens)} tokens (
                {plan.epochs?.toFixed(1)} passes)
              </Field>
            </>
          )}
          <Field label="context">{plan.seq_len.toLocaleString()} tokens</Field>
          <Field label="batch">
            {plan.micro_batch} × {plan.grad_accum} accum = {compact(plan.tokens_per_step)}{' '}
            tokens/step
          </Field>
          <Field label="learning rate">
            {plan.lr.toExponential(1)} → {plan.min_lr.toExponential(1)} cosine,{' '}
            {plan.warmup_steps} warmup
          </Field>
          <Field label="memory">
            ~{plan.estimated_memory_gb.toFixed(1)} GB of {hardware.memory_gb.toFixed(0)} GB
          </Field>
          <Field label="estimated time">
            <strong>{duration(plan.estimated_hours * 3600)}</strong>
            <span className="text-neutral-500"> — refined once training starts</span>
          </Field>
        </dl>

        <div className="mt-3">
          <Notes items={plan.notes} />
        </div>

        <div className="mt-4 flex justify-end">
          <Button variant="primary" onClick={onStart} disabled={starting}>
            {starting ? 'Starting…' : 'Start training'}
          </Button>
        </div>
      </Panel>
    </>
  )
}

/**
 * Every knob the CLI exposes, surfaced where it applies.
 *
 * The premise of the planner is that defaults are derived, so nothing here is
 * required — each field left blank means "you decide". What the panel adds is the
 * ability to disagree with a derived choice without dropping to a terminal.
 */

import type { ReactNode } from 'react'

export interface AdvancedConfig {
  name: string
  tier: string
  vocabSize: string
  method: string
  epochs: string
  loraRank: string
  temperature: string
  alpha: string
  seqLen: string
  seed: string
  force: boolean
}

export const EMPTY_ADVANCED: AdvancedConfig = {
  name: '',
  tier: 'auto',
  vocabSize: '',
  method: 'auto',
  epochs: '',
  loraRank: '',
  temperature: '',
  alpha: '',
  seqLen: '',
  seed: '',
  force: false,
}

const TIERS = ['auto', 'nano', 'micro', 'small', 'medium', 'large', 'xl', 'xxl', '8b', '12b', '20b', '40b', '60b', '80b']
const METHODS = ['auto', 'full', 'lora', 'qlora']

/** Turn the form's strings into the JSON the API expects, dropping anything unset. */
export function toRequest(cfg: AdvancedConfig, mode: 'scratch' | 'finetune' | 'distill') {
  const num = (v: string) => (v.trim() === '' ? null : Number(v))
  const scratchish = mode !== 'finetune'

  return {
    tier: scratchish && cfg.tier !== 'auto' ? cfg.tier : null,
    vocab_size: mode === 'scratch' ? num(cfg.vocabSize) : null,
    method: mode === 'finetune' && cfg.method !== 'auto' ? cfg.method : null,
    epochs: mode === 'finetune' ? num(cfg.epochs) : null,
    lora_rank: mode === 'finetune' ? num(cfg.loraRank) : null,
    temperature: mode === 'distill' ? num(cfg.temperature) : null,
    alpha: mode === 'distill' ? num(cfg.alpha) : null,
    seq_len: num(cfg.seqLen),
    seed: num(cfg.seed) ?? 1337,
    force: cfg.force,
  }
}

function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <label className="text-sm">
      <span className="mb-1 block text-neutral-400">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-neutral-600">{hint}</span>}
    </label>
  )
}

const inputClass =
  'w-full rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-sm outline-none focus:border-neutral-500'

export function Advanced({
  cfg,
  onChange,
  mode,
}: {
  cfg: AdvancedConfig
  onChange: (next: AdvancedConfig) => void
  mode: 'scratch' | 'finetune' | 'distill'
}) {
  const set = <K extends keyof AdvancedConfig>(key: K, value: AdvancedConfig[K]) =>
    onChange({ ...cfg, [key]: value })

  return (
    <div className="space-y-4 rounded-md border border-neutral-800 bg-neutral-950/50 p-3">
      <p className="text-xs text-neutral-500">
        Leave a field blank to let the planner decide. It derives every default from your
        corpus and this machine.
      </p>

      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Model name" hint="Used for exported filenames.">
          <input
            value={cfg.name}
            onChange={(e) => set('name', e.target.value)}
            placeholder="e.g. support-assistant-v1"
            className={inputClass}
          />
        </Field>

        <Field label="Seed" hint="Same seed, same batch order.">
          <input
            value={cfg.seed}
            onChange={(e) => set('seed', e.target.value)}
            placeholder="1337"
            inputMode="numeric"
            className={inputClass}
          />
        </Field>

        {mode !== 'finetune' && (
          <Field
            label={mode === 'distill' ? 'Student size' : 'Model size'}
            hint="Derived from how many tokens your corpus has."
          >
            <select
              value={cfg.tier}
              onChange={(e) => set('tier', e.target.value)}
              className={inputClass}
            >
              {TIERS.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </Field>
        )}

        {mode === 'scratch' && (
          <Field label="Vocabulary size" hint="Larger fits the corpus better, costs parameters.">
            <input
              value={cfg.vocabSize}
              onChange={(e) => set('vocabSize', e.target.value)}
              placeholder="derived"
              inputMode="numeric"
              className={inputClass}
            />
          </Field>
        )}

        {mode === 'finetune' && (
          <>
            <Field label="Method" hint="Chosen against the memory budget.">
              <select
                value={cfg.method}
                onChange={(e) => set('method', e.target.value)}
                className={inputClass}
              >
                {METHODS.map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
            </Field>

            <Field label="Epochs" hint="Extended automatically on small datasets.">
              <input
                value={cfg.epochs}
                onChange={(e) => set('epochs', e.target.value)}
                placeholder="derived"
                inputMode="decimal"
                className={inputClass}
              />
            </Field>

            <Field label="LoRA rank" hint="Higher adapts more, and costs more memory.">
              <input
                value={cfg.loraRank}
                onChange={(e) => set('loraRank', e.target.value)}
                placeholder="16"
                inputMode="numeric"
                className={inputClass}
              />
            </Field>
          </>
        )}

        {mode === 'distill' && (
          <>
            <Field label="Temperature" hint="Softens both distributions. Above 1 exposes the teacher's second choices.">
              <input
                value={cfg.temperature}
                onChange={(e) => set('temperature', e.target.value)}
                placeholder="2.0"
                inputMode="decimal"
                className={inputClass}
              />
            </Field>

            <Field label="Alpha" hint="Share of the loss taken from the teacher rather than the true token.">
              <input
                value={cfg.alpha}
                onChange={(e) => set('alpha', e.target.value)}
                placeholder="0.7"
                inputMode="decimal"
                className={inputClass}
              />
            </Field>
          </>
        )}

        <Field label="Context length" hint="Tokens per sequence. Memory scales with it.">
          <input
            value={cfg.seqLen}
            onChange={(e) => set('seqLen', e.target.value)}
            placeholder="derived"
            inputMode="numeric"
            className={inputClass}
          />
        </Field>
      </div>

      <label className="flex items-center gap-2 text-sm text-neutral-400">
        <input
          type="checkbox"
          checked={cfg.force}
          onChange={(e) => set('force', e.target.checked)}
          className="accent-emerald-600"
        />
        Re-read the corpus from scratch
        <span className="text-xs text-neutral-600">(ignore the cache)</span>
      </label>
    </div>
  )
}

/**
 * The API surface, typed.
 *
 * Mirrors llmforge/api/main.py. Everything the GUI can do is something the CLI can
 * also do — the GUI never computes a plan itself, it only displays one the backend
 * derived and sends back the spec that produced it.
 */

export interface CorpusAnalysis {
  root: string
  content_hash: string
  kind: 'raw' | 'instruction'
  chat_fraction: number
  n_files_scanned: number
  n_files_used: number
  n_files_skipped: number
  n_documents: number
  n_chars: number
  n_dropped_quality: number
  n_dropped_duplicate: number
  est_tokens: number
  exact_tokens: number | null
  by_extension: Record<string, { files: number; documents: number; chars: number }>
  warnings: string[]
}

export interface Plan {
  mode: 'pretrain' | 'finetune' | 'distill'
  seq_len: number
  micro_batch: number
  grad_accum: number
  tokens_per_step: number
  total_steps: number
  lr: number
  min_lr: number
  warmup_steps: number
  estimated_hours: number
  estimated_memory_gb: number
  notes: string[]
  // pretrain
  tier?: string
  n_params?: number
  n_layer?: number
  n_head?: number
  n_kv_head?: number
  d_model?: number
  d_ff?: number
  vocab_size?: number
  total_tokens?: number
  epochs?: number
  // finetune
  base_model?: string
  base_params?: number
  base_label?: string
  architecture?: string
  method?: 'full' | 'lora' | 'qlora'
  trainable_params?: number
  supervised?: boolean
  n_examples?: number
  // distill
  teacher?: string
  teacher_params?: number
  teacher_label?: string
  teacher_load_4bit?: boolean
  temperature?: number
  alpha?: number
}

export interface Hardware {
  gpu: string
  capability: string
  bf16_tflops: number
  bandwidth_gbps: number
  memory_gb: number
  compile_ok: boolean
  flash_sdpa: boolean
  n_gpus: number
  unified_memory: boolean
  flash_attn: boolean
  gpus: string[]
}

export interface RunSpec {
  folder: string
  base: string | null
  teacher: string | null
  tier: string | null
  vocab_size: number | null
  method: string | null
  epochs: number | null
  lora_rank: number | null
  temperature: number | null
  alpha: number | null
  seq_len: number | null
  seed: number
  name: string | null
}

export interface HelpPage {
  slug: string
  title: string
}

export interface DoctorCheck {
  name: string
  status: 'pass' | 'warn' | 'fail' | 'skip'
  detail: string
}

export interface DoctorReport {
  checks: DoctorCheck[]
  environment: Record<string, unknown>
  ok: boolean
}

export interface Proposal {
  analysis: CorpusAnalysis
  plan: Plan
  hardware: Hardware
  spec: RunSpec
  base_model: Record<string, unknown> | null
}

export type RunStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface Run {
  id: string
  name: string | null
  mode: 'pretrain' | 'finetune' | 'distill' | 'generate'
  status: RunStatus
  created_at: string
  updated_at: string
  step: number
  total_steps: number
  progress: number
  train_loss: number | null
  val_loss: number | null
  best_val_loss: number | null
  tokens_seen: number
  elapsed_s: number
  error: string | null
  corpus_root: string | null
  base_model: string | null
  alive: boolean
  tier?: string
  method?: string
  n_params?: number
  output_dir?: string | null
}

export interface MetricRecord {
  step: number
  loss?: number
  lr?: number
  grad_norm?: number
  tokens?: number
  tokens_per_sec?: number
  elapsed_s?: number
  eta_s?: number
  peak_gb?: number
  val_loss?: number
  val_ppl?: number
}

export interface RunDetail extends Run {
  plan: Plan
  metrics: MetricRecord[]
  samples: { step: number; text: string }[]
  log: string
}

export interface BrowseEntry {
  name: string
  path: string
  is_dir: boolean
  n_files: number | null
  is_model: boolean
}

export interface BrowseResponse {
  path: string
  parent: string | null
  entries: BrowseEntry[]
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'content-type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    // FastAPI puts the message in `detail`; fall back to the status line.
    let message = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body?.detail) message = typeof body.detail === 'string' ? body.detail : message
    } catch {
      /* body was not json */
    }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ ok: boolean; version: string; workspace: string }>('/api/health'),
  hardware: () => request<Hardware>('/api/hardware'),

  browse: (path?: string, mode: 'corpus' | 'model' = 'corpus') =>
    request<BrowseResponse>(
      `/api/browse?mode=${mode}` + (path ? `&path=${encodeURIComponent(path)}` : ''),
    ),

  analyze: (body: Record<string, unknown>) =>
    request<Proposal>('/api/analyze', { method: 'POST', body: JSON.stringify(body) }),

  runs: () => request<Run[]>('/api/runs'),
  run: (id: string) => request<RunDetail>(`/api/runs/${id}`),

  start: (spec: RunSpec, name?: string) =>
    request<{ run_id: string }>('/api/runs', {
      method: 'POST',
      body: JSON.stringify({ spec, name: name || null }),
    }),

  cancel: (id: string) => request<{ ok: boolean }>(`/api/runs/${id}/cancel`, { method: 'POST' }),
  resume: (id: string) => request<{ ok: boolean }>(`/api/runs/${id}/resume`, { method: 'POST' }),

  chat: (id: string, prompt: string, maxTokens = 200, temperature = 0.8) =>
    request<{ text: string }>(`/api/runs/${id}/chat`, {
      method: 'POST',
      body: JSON.stringify({ prompt, max_tokens: maxTokens, temperature }),
    }),

  exportLevels: () => request<ExportLevels>('/api/export/levels'),

  exportRun: (id: string, format: string, quantization: string, name?: string) =>
    request<ExportResult>(`/api/runs/${id}/export`, {
      method: 'POST',
      body: JSON.stringify({
        format,
        quantization,
        checkpoint: 'best',
        name: name || null,
      }),
    }),

  rename: (id: string, name: string) =>
    request<{ ok: boolean; slug: string }>(`/api/runs/${id}/rename`, {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),

  doctor: () => request<DoctorReport>('/api/doctor'),

  helpPages: () => request<HelpPage[]>('/api/help'),
  helpPage: (slug: string) =>
    request<{ slug: string; title: string; markdown: string }>(`/api/help/${slug}`),

  generate: (body: {
    teacher: string
    source: string
    name?: string
    samples_per_prompt?: number
    max_new_tokens?: number
    temperature?: number
    batch_size?: number
    limit?: number
    system?: string
  }) => request<{ run_id: string }>('/api/generate', {
    method: 'POST',
    body: JSON.stringify(body),
  }),

  evalRun: (id: string) =>
    request<EvalReport>(`/api/runs/${id}/eval`, {
      method: 'POST',
      body: JSON.stringify({ examples: 64, prompts: 5, checkpoint: 'best' }),
    }),
}

export interface QuantLevel {
  name: string
  format: 'gguf' | 'safetensors'
  bits: number
  summary: string
  available: boolean
  needs_llama_cpp: boolean
}

export interface ExportLevels {
  formats: ('gguf' | 'safetensors')[]
  defaults: Record<string, string>
  levels: QuantLevel[]
}

export interface ExportResult {
  path: string
  format: string
  quantization: string
  megabytes: number
}

export interface EvalReport {
  run_id: string
  mode: string
  before_ppl: number | null
  after_ppl: number | null
  improvement: number | null
  n_examples: number
  comparisons: { prompt: string; before: string; after: string }[]
  notes: string[]
}

export type StreamMessage =
  | { type: 'metrics'; records: MetricRecord[] }
  | { type: 'status'; run: Run }
  | { type: 'progress'; run: Run }
  | { type: 'done'; samples: { step: number; text: string }[] }

export function streamRun(id: string, onMessage: (m: StreamMessage) => void): () => void {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const socket = new WebSocket(`${protocol}//${location.host}/api/runs/${id}/stream`)
  socket.onmessage = (event) => onMessage(JSON.parse(event.data) as StreamMessage)
  return () => socket.close()
}

// --- formatting ------------------------------------------------------------

export function compact(n: number | null | undefined): string {
  if (n === null || n === undefined) return '—'
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return String(Math.round(n))
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '—'
  if (seconds < 90) return `${Math.round(seconds)}s`
  if (seconds < 5400) return `${Math.round(seconds / 60)}m`
  if (seconds < 172800) return `${(seconds / 3600).toFixed(1)}h`
  return `${(seconds / 86400).toFixed(1)}d`
}

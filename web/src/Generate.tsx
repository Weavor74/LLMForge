/**
 * Have a teacher write a training corpus.
 *
 * The alternative to scoring a teacher on every token of every epoch: it answers a set
 * of prompts once, and the result is a folder the New Model page trains on. Runs as a
 * job like training does, because against a large teacher it takes hours.
 */

import { useState } from 'react'
import { api } from './api'
import { Button, ErrorBox, Panel } from './components'
import { FolderPicker } from './FolderPicker'

const inputClass =
  'w-full rounded border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none focus:border-neutral-500'

export function Generate({ onStarted }: { onStarted: (runId: string) => void }) {
  const [teacher, setTeacher] = useState('')
  const [source, setSource] = useState('')
  const [name, setName] = useState('')
  const [samples, setSamples] = useState('1')
  const [maxTokens, setMaxTokens] = useState('512')
  const [temperature, setTemperature] = useState('0.8')
  const [batch, setBatch] = useState('8')
  const [limit, setLimit] = useState('')
  const [system, setSystem] = useState('')
  const [picking, setPicking] = useState<null | 'teacher' | 'source'>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const ready = teacher.trim().length > 0 && source.trim().length > 0

  async function start() {
    setBusy(true)
    setError(null)
    try {
      const num = (v: string) => (v.trim() === '' ? undefined : Number(v))
      const { run_id } = await api.generate({
        teacher: teacher.trim(),
        source: source.trim(),
        name: name.trim() || undefined,
        samples_per_prompt: num(samples),
        max_new_tokens: num(maxTokens),
        temperature: num(temperature),
        batch_size: num(batch),
        limit: num(limit),
        system: system.trim() || undefined,
      })
      onStarted(run_id)
    } catch (e) {
      setError((e as Error).message)
      setBusy(false)
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      {picking && (
        <FolderPicker
          value={picking === 'teacher' ? teacher : source}
          mode={picking === 'teacher' ? 'model' : 'corpus'}
          onChange={picking === 'teacher' ? setTeacher : setSource}
          onClose={() => setPicking(null)}
        />
      )}

      <Panel title="Teacher writes the data">
        <div className="space-y-4">
          <p className="text-sm text-neutral-500">
            The teacher answers each prompt once, producing a folder you can train a
            smaller model on. Unlike scoring it on every token of every epoch, this pays
            for the teacher a single time — and the student keeps its own tokenizer.
          </p>

          <div>
            <label className="mb-1.5 block text-sm text-neutral-400">Teacher</label>
            <div className="flex gap-2">
              <input
                value={teacher}
                onChange={(e) => setTeacher(e.target.value)}
                placeholder="a run id, a Hugging Face id, or a local folder"
                className={`min-w-0 flex-1 font-mono ${inputClass}`}
              />
              <Button onClick={() => setPicking('teacher')}>Browse…</Button>
            </div>
            <p className="mt-1 text-xs text-neutral-600">
              A fine-tuned run works directly — its adapter never has to be merged first.
            </p>
          </div>

          <div>
            <label className="mb-1.5 block text-sm text-neutral-400">Prompts</label>
            <div className="flex gap-2">
              <input
                value={source}
                onChange={(e) => setSource(e.target.value)}
                placeholder="a corpus folder, or a text file with one prompt per line"
                className={`min-w-0 flex-1 font-mono ${inputClass}`}
              />
              <Button onClick={() => setPicking('source')}>Browse…</Button>
            </div>
            <p className="mt-1 text-xs text-neutral-600">
              From a folder, the user turns of its conversations become the prompts.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            <label className="text-sm">
              <span className="mb-1 block text-neutral-400">Name</span>
              <input value={name} onChange={(e) => setName(e.target.value)}
                placeholder="optional" className={inputClass} />
            </label>
            <label className="text-sm">
              <span className="mb-1 block text-neutral-400">Answers per prompt</span>
              <input value={samples} onChange={(e) => setSamples(e.target.value)}
                inputMode="numeric" className={inputClass} />
            </label>
            <label className="text-sm">
              <span className="mb-1 block text-neutral-400">Prompt limit</span>
              <input value={limit} onChange={(e) => setLimit(e.target.value)}
                placeholder="all" inputMode="numeric" className={inputClass} />
            </label>
            <label className="text-sm">
              <span className="mb-1 block text-neutral-400">Max tokens</span>
              <input value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)}
                inputMode="numeric" className={inputClass} />
            </label>
            <label className="text-sm">
              <span className="mb-1 block text-neutral-400">Temperature</span>
              <input value={temperature} onChange={(e) => setTemperature(e.target.value)}
                inputMode="decimal" className={inputClass} />
            </label>
            <label className="text-sm">
              <span className="mb-1 block text-neutral-400">Batch size</span>
              <input value={batch} onChange={(e) => setBatch(e.target.value)}
                inputMode="numeric" className={inputClass} />
            </label>
          </div>

          <label className="block text-sm">
            <span className="mb-1 block text-neutral-400">System prompt</span>
            <textarea
              value={system}
              onChange={(e) => setSystem(e.target.value)}
              rows={2}
              placeholder="optional — shapes how the teacher answers"
              className={inputClass}
            />
          </label>

          {error && <ErrorBox message={error} />}

          <div className="flex justify-end">
            <Button variant="primary" onClick={start} disabled={!ready || busy}>
              {busy ? 'Starting…' : 'Start generating'}
            </Button>
          </div>
        </div>
      </Panel>
    </div>
  )
}

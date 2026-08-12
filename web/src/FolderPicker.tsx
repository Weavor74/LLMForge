/**
 * Server-side folder picker.
 *
 * The corpus and any local models live on the machine that trains, which may not be
 * the machine running the browser, so this browses the server's filesystem rather
 * than using a file input.
 *
 * The same component picks corpora and model directories; `mode` only changes what is
 * highlighted, since in both cases the answer is a path.
 */

import { useEffect, useState } from 'react'
import { api, type BrowseResponse } from './api'
import { Button, ErrorBox, Spinner } from './components'

export type PickerMode = 'corpus' | 'model'

/** How many usable files a directory holds. The count is capped server-side. */
function fileCount(n: number | null): string {
  if (n === null || n === 0) return ''
  if (n >= 2000) return '2000+ files'
  return `${n} file${n === 1 ? '' : 's'}`
}

export function FolderPicker({
  value,
  mode = 'corpus',
  onChange,
  onClose,
}: {
  value: string
  mode?: PickerMode
  onChange: (path: string) => void
  onClose: () => void
}) {
  const [listing, setListing] = useState<BrowseResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const load = (path?: string) => {
    setLoading(true)
    setError(null)
    api
      .browse(path, mode)
      .then(setListing)
      .catch((e) => setError(String(e.message)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    // Start from the current value if it looks like a path, otherwise from home.
    load(value.startsWith('/') ? value : undefined)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const pickingModel = mode === 'model'
  const currentIsModel = listing?.entries.some((e) => e.is_model) ?? false

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="flex max-h-[80vh] w-full max-w-2xl flex-col rounded-lg border border-neutral-800 bg-neutral-900">
        <header className="border-b border-neutral-800 px-4 py-3">
          <h2 className="text-sm font-semibold">
            {pickingModel ? 'Choose a model folder' : 'Choose a corpus folder'}
          </h2>
          <p className="mt-1 truncate font-mono text-xs text-neutral-500">
            {listing?.path ?? '…'}
          </p>
          {pickingModel && (
            <p className="mt-1 text-xs text-neutral-500">
              A model folder contains <span className="font-mono">config.json</span>.
              {currentIsModel && ' Highlighted below.'}
            </p>
          )}
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          {error && <ErrorBox message={error} />}
          {loading && !listing && (
            <div className="p-4">
              <Spinner label="reading directory" />
            </div>
          )}

          {listing?.parent && (
            <button
              onClick={() => load(listing.parent!)}
              className="w-full rounded px-3 py-2 text-left text-sm text-neutral-400 hover:bg-neutral-800"
            >
              ../
            </button>
          )}

          {listing?.entries.map((entry) => (
            <div key={entry.path} className="flex items-center gap-1">
              <button
                onClick={() => load(entry.path)}
                className="flex min-w-0 flex-1 items-center justify-between rounded px-3 py-2 text-left text-sm hover:bg-neutral-800"
              >
                <span className="truncate">
                  {entry.name}/
                  {entry.is_model && (
                    <span className="ml-2 rounded border border-emerald-900 bg-emerald-950 px-1.5 py-0.5 text-xs text-emerald-300">
                      model
                    </span>
                  )}
                </span>
                <span className="ml-3 shrink-0 text-xs text-neutral-500">
                  {fileCount(entry.n_files)}
                </span>
              </button>
              {/* Select a child directly, without having to step into it first. */}
              <button
                onClick={() => {
                  onChange(entry.path)
                  onClose()
                }}
                className="shrink-0 rounded px-2 py-2 text-xs text-neutral-500 hover:bg-neutral-800 hover:text-neutral-200"
                title="Use this folder"
              >
                use
              </button>
            </div>
          ))}

          {listing && listing.entries.length === 0 && (
            <p className="p-4 text-sm text-neutral-500">No subdirectories here.</p>
          )}
        </div>

        <footer className="flex items-center justify-end gap-2 border-t border-neutral-800 px-4 py-3">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            disabled={!listing}
            onClick={() => {
              if (listing) {
                onChange(listing.path)
                onClose()
              }
            }}
          >
            Use this folder
          </Button>
        </footer>
      </div>
    </div>
  )
}

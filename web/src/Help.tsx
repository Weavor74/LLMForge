/**
 * The help documents, rendered from the markdown the API serves.
 *
 * Markdown is the source of truth rather than JSX so the same text is readable in
 * the repository, and so correcting a number does not mean editing a component. The
 * renderer below handles only what these documents actually use — headings, tables,
 * lists, code blocks, bold — which is far less code than a dependency.
 */

import { useEffect, useState } from 'react'
import { api, type HelpPage } from './api'
import { ErrorBox, Panel, Spinner } from './components'

export function Help() {
  const [pages, setPages] = useState<HelpPage[]>([])
  const [slug, setSlug] = useState<string | null>(null)
  const [markdown, setMarkdown] = useState<string>('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .helpPages()
      .then((p) => {
        setPages(p)
        if (p.length) setSlug(p[0].slug)
      })
      .catch((e) => setError((e as Error).message))
  }, [])

  useEffect(() => {
    if (!slug) return
    let cancelled = false
    api
      .helpPage(slug)
      .then((d) => !cancelled && setMarkdown(d.markdown))
      .catch((e) => !cancelled && setError((e as Error).message))
    return () => {
      cancelled = true
    }
  }, [slug])

  if (error) return <ErrorBox message={error} />
  if (!pages.length) return <Spinner label="loading help" />

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <nav className="flex flex-wrap gap-2">
        {pages.map((p) => (
          <button
            key={p.slug}
            onClick={() => setSlug(p.slug)}
            className={`rounded-md border px-3 py-1.5 text-sm transition-colors ${
              slug === p.slug
                ? 'border-emerald-700 bg-emerald-950/50 text-emerald-200'
                : 'border-neutral-700 bg-neutral-900 text-neutral-400 hover:border-neutral-600'
            }`}
          >
            {p.title}
          </button>
        ))}
      </nav>

      <Panel>
        <Markdown source={markdown} />
      </Panel>
    </div>
  )
}

/** Inline formatting: bold, code spans. */
function inline(text: string, key: number) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean)
  return (
    <span key={key}>
      {parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return (
            <strong key={i} className="font-semibold text-neutral-100">
              {part.slice(2, -2)}
            </strong>
          )
        }
        if (part.startsWith('`') && part.endsWith('`')) {
          return (
            <code key={i} className="rounded bg-neutral-800 px-1 py-0.5 font-mono text-[0.85em] text-cyan-300">
              {part.slice(1, -1)}
            </code>
          )
        }
        return <span key={i}>{part}</span>
      })}
    </span>
  )
}

function Markdown({ source }: { source: string }) {
  const lines = source.split('\n')
  const blocks: React.ReactNode[] = []
  let i = 0
  let key = 0

  while (i < lines.length) {
    const line = lines[i]

    // Tables: a header row, a separator, then body rows.
    if (line.startsWith('|') && lines[i + 1]?.startsWith('|')) {
      const rows: string[][] = []
      while (i < lines.length && lines[i].startsWith('|')) {
        const cells = lines[i].split('|').slice(1, -1).map((c) => c.trim())
        if (!cells.every((c) => /^-+$/.test(c.replace(/:/g, '')) || c === '')) rows.push(cells)
        i++
      }
      const [head, ...body] = rows
      blocks.push(
        <div key={key++} className="scroll-x my-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-neutral-700 text-left">
                {head.map((c, j) => (
                  <th key={j} className="whitespace-nowrap px-2 py-1.5 font-semibold text-neutral-300">
                    {inline(c, j)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {body.map((row, r) => (
                <tr key={r} className="border-b border-neutral-900">
                  {row.map((c, j) => (
                    <td key={j} className={`px-2 py-1.5 ${j === 0 ? 'text-neutral-300' : 'text-neutral-400'}`}>
                      {inline(c, j)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      )
      continue
    }

    // Indented blocks are shown as code — the documents use them for formulas.
    if (line.startsWith('    ') && line.trim()) {
      const buffer: string[] = []
      while (i < lines.length && (lines[i].startsWith('    ') || !lines[i].trim())) {
        buffer.push(lines[i].replace(/^ {4}/, ''))
        i++
      }
      blocks.push(
        <pre key={key++} className="scroll-x my-4 rounded border border-neutral-800 bg-neutral-950 p-3 font-mono text-xs text-cyan-300">
          {buffer.join('\n').trim()}
        </pre>,
      )
      continue
    }

    if (line.startsWith('## ')) {
      blocks.push(
        <h2 key={key++} className="mt-8 mb-3 text-lg font-semibold text-neutral-100">
          {line.slice(3)}
        </h2>,
      )
      i++
      continue
    }
    if (line.startsWith('# ')) {
      blocks.push(
        <h1 key={key++} className="mb-4 text-2xl font-semibold text-neutral-50">
          {line.slice(2)}
        </h1>,
      )
      i++
      continue
    }

    if (line.startsWith('- ')) {
      const items: string[] = []
      while (i < lines.length && lines[i].startsWith('- ')) {
        let item = lines[i].slice(2)
        i++
        // Continuation lines are indented under the bullet.
        while (i < lines.length && lines[i].startsWith('  ') && lines[i].trim()) {
          item += ' ' + lines[i].trim()
          i++
        }
        items.push(item)
      }
      blocks.push(
        <ul key={key++} className="my-3 space-y-1.5">
          {items.map((item, j) => (
            <li key={j} className="flex gap-2 text-sm text-neutral-400">
              <span className="select-none text-neutral-600">·</span>
              <span>{inline(item, j)}</span>
            </li>
          ))}
        </ul>,
      )
      continue
    }

    if (!line.trim()) {
      i++
      continue
    }

    // Everything else is a paragraph, joined until a blank line.
    const buffer: string[] = []
    while (i < lines.length && lines[i].trim() && !/^[|#\-]/.test(lines[i]) && !lines[i].startsWith('    ')) {
      buffer.push(lines[i])
      i++
    }
    blocks.push(
      <p key={key++} className="my-3 text-sm leading-relaxed text-neutral-400">
        {inline(buffer.join(' '), 0)}
      </p>,
    )
  }

  return <div>{blocks}</div>
}

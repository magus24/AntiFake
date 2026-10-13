import { useCallback, useEffect, useRef, useState } from 'react'

import {
  ProtectError,
  base64ToBytes,
  fetchHealth,
  protectImage,
  type HealthInfo,
  type ProtectResult,
} from './api/client'
import { DifferenceView } from './components/DifferenceView'
import { DropZone } from './components/DropZone'
import { ImagePanel } from './components/ImagePanel'
import { MetricsReadout } from './components/MetricsReadout'
import { decodeBlob, type RgbImage } from './lib/difference'

type Phase = 'idle' | 'working' | 'done' | 'failed'

interface Selection {
  file: File
  url: string
  width: number
  height: number
}

export default function App() {
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [healthProblem, setHealthProblem] = useState<string | null>(null)
  const [selection, setSelection] = useState<Selection | null>(null)
  const [result, setResult] = useState<ProtectResult | null>(null)
  const [protectedUrl, setProtectedUrl] = useState<string | null>(null)
  const [originalPixels, setOriginalPixels] = useState<RgbImage | null>(null)
  const [protectedPixels, setProtectedPixels] = useState<RgbImage | null>(null)
  const [phase, setPhase] = useState<Phase>('idle')
  const [elapsed, setElapsed] = useState(0)
  const [problem, setProblem] = useState<string | null>(null)

  // Every object URL and in-flight request is released on unmount or reset,
  // so repeated use cannot leak blobs in a long-lived tab.
  const objectUrls = useRef<string[]>([])
  const controller = useRef<AbortController | null>(null)

  const track = useCallback((url: string) => {
    objectUrls.current.push(url)
    return url
  }, [])

  const releaseUrls = useCallback(() => {
    objectUrls.current.forEach((url) => URL.revokeObjectURL(url))
    objectUrls.current = []
  }, [])

  useEffect(() => {
    const ac = new AbortController()
    void fetchHealth(ac.signal)
      .then(setHealth)
      .catch((error: unknown) => {
        if (ac.signal.aborted) return
        setHealthProblem(
          error instanceof ProtectError
            ? error.message
            : 'The service did not answer. Start the backend on port 8000 and reload.',
        )
      })
    return () => ac.abort()
  }, [])

  useEffect(() => releaseUrls, [releaseUrls])

  // Elapsed seconds are shown while working. There is no progress percentage
  // because the server does not report one and inventing one would be a lie.
  useEffect(() => {
    if (phase !== 'working') return
    const started = performance.now()
    const handle = window.setInterval(() => {
      setElapsed((performance.now() - started) / 1000)
    }, 100)
    return () => window.clearInterval(handle)
  }, [phase])

  const reset = useCallback(() => {
    controller.current?.abort()
    controller.current = null
    releaseUrls()
    setSelection(null)
    setResult(null)
    setProtectedUrl(null)
    setOriginalPixels(null)
    setProtectedPixels(null)
    setProblem(null)
    setElapsed(0)
    setPhase('idle')
  }, [releaseUrls])

  const run = useCallback(
    async (file: File, url: string, dimensions: { width: number; height: number }) => {
      controller.current?.abort()
      const ac = new AbortController()
      controller.current = ac

      releaseUrls()
      setSelection({ file, url: track(url), ...dimensions })
      setResult(null)
      setProtectedUrl(null)
      setProtectedPixels(null)
      setProblem(null)
      setElapsed(0)
      setPhase('working')
      setOriginalPixels(null)

      try {
        // The two can fail for unrelated reasons, so they are reported
        // separately: blaming the service for a file the browser cannot read
        // would send the user to the wrong place.
        let pixels: RgbImage
        try {
          pixels = await decodeBlob(file)
        } catch {
          setProblem(
            'Your browser could not decode that file. It may be damaged, or only partly downloaded. Try re-saving it as a JPG or PNG.',
          )
          setPhase('failed')
          return
        }

        let response: ProtectResult
        try {
          response = await protectImage(file, ac.signal)
        } catch (error: unknown) {
          if (ac.signal.aborted) return
          setProblem(
            error instanceof ProtectError
              ? error.message
              : 'The service could not be reached. Check that the backend is running on port 8000.',
          )
          setPhase('failed')
          return
        }
        if (ac.signal.aborted) return

        const bytes = base64ToBytes(response)
        const blob = new Blob([bytes as BlobPart], { type: response.media_type })
        setOriginalPixels(pixels)
        setResult(response)
        setProtectedUrl(track(URL.createObjectURL(blob)))
        setProtectedPixels(await decodeBlob(blob))
        setPhase('done')
      } catch (error: unknown) {
        if (ac.signal.aborted) return
        setProblem('Something went wrong while preparing the comparison. Start over and try again.')
        setPhase('failed')
      } finally {
        if (controller.current === ac) controller.current = null
      }
    },
    [releaseUrls, track],
  )

  const onFile = useCallback(
    (file: File) => {
      const url = URL.createObjectURL(file)
      const probe = new Image()
      probe.onload = () => void run(file, url, { width: probe.naturalWidth, height: probe.naturalHeight })
      probe.onerror = () => {
        URL.revokeObjectURL(url)
        setProblem('That file could not be read as an image.')
        setPhase('failed')
      }
      probe.src = url
    },
    [run],
  )

  const working = phase === 'working'
  const originalMeta = selection
    ? `${selection.width} x ${selection.height} · ${selection.file.name}`
    : undefined
  const protectedMeta =
    phase === 'done' && result
      ? `${result.width} x ${result.height} · ${result.output_format}`
      : working
        ? `protecting · ${elapsed.toFixed(1)}s`
        : undefined

  return (
    <div className="shell">
      <header className="masthead">
        <div className="masthead__brand">
          <span className="masthead__mark" aria-hidden="true" />
          <h1 className="masthead__word">AntiFake</h1>
        </div>
        <p className="masthead__tagline">
          Image protection against one named AI model
        </p>
      </header>

      <section className="statement">
        <p className="statement__lead">
          A perturbation is added to your image, tuned to break a single
          classifier at its own preprocessing. The file comes back the same
          size, the same format, and to the eye the same picture.
        </p>
        <p className="statement__disclosure">
          {health
            ? `Covered: ${health.engine.target_model}. Nothing is claimed for any other model, or for generative models.`
            : healthProblem
              ? healthProblem
              : 'Reading the active target configuration…'}
        </p>
      </section>

      <main className="workbench">
        <div className="compare">
          <ImagePanel
            label="Original"
            src={selection?.url ?? null}
            alt="The image you selected"
            pending={false}
            emptyHint="Your image appears here."
            meta={originalMeta}
          />
          <ImagePanel
            label="Protected"
            src={protectedUrl}
            alt="The protected image, which looks the same as the original"
            pending={working}
            emptyHint="The protected copy appears here."
            meta={protectedMeta}
            accent="signal"
          />
        </div>

        <DifferenceView
          originalPixels={originalPixels}
          protectedPixels={protectedPixels}
          reportedPeak={result?.delivered_metrics.linf_8bit ?? null}
        />

        <MetricsReadout result={result} />

        {phase === 'failed' && problem ? (
          <p className="alert" role="alert">
            {problem}
          </p>
        ) : null}

        <div className="actions">
          {phase === 'done' && result && protectedUrl ? (
            <a className="button button--primary" href={protectedUrl} download={result.filename}>
              Download protected image
            </a>
          ) : (
            <DropZone onFile={onFile} disabled={working} compact={phase !== 'idle'} />
          )}
          {phase === 'done' || phase === 'failed' ? (
            <button type="button" className="button button--quiet" onClick={reset}>
              Start over
            </button>
          ) : null}
        </div>
      </main>
    </div>
  )
}

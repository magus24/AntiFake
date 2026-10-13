import { useEffect, useState } from 'react'

import { renderDifference, type RgbImage } from '../lib/difference'

interface DifferenceViewProps {
  originalPixels: RgbImage | null
  protectedPixels: RgbImage | null
  /** Server-reported peak difference, in 8-bit levels. */
  reportedPeak: number | null
}

const GAINS = [1, 4, 8, 16, 32, 64] as const
type Gain = (typeof GAINS)[number]

/**
 * The instrument's one loud element: the perturbation, made visible.
 *
 * At 1x this frame is almost black, which is the honest picture of the
 * delivered result. Turning the gain up reveals the structure of the field
 * without ever claiming the un-amplified image is different.
 */
export function DifferenceView({
  originalPixels,
  protectedPixels,
  reportedPeak,
}: DifferenceViewProps) {
  const [gain, setGain] = useState<Gain>(16)
  const [rendered, setRendered] = useState<string | null>(null)
  const [size, setSize] = useState<string>('')

  useEffect(() => {
    let cancelled = false
    if (!originalPixels || !protectedPixels) {
      setRendered(null)
      setSize('')
      return
    }
    // Decoding is cheap but not free; yield a frame so the slider stays live.
    const handle = window.setTimeout(() => {
      void renderDifference(originalPixels, protectedPixels, gain).then((result) => {
        if (cancelled) return
        if (!result) {
          setRendered(null)
          return
        }
        setRendered(result.url)
        setSize(`${result.width} x ${result.height}`)
      })
    }, 16)
    return () => {
      cancelled = true
      window.clearTimeout(handle)
    }
  }, [originalPixels, protectedPixels, gain])

  const ready = rendered !== null

  return (
    <section className="difference" aria-labelledby="difference-heading">
      <div className="difference__header">
        <h2 className="difference__heading" id="difference-heading">
          What was added
        </h2>
        <div className="difference__controls">
          <label className="gain" htmlFor="gain-slider">
            <span className="gain__label">Amplify</span>
            <input
              id="gain-slider"
              className="gain__slider"
              type="range"
              min={0}
              max={GAINS.length - 1}
              step={1}
              value={GAINS.indexOf(gain)}
              disabled={!ready}
              onChange={(event) => {
                const next = GAINS[Number(event.target.value)]
                if (next) setGain(next)
              }}
            />
            <output className="gain__value" htmlFor="gain-slider">
              {gain}x
            </output>
          </label>
        </div>
      </div>

      <div className="difference__stage">
        {ready && rendered ? (
          <img className="difference__image" src={rendered} alt="Difference between the original and protected image, amplified for visibility" />
        ) : originalPixels && protectedPixels ? (
          <p className="difference__empty">Reading pixels…</p>
        ) : (
          <p className="difference__empty">
            Protect an image and the exact pixels that were added will appear
            here.
          </p>
        )}
      </div>

      <p className="difference__caption">
        {ready
          ? `Absolute per-channel difference, shown ${gain} times larger than it really is. Peak measured by the server: ${
              reportedPeak === null ? '—' : `${reportedPeak}/255`
            }.`
          : 'Amplification is a viewing aid. The delivered change is never larger than the peak stated here.'}
      </p>
      {size ? <p className="difference__size">{size} px</p> : null}
    </section>
  )
}

import type { ProtectResult } from '../api/client'

interface MetricsReadoutProps {
  result: ProtectResult | null
}

/** Format a value that may legitimately be infinite (identical images). */
function psnr(value: number): string {
  return value >= 999 ? '∞' : value.toFixed(1)
}

export function MetricsReadout({ result }: MetricsReadoutProps) {
  const metrics = result?.delivered_metrics

  return (
    <section className="readout" aria-labelledby="readout-heading">
      <h2 className="readout__heading" id="readout-heading">
        Measured on the file you receive
      </h2>

      {result && metrics ? (
        <dl className="readout__grid">
          <div className="readout__cell">
            <dt>Peak change</dt>
            <dd>
              {metrics.linf_8bit}
              <span className="readout__unit">/255</span>
            </dd>
          </div>
          <div className="readout__cell">
            <dt>PSNR</dt>
            <dd>
              {psnr(metrics.psnr_db)}
              <span className="readout__unit">dB</span>
            </dd>
          </div>
          <div className="readout__cell">
            <dt>SSIM</dt>
            <dd>{metrics.ssim.toFixed(3)}</dd>
          </div>
          <div className="readout__cell">
            <dt>Average change</dt>
            <dd>
              {metrics.mean_abs.toFixed(2)}
              <span className="readout__unit">/255</span>
            </dd>
          </div>
          <div className="readout__cell">
            <dt>Size</dt>
            <dd className="readout__value--text">
              {result.width} x {result.height}
            </dd>
          </div>
          <div className="readout__cell">
            <dt>Took</dt>
            <dd>
              {(result.process_ms / 1000).toFixed(1)}
              <span className="readout__unit">s</span>
            </dd>
          </div>
        </dl>
      ) : (
        <p className="readout__empty">
          Similarity numbers appear once a file has been protected. They are
          measured after encoding, so they describe the downloaded bytes rather
          than an ideal in-memory result.
        </p>
      )}

      {result ? (
        <p className="readout__footnote">
          Engine <code>{result.engine.engine}</code> on{' '}
          <code>{result.engine.device}</code>, {result.engine.steps} steps, budget{' '}
          {result.engine.eps_255.toFixed(1)}/255. {result.disclaimer}
        </p>
      ) : null}
    </section>
  )
}

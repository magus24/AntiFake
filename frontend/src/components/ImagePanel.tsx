interface ImagePanelProps {
  label: string
  src: string | null
  alt: string
  /** Shown instead of the image while a request is in flight. */
  pending: boolean
  emptyHint: string
  meta?: string | undefined
  accent?: 'plain' | 'signal'
}

export function ImagePanel({
  label,
  src,
  alt,
  pending,
  emptyHint,
  meta,
  accent = 'plain',
}: ImagePanelProps) {
  return (
    <figure className={`panel panel--${accent}`}>
      <figcaption className="panel__label">{label}</figcaption>
      <div className="panel__stage">
        {src ? (
          <img className="panel__image" src={src} alt={alt} />
        ) : pending ? (
          <div className="panel__pending" role="status">
            <span className="panel__sweep" aria-hidden="true" />
            <span className="panel__pending-text">working</span>
          </div>
        ) : (
          <p className="panel__empty">{emptyHint}</p>
        )}
      </div>
      <p className="panel__meta">{meta ?? '—'}</p>
    </figure>
  )
}

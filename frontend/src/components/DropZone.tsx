import { useCallback, useId, useRef, useState } from 'react'

import { validateFile } from '../api/client'

interface DropZoneProps {
  onFile: (file: File) => void
  disabled: boolean
  compact: boolean
}

export function DropZone({ onFile, disabled, compact }: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)
  const inputId = useId()

  const accept = useCallback(
    (file: File | undefined) => {
      if (!file) return
      const issue = validateFile(file)
      setProblem(issue)
      if (!issue) onFile(file)
    },
    [onFile],
  )

  return (
    <div className="dropzone-block">
      <button
        type="button"
        className={[
          'dropzone',
          compact ? 'dropzone--compact' : '',
          dragging ? 'dropzone--dragging' : '',
        ]
          .filter(Boolean)
          .join(' ')}
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault()
          if (!disabled) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setDragging(false)
          if (disabled) return
          accept(event.dataTransfer.files[0])
        }}
        aria-describedby={problem ? `${inputId}-problem` : undefined}
      >
        <span className="dropzone__mark" aria-hidden="true" />
        <span className="dropzone__body">
          <span className="dropzone__headline">
            {compact ? 'Protect a different image' : 'Drop a JPG or PNG here'}
          </span>
          <span className="dropzone__hint">
            {compact ? 'Click to choose a file' : 'or click to choose one file'}
          </span>
        </span>
      </button>

      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept="image/jpeg,image/png,.jpg,.jpeg,.png"
        hidden
        onChange={(event) => {
          accept(event.target.files?.[0])
          // Reset so picking the same file twice still fires a change event.
          event.target.value = ''
        }}
      />

      {problem ? (
        <p className="dropzone__problem" id={`${inputId}-problem`} role="alert">
          {problem}
        </p>
      ) : null}
    </div>
  )
}

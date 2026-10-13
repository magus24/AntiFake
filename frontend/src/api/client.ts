/**
 * Thin client for the AntiFake API.
 *
 * The JSON envelope is used rather than the binary endpoint because the
 * difference view needs both the protected bytes and the metrics from a single
 * round trip, and the original pixels are already in the browser from the
 * picked file.
 */

export interface DifferenceMetrics {
  linf_255: number
  linf_8bit: number
  l2: number
  mean_abs: number
  psnr_db: number
  ssim: number
}

export interface EngineInfo {
  engine: string
  target_model: string
  preserves_dimensions: boolean
  device: string
  eps_255: number
  steps: number
}

export interface HealthInfo {
  status: string
  app: string
  version: string
  engine: EngineInfo
  supported_formats: string[]
  max_upload_bytes: number
}

export interface ProtectResult {
  filename: string
  media_type: string
  width: number
  height: number
  source_format: string
  output_format: string
  process_ms: number
  perturbation_metrics: DifferenceMetrics
  delivered_metrics: DifferenceMetrics
  engine: EngineInfo
  disclaimer: string
  image_base64: string
}

export class ProtectError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ProtectError'
    this.status = status
  }
}

const ENDPOINT = '/api/protect'

/** Client-side guard mirroring the server's own limit, to fail fast. */
export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024
export const ACCEPTED_TYPES = ['image/jpeg', 'image/jpg', 'image/png']

function isAccepted(file: File): boolean {
  if (ACCEPTED_TYPES.includes(file.type)) return true
  // Some browsers and OSes report an empty type for a valid file.
  return file.type === '' && /\.(jpe?g|png)$/i.test(file.name)
}

export function validateFile(file: File): string | null {
  /** Returns a human-readable problem, or null when the file is acceptable. */
  if (!isAccepted(file)) {
    return 'Pick a JPG or PNG. Other formats are not read by this service.'
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    const limitMb = (MAX_UPLOAD_BYTES / (1024 * 1024)).toFixed(0)
    return `That file is ${(file.size / (1024 * 1024)).toFixed(1)} MB. The limit is ${limitMb} MB.`
  }
  if (file.size === 0) {
    return 'That file is empty.'
  }
  return null
}

async function readDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json()
    if (typeof body === 'object' && body !== null && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail
      if (typeof detail === 'string') return detail
      // FastAPI validation errors arrive as a list of objects.
      if (Array.isArray(detail) && detail.length > 0) {
        const first = detail[0] as { msg?: unknown }
        if (typeof first?.msg === 'string') return first.msg
      }
    }
  } catch {
    // Non-JSON body; fall through to the status text.
  }
  return response.statusText || 'The request could not be completed.'
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthInfo> {
  const response = await fetch('/health', signal ? { signal } : {})
  if (!response.ok) {
    throw new ProtectError(await readDetail(response), response.status)
  }
  return (await response.json()) as HealthInfo
}

export async function protectImage(
  file: File,
  signal?: AbortSignal,
): Promise<ProtectResult> {
  const form = new FormData()
  form.append('image', file, file.name)

  const init: RequestInit = { method: 'POST', body: form }
  if (signal) init.signal = signal
  const response = await fetch(`${ENDPOINT}?return_image=false`, init)

  if (!response.ok) {
    throw new ProtectError(await readDetail(response), response.status)
  }
  return (await response.json()) as ProtectResult
}

/** Decode the protected payload into the raw bytes the server produced. */
export function base64ToBytes(result: ProtectResult): Uint8Array {
  const binary = atob(result.image_base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes
}

/** Decode the protected payload into an object URL the browser can display. */
export function base64ToObjectUrl(result: ProtectResult): string {
  return URL.createObjectURL(
    new Blob([base64ToBytes(result) as BlobPart], { type: result.media_type }),
  )
}

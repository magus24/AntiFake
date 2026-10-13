/**
 * Client-side difference rendering.
 *
 * The whole point of AntiFake is that the perturbation is invisible, which
 * makes it hard to trust. This computes the absolute per-channel difference
 * between the two images and lets the user amplify it, so the claim can be
 * inspected rather than taken on faith.
 *
 * It runs on the browser's already-decoded bitmaps, so nothing extra is
 * requested from the server.
 */

export interface RgbImage {
  data: Uint8ClampedArray
  width: number
  height: number
}

/** Load a blob URL into raw RGBA pixels. */
export async function readPixels(url: string): Promise<RgbImage> {
  const response = await fetch(url)
  const blob = await response.blob()
  return decodeBlob(blob)
}

export async function decodeBlob(blob: Blob): Promise<RgbImage> {
  const bitmap = await createImageBitmap(blob)
  try {
    const canvas = document.createElement('canvas')
    canvas.width = bitmap.width
    canvas.height = bitmap.height
    const context = canvas.getContext('2d', { willReadFrequently: true })
    if (!context) {
      throw new Error('This browser did not provide a 2D canvas context.')
    }
    context.drawImage(bitmap, 0, 0)
    const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height)
    return { data, width: bitmap.width, height: bitmap.height }
  } finally {
    bitmap.close()
  }
}

/** Peak absolute per-channel difference between two same-sized images. */
export function peakDifference(a: RgbImage, b: RgbImage): number {
  if (a.width !== b.width || a.height !== b.height) {
    return 0
  }
  let peak = 0
  const left = a.data
  const right = b.data
  for (let i = 0; i < left.length; i += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      const delta = Math.abs(left[i + channel]! - right[i + channel]!)
      if (delta > peak) peak = delta
    }
  }
  return peak
}

/**
 * Render |original - protected| amplified, as a PNG blob URL.
 *
 * @param gain Multiplier applied to the difference before display. Values
 *   below 1 are clamped: a protection is meant to be looked *into*, not
 *   dialled down.
 * @param maxSize Longest edge of the output, in device pixels. The
 *   difference field is high-frequency, so it is downsampled with smoothing
 *   to stay legible instead of aliasing into noise.
 */
export async function renderDifference(
  original: RgbImage,
  protectedImage: RgbImage,
  gain: number,
  maxSize = 1400,
): Promise<{ url: string; width: number; height: number } | null> {
  if (original.width !== protectedImage.width || original.height !== protectedImage.height) {
    return null
  }

  const width = original.width
  const height = original.height
  const source = document.createElement('canvas')
  source.width = width
  source.height = height
  const sourceContext = source.getContext('2d')
  if (!sourceContext) return null

  const output = new ImageData(width, height)
  const left = original.data
  const right = protectedImage.data
  const target = output.data
  const safeGain = Math.max(1, gain)

  for (let i = 0; i < left.length; i += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      const delta = Math.abs(left[i + channel]! - right[i + channel]!) * safeGain
      target[i + channel] = delta > 255 ? 255 : delta
    }
    // Opaque: an empty difference field must read as "nothing was added",
    // which is the honest result, not as missing data.
    target[i + 3] = 255
  }
  sourceContext.putImageData(output, 0, 0)

  const scale = Math.min(1, maxSize / Math.max(width, height))
  if (scale < 1) {
    const smoothed = document.createElement('canvas')
    smoothed.width = Math.max(1, Math.round(width * scale))
    smoothed.height = Math.max(1, Math.round(height * scale))
    const smoothedContext = smoothed.getContext('2d')
    if (!smoothedContext) return null
    smoothedContext.imageSmoothingEnabled = true
    smoothedContext.imageSmoothingQuality = 'high'
    smoothedContext.drawImage(source, 0, 0, smoothed.width, smoothed.height)
    return {
      url: smoothed.toDataURL('image/png'),
      width: smoothed.width,
      height: smoothed.height,
    }
  }

  return { url: source.toDataURL('image/png'), width, height }
}

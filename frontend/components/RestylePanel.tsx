import { useState, useEffect, useCallback, useRef } from 'react'
import { VideoPreviewPanel } from './VideoPreviewPanel'
import { validateVideoSource } from '../lib/video-constraints'
import { pathToFileUrl } from '../lib/file-url'
import { ApiClient } from '../lib/api-client'
import { Image, X, Loader2, Check, Wand2, Film } from 'lucide-react'
import { logger } from '../lib/logger'

// Restyle panel for the two-step identity-preserving workflow:
//   1. Drop a video  ->  its first frame is extracted automatically
//   2. A style prompt drives a Z-Image edit of that frame (local Z-edit, or the
//      remote / FAL fallthroughs) producing a stylized first frame
//   3. The user accepts it  ->  the accepted image becomes the stylized first
//      frame passed to id-v2v /restyle.
// The parent (GenSpace) prefills the main prompt with the default "restyle this
// video" once the stylized frame is accepted. The panel still lets the user drop
// their own stylized image as a direct shortcut.

export const DEFAULT_RESTYLE_PROMPT = 'restyle this video'

interface RestylePanelProps {
  initialVideoPath?: string | null
  initialImagePath?: string | null
  resetKey?: number
  isProcessing?: boolean
  processingStatus?: string
  fillHeight?: boolean
  enforceApiConstraints?: boolean
  onChange?: (data: {
    videoPath: string | null
    stylizedImagePath: string | null
    ready: boolean
  }) => void
  // Called when the user accepts a generated/first-frame stylized image, with the
  // accepted image path. The parent uses it to prefill the default restyle prompt.
  onAccept?: (acceptedImagePath: string, videoPath: string | null) => void
}

export function RestylePanel({
  initialVideoPath,
  initialImagePath,
  resetKey,
  isProcessing = false,
  processingStatus = '',
  enforceApiConstraints = true,
  onChange,
  onAccept,
}: RestylePanelProps) {
  const [videoPath, setVideoPath] = useState<string | null>(initialVideoPath || null)
  const [stylizedImagePath, setStylizedImagePath] = useState<string | null>(initialImagePath || null)
  const [dimensions, setDimensions] = useState<{ width: number; height: number }>({ width: 0, height: 0 })
  const [videoDuration, setVideoDuration] = useState(0)

  // First-frame styling workflow state.
  const [extractedFramePath, setExtractedFramePath] = useState<string | null>(null)
  const [stylePrompt, setStylePrompt] = useState('')
  const [candidateImagePath, setCandidateImagePath] = useState<string | null>(null)
  const [isStyling, setIsStyling] = useState(false)
  const [isExtracting, setIsExtracting] = useState(false)
  const [stylingError, setStylingError] = useState<string | null>(null)

  const videoKnownRef = useRef<string | null>(null)

  const handleSourceChange = useCallback(async (data: { videoPath: string | null; videoDuration: number; width: number; height: number }) => {
    setVideoPath(data.videoPath)
    setVideoDuration(data.videoDuration)
    setDimensions({ width: data.width, height: data.height })

    // Auto-extract the first frame whenever a new source video lands.
    if (data.videoPath && data.videoPath !== videoKnownRef.current) {
      videoKnownRef.current = data.videoPath
      setStylizedImagePath(null)
      setCandidateImagePath(null)
      setExtractedFramePath(null)
      await extractFirstFrame(data.videoPath)
    }
  }, [])

  const extractFirstFrame = useCallback(async (path: string) => {
    setIsExtracting(true)
    setStylingError(null)
    try {
      const res = await ApiClient.extractFirstFrame({ video_path: path })
      if (!res.ok) {
        logger.error(`First-frame extraction failed: ${res.error?.message}`)
        setStylingError(res.error?.message ?? 'Could not extract the first frame')
        return
      }
      setExtractedFramePath(res.data.imagePath)
    } catch (e) {
      logger.error(`First-frame extraction exception: ${e}`)
      setStylingError(e instanceof Error ? e.message : 'Could not extract the first frame')
    } finally {
      setIsExtracting(false)
    }
  }, [])

  const canStyle = !!(extractedFramePath && stylePrompt.trim() && !isStyling && !isProcessing)

  // Precedence for what's shown in the first-frame panel: a freshly generated
  // candidate (not yet accepted) > the accepted/stylized image > the raw extracted frame.
  const displayFramePath = candidateImagePath || stylizedImagePath || extractedFramePath

  const handleStyleFirstFrame = useCallback(async () => {
    if (!extractedFramePath) return
    if (!stylePrompt.trim()) return
    setIsStyling(true)
    setStylingError(null)
    try {
      // Z-Image edit of the extracted first frame, driven by the style prompt. The
      // backend routes this through local Z-edit, the remote Livepeer runner, or the
      // FAL fallthrough automatically.
      const res = await ApiClient.generateImage({
        prompt: stylePrompt.trim(),
        width: 1024,
        height: 1024,
        numSteps: 8,
        numImages: 1,
        guidanceScale: 0.0,
        strength: 0.6,
        imagePath: extractedFramePath,
      })
      if (!res.ok) {
        logger.error(`First-frame restyle failed: ${res.error?.message}`)
        setStylingError(res.error?.message ?? 'Restyle failed')
        return
      }
      const payload = res.data
      if (payload.status === 'cancelled') {
        setStylingError('Restyle cancelled')
        return
      }
      const path = payload.image_paths?.[0]
      if (!path) {
        setStylingError('Restyle produced no image')
        return
      }
      setCandidateImagePath(path)
    } catch (e) {
      logger.error(`First-frame restyle exception: ${e}`)
      setStylingError(e instanceof Error ? e.message : 'Restyle failed')
    } finally {
      setIsStyling(false)
    }
  }, [extractedFramePath, stylePrompt, isProcessing])

  const handleAccept = useCallback(() => {
    const accepted = candidateImagePath || extractedFramePath
    if (!accepted) return
    setStylizedImagePath(accepted)
    onAccept?.(accepted, videoPath)
  }, [candidateImagePath, extractedFramePath, videoPath, onAccept])

  const error = enforceApiConstraints && videoPath
    ? validateVideoSource({ width: dimensions.width, height: dimensions.height, duration: videoDuration })
    : null

  useEffect(() => {
    if (resetKey === undefined) return
    setStylizedImagePath(initialImagePath || null)
    setCandidateImagePath(initialImagePath || null)
  }, [resetKey, initialImagePath])

  useEffect(() => {
    onChange?.({
      videoPath,
      stylizedImagePath,
      ready: !!videoPath && !!stylizedImagePath && !error,
    })
  }, [videoPath, stylizedImagePath, error, onChange])

  return (
    <div className="flex-1 flex flex-col min-h-0 gap-3">
      {/* Side-by-side: source video preview on the left, extracted first frame on the right */}
      <div className="flex-1 min-h-0 flex gap-3">
        <div className="flex-1 min-w-0 min-h-0">
          <VideoPreviewPanel
            title="Restyle"
            initialVideoPath={initialVideoPath}
            resetKey={resetKey}
            isProcessing={isProcessing}
            processingStatus={processingStatus}
            processingDefault="Restyling video..."
            fillHeight
            emptyTitle="Drop a video to restyle"
            hint={{ title: 'Restyle your video in a new style' }}
            errorMessage={error ?? undefined}
            onSourceChange={handleSourceChange}
          />
        </div>

        {/* Extracted-first-frame panel (right of the video preview) */}
        <div className="w-[300px] flex-shrink-0 min-h-0 flex flex-col gap-2 rounded-2xl border border-zinc-800 bg-zinc-900/60 p-3 overflow-hidden">
          <div className="flex items-center gap-2 flex-shrink-0">
            <Film className="h-4 w-4 text-zinc-400" />
            <span className="text-xs text-zinc-300 font-medium">
              {isExtracting ? 'Extracting first frame...' : 'First frame'}
            </span>
            {displayFramePath !== extractedFramePath && displayFramePath && (
              <span className="ml-auto text-[10px] text-emerald-400">
                {candidateImagePath && stylizedImagePath !== candidateImagePath ? 'candidate' : 'stylized'}
              </span>
            )}
          </div>

          {/* Extracted (then optionally stylized) first frame */}
          <div className="flex-1 min-h-0 relative rounded-xl border border-zinc-800 bg-black flex items-center justify-center overflow-hidden">
            {isExtracting ? (
              <div className="flex flex-col items-center gap-2 text-zinc-500">
                <Loader2 className="h-5 w-5 animate-spin" />
                <span className="text-xs">Extracting first frame...</span>
              </div>
            ) : displayFramePath ? (
              <img src={pathToFileUrl(displayFramePath)} alt="" className="w-full h-full object-contain" />
            ) : (
              <div className="flex flex-col items-center gap-2 text-zinc-600">
                <Image className="h-6 w-6" />
                <span className="text-[11px] px-3 text-center">Drop a video to extract its first frame</span>
              </div>
            )}
            {displayFramePath && !isExtracting && stylizedImagePath !== extractedFramePath && (
              <button
                onClick={(e) => { e.stopPropagation(); setStylizedImagePath(null) }}
                className="absolute top-1.5 right-1.5 p-1 rounded-full bg-zinc-800/80 text-zinc-400 hover:text-white"
                title="Clear"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>

          {/* Style prompt + restyle/accept controls */}
          <div className="flex-shrink-0 flex flex-col gap-2">
            <input
              value={stylePrompt}
              onChange={(e) => setStylePrompt(e.target.value)}
              placeholder="Describe a style..."
              className="w-full bg-zinc-800/60 border border-zinc-700 rounded-md px-2 py-1.5 text-sm text-white placeholder:text-zinc-500 focus:outline-none focus:border-zinc-500"
              disabled={!extractedFramePath || isStyling || isProcessing}
            />

            {extractedFramePath && !candidateImagePath ? (
              // Step 1 done: raw first frame extracted. Offer to restyle it, or accept
              // the extracted frame as-is to proceed with the restyle.
              <div className="flex gap-2">
                <button
                  onClick={handleStyleFirstFrame}
                  disabled={!canStyle}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md bg-zinc-700 text-white text-xs font-medium hover:bg-zinc-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  {isStyling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
                  Restyle frame
                </button>
                <button
                  onClick={handleAccept}
                  disabled={!extractedFramePath}
                  className="flex-1 flex items-center justify-center gap-1 px-3 py-1.5 rounded-md bg-emerald-600 text-white text-xs font-medium hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  <Check className="h-3.5 w-3.5" />
                  Accept
                </button>
              </div>
            ) : candidateImagePath && stylizedImagePath !== candidateImagePath ? (
              // Step 2 done: a stylized candidate is generated but not yet accepted.
              <div className="flex gap-2">
                <button
                  onClick={handleStyleFirstFrame}
                  disabled={!canStyle}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md bg-zinc-700 text-white text-xs font-medium hover:bg-zinc-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  <Wand2 className="h-3.5 w-3.5" />
                  Re-style
                </button>
                <button
                  onClick={handleAccept}
                  className="flex-1 flex items-center justify-center gap-1 px-3 py-1.5 rounded-md bg-emerald-600 text-white text-xs font-medium hover:bg-emerald-500 transition-colors"
                >
                  <Check className="h-3.5 w-3.5" />
                  Accept
                </button>
              </div>
            ) : stylizedImagePath ? (
              // An image is accepted (extracted or stylized); allow re-styling.
              <button
                onClick={handleStyleFirstFrame}
                disabled={!canStyle}
                className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md bg-zinc-700 text-white text-xs font-medium hover:bg-zinc-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {isStyling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
                Re-style
              </button>
            ) : (
              <button
                onClick={handleStyleFirstFrame}
                disabled={!canStyle}
                className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md bg-zinc-700 text-white text-xs font-medium hover:bg-zinc-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {isStyling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
                Restyle frame
              </button>
            )}

            {stylingError && (
              <p className="text-[11px] text-red-400">{stylingError}</p>
            )}
            {candidateImagePath && stylizedImagePath !== candidateImagePath && (
              <p className="text-[10px] text-emerald-400">Generated a stylized frame — accept to use it.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

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
  fillHeight = false,
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

  const imageInputRef = useRef<HTMLInputElement>(null)
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

  const handleImageFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const filePath = window.electronAPI?.getPathForFile?.(file)
    const resolved = filePath || URL.createObjectURL(file)
    setStylizedImagePath(resolved)
    setCandidateImagePath(resolved)
  }, [])

  const handleImageDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      try {
        const asset = JSON.parse(assetData) as { type?: string; path?: string }
        if (asset.type === 'image' && asset.path) {
          setStylizedImagePath(asset.path)
          setCandidateImagePath(asset.path)
          return
        }
      } catch {
        // fall through to file drop
      }
    }
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) {
      const filePath = window.electronAPI?.getPathForFile?.(file)
      const resolved = filePath || URL.createObjectURL(file)
      setStylizedImagePath(resolved)
      setCandidateImagePath(resolved)
    }
  }, [])

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
    <div className="flex-1 flex flex-col gap-3 min-h-0">
      <div className="flex-1 min-h-0">
        <VideoPreviewPanel
          title="Restyle"
          initialVideoPath={initialVideoPath}
          resetKey={resetKey}
          isProcessing={isProcessing}
          processingStatus={processingStatus}
          processingDefault="Restyling video..."
          fillHeight={fillHeight}
          emptyTitle="Drop a video to restyle"
          hint={{ title: 'Restyle your video in a new style', subtitle: 'Describe the style below to restyle the first frame, then confirm it' }}
          errorMessage={error ?? undefined}
          onSourceChange={handleSourceChange}
        />
      </div>

      {/* First-frame stylize step: style prompt + generate + accept */}
      <div className="flex-shrink-0 rounded-xl border border-zinc-800 bg-zinc-900/60 p-3 flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <Film className="h-4 w-4 text-zinc-400 flex-shrink-0" />
          <span className="text-xs text-zinc-400">
            {isExtracting ? 'Extracting first frame...' : 'Restyle the first frame'}
          </span>
        </div>
        <div className="flex gap-2">
          <input
            value={stylePrompt}
            onChange={(e) => setStylePrompt(e.target.value)}
            placeholder="e.g. cyberpunk anime, watercolor, cinematic film still..."
            className="flex-1 bg-zinc-800/60 border border-zinc-700 rounded-md px-2 py-1.5 text-sm text-white placeholder:text-zinc-500 focus:outline-none focus:border-zinc-500"
            disabled={!extractedFramePath || isStyling || isProcessing}
          />
          <button
            onClick={handleStyleFirstFrame}
            disabled={!canStyle}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-zinc-700 text-white text-xs font-medium hover:bg-zinc-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex-shrink-0"
          >
            {isStyling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
            Restyle frame
          </button>
        </div>

        {stylingError && (
          <p className="text-[11px] text-red-400">{stylingError}</p>
        )}

        {/* Candidate result + accept */}
        {candidateImagePath && (
          <div className="flex items-center gap-2">
            <div className="relative w-16 h-16 rounded-md overflow-hidden flex-shrink-0">
              <img src={pathToFileUrl(candidateImagePath)} alt="" className="w-full h-full object-cover" />
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-[11px] text-emerald-400 font-medium">Generated stylized first frame</span>
              <div className="flex gap-2">
                <button
                  onClick={handleAccept}
                  className="flex items-center gap-1 px-3 py-1 rounded-md bg-emerald-600 text-white text-xs font-medium hover:bg-emerald-500 transition-colors"
                >
                  <Check className="h-3.5 w-3.5" />
                  Accept
                </button>
                <button
                  onClick={() => setCandidateImagePath(null)}
                  className="px-3 py-1 rounded-md bg-zinc-800 text-zinc-300 text-xs font-medium hover:bg-zinc-700 transition-colors"
                >
                  Reject
                </button>
              </div>
            </div>
          </div>
        )}

        {!candidateImagePath && stylizedImagePath && (
          <p className="text-[11px] text-zinc-400">Using {stylizedImagePath === extractedFramePath ? 'the extracted first frame' : 'a stylized image'}</p>
        )}
      </div>

      {/* Stylized first-frame image (manual shortcut / shows accepted result) */}
      <div
        className={`relative h-28 rounded-xl border-2 border-dashed transition-colors flex items-center justify-center flex-shrink-0 cursor-pointer overflow-hidden ${
          stylizedImagePath ? 'border-emerald-600' : 'border-zinc-700 hover:border-zinc-500'
        }`}
        onClick={() => imageInputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleImageDrop}
      >
        {stylizedImagePath ? (
          <>
            <img src={pathToFileUrl(stylizedImagePath)} alt="" className="w-full h-full object-cover" />
            <span className="absolute bottom-1 left-1 px-1.5 py-0.5 rounded bg-black/60 text-[10px] text-emerald-300">
              Stylized first frame
            </span>
            <button
              onClick={(e) => { e.stopPropagation(); setStylizedImagePath(null) }}
              className="absolute top-1 right-1 p-1 rounded-full bg-zinc-800 text-zinc-400 hover:text-white"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </>
        ) : (
          <div className="flex flex-col items-center gap-1 text-zinc-500">
            <Image className="h-5 w-5" />
            <span className="text-xs">or drop a stylized first-frame image</span>
          </div>
        )}
        <input
          ref={imageInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageFileSelect}
          className="hidden"
        />
      </div>
    </div>
  )
}

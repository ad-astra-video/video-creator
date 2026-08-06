import { useState, useEffect, useCallback, useRef } from 'react'
import { VideoPreviewPanel } from './VideoPreviewPanel'
import { validateVideoSource } from '../lib/video-constraints'
import { pathToFileUrl } from '../lib/file-url'
import { Image, X } from 'lucide-react'

// Preview panel for Restyle: a source video + a stylized first-frame image, both
// required to submit (identity-preserving video restylization via id-v2v).

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
}: RestylePanelProps) {
  const [videoPath, setVideoPath] = useState<string | null>(initialVideoPath || null)
  const [stylizedImagePath, setStylizedImagePath] = useState<string | null>(initialImagePath || null)
  const [dimensions, setDimensions] = useState<{ width: number; height: number }>({ width: 0, height: 0 })
  const [videoDuration, setVideoDuration] = useState(0)
  const imageInputRef = useRef<HTMLInputElement>(null)

  const handleSourceChange = useCallback((data: { videoPath: string | null; videoDuration: number; width: number; height: number }) => {
    setVideoPath(data.videoPath)
    setVideoDuration(data.videoDuration)
    setDimensions({ width: data.width, height: data.height })
  }, [])

  const error = enforceApiConstraints && videoPath
    ? validateVideoSource({ width: dimensions.width, height: dimensions.height, duration: videoDuration })
    : null

  const handleImageFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const filePath = window.electronAPI?.getPathForFile?.(file)
    setStylizedImagePath(filePath || URL.createObjectURL(file))
  }, [])

  const handleImageDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      try {
        const asset = JSON.parse(assetData) as { type?: string; path?: string }
        if (asset.type === 'image' && asset.path) {
          setStylizedImagePath(asset.path)
          return
        }
      } catch {
        // fall through to file drop
      }
    }
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) {
      const filePath = window.electronAPI?.getPathForFile?.(file)
      setStylizedImagePath(filePath || URL.createObjectURL(file))
    }
  }, [])

  useEffect(() => {
    if (resetKey === undefined) return
    setStylizedImagePath(initialImagePath || null)
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
          hint={{ title: 'Restyle your video in a new style', subtitle: 'Add a stylized first-frame image below to set the look' }}
          errorMessage={error ?? undefined}
          onSourceChange={handleSourceChange}
        />
      </div>

      {/* Stylized first-frame image dropzone */}
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
            <span className="text-xs">Drop a stylized first-frame image to set the style</span>
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

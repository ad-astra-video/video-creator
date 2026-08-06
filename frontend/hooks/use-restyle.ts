import { useCallback, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import { withGenerationActive } from '../lib/generation-active'
import { logger } from '../lib/logger'

export interface RestyleSubmitParams {
  videoPath: string
  stylizedImagePath: string
  prompt: string
  maxFrames?: number
  inferenceSteps?: number
  cfgScale?: number
}

export interface RestyleResult {
  videoPath: string
}

interface UseRestyleState {
  isRestyling: boolean
  restyleStatus: string
  restyleError: string | null
  result: RestyleResult | null
}

export function useRestyle() {
  const [state, setState] = useState<UseRestyleState>({
    isRestyling: false,
    restyleStatus: '',
    restyleError: null,
    result: null,
  })

  const submitRestyle = useCallback(async (params: RestyleSubmitParams) => {
    if (!params.videoPath || !params.stylizedImagePath) return

    setState({
      isRestyling: true,
      restyleStatus: 'Restyling',
      restyleError: null,
      result: null,
    })

    await withGenerationActive(async () => {
      const result = await ApiClient.restyle({
        video_path: params.videoPath,
        stylized_image_path: params.stylizedImagePath,
        prompt: params.prompt,
        max_frames: params.maxFrames ?? 81,
        inference_steps: params.inferenceSteps ?? 30,
        cfg_scale: params.cfgScale ?? 5.0,
      })

      if (!result.ok) {
        logger.error(`Restyle error: ${result.error.message}`)
        setState({
          isRestyling: false,
          restyleStatus: '',
          restyleError: result.error.message,
          result: null,
        })
        return
      }

      const payload = result.data

      if (payload.status === 'cancelled') {
        setState({
          isRestyling: false,
          restyleStatus: 'Cancelled',
          restyleError: null,
          result: null,
        })
        return
      }

      setState({
        isRestyling: false,
        restyleStatus: 'Restyle complete!',
        restyleError: null,
        result: {
          videoPath: payload.video_path,
        },
      })
    })
  }, [])

  const resetRestyle = useCallback(() => {
    setState({
      isRestyling: false,
      restyleStatus: '',
      restyleError: null,
      result: null,
    })
  }, [])

  return {
    submitRestyle,
    resetRestyle,
    isRestyling: state.isRestyling,
    restyleStatus: state.restyleStatus,
    restyleError: state.restyleError,
    restyleResult: state.result,
  }
}

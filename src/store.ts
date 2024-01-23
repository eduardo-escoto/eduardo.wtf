import { atom } from 'nanostores'
import { type AnimationPoint2D } from '@/animation/types'

export const $pointState = atom<Array<AnimationPoint2D>>([])
// export const $connectionState = atom<Array<PathAnimationData>>([])
export const $animationInterval = atom()

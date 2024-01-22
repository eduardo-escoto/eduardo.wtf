import { atom, map } from 'nanostores'

export interface RectangleLocation {
  x: number
  y: number
}

export interface CircleLocation {
  x: number
  y: number
  r: number
}

export const $animationState = atom<Array<RectangleLocation | CircleLocation>>([])
export const $animationInterval = atom<Interval1D>()

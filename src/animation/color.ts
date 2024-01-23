import { type AnimationHSLPosition } from './types'

export class HSLPosition implements AnimationHSLPosition {
  hue: number
  saturation: number
  lightness: number
  constructor(hue: number, saturation: number, lightness: number) {
    this.hue = hue
    this.saturation = saturation
    this.lightness = lightness
  }
  toHSLDefinitionString(): string {
    return `hsl(${this.hue}, ${this.saturation}%, ${this.lightness}%)`
  }
  get h() {
    return this.hue
  }
  get s() {
    return this.saturation
  }
  get l() {
    return this.lightness
  }
}

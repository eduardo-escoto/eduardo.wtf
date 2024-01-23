import { type AnimationVelocityVector2D } from './types'

export class Vector2D implements AnimationVelocityVector2D {
  magnitude: number
  angle: number
  constructor(magnitude: number, angle: number) {
    this.magnitude = magnitude
    this.angle = angle
  }
  get x() {
    return this.magnitude * Math.cos(this.angle)
  }
  get y() {
    return this.magnitude * Math.sin(this.angle)
  }
}

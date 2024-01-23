import { HSLPosition } from './color'
import { Position2D } from './position'

import {
  type AnimationHSLPosition,
  type AnimationPoint2D,
  type AnimationPositionTuple2D,
  type AnimationVelocityVector2D,
} from './types'

import { Vector2D } from './vectors'

export class Point2D implements AnimationPoint2D {
  radius_len: number
  position: AnimationPositionTuple2D
  velocity_vec: AnimationVelocityVector2D
  color: AnimationHSLPosition
  constructor(
    radius_len: number,
    position: AnimationPositionTuple2D,
    velocity_vec: AnimationVelocityVector2D,
    color: AnimationHSLPosition
  ) {
    this.radius_len = radius_len
    this.position = position
    this.velocity_vec = velocity_vec
    this.color = color
  }
  static RandomPointFactory(): AnimationPoint2D {
    return new Point2D(
      2 as number,
      new Position2D(Math.random(), Math.random()),
      new Vector2D(Math.random() * 0.1, Math.random()),
      new HSLPosition(Math.random(), Math.random(), Math.random())
    )
  }
  get x() {
    return this.position.x
  }
  get y() {
    return this.position.y
  }
  get velocity_vec_angle() {
    return this.velocity_vec.angle
  }
  get velocity_vec_magnitude() {
    return this.velocity_vec.magnitude
  }
  get hue() {
    return this.color.hue
  }
  get saturation() {
    return this.color.saturation
  }
  get lightness() {
    return this.color.lightness
  }
}

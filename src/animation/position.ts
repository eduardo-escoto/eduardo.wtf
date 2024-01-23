import { type AnimationPositionTuple2D, type AnimationVelocityVector2D } from './types'

export class Position2D implements AnimationPositionTuple2D {
  x: number
  y: number
  constructor(x: number, y: number) {
    this.x = x
    this.y = y
  }
  add(directionVector: AnimationVelocityVector2D): AnimationPositionTuple2D {
    return new Position2D(this.x + directionVector.x, this.y + directionVector.y)
  }
}

export interface AnimationHSLPosition {
  hue: number
  saturation: number
  lightness: number
  toHSLDefinitionString(): string
}

export interface AnimationVelocityVector1D {
  magnitude: number
}

export interface AnimationVelocityVector2D {
  magnitude: number
  angle: number
  readonly x: number
  readonly y: number
}

export interface AnimationPositionTuple2D {
  x: number
  y: number
  add: (directionVector: AnimationVelocityVector2D) => AnimationPositionTuple2D
}

export interface AnimationPoint2D {
  radius_len: number
  position: AnimationPositionTuple2D
  velocity_vec: AnimationVelocityVector2D
  color: AnimationHSLPosition
}

export interface PathAnimationData {
  startPoint: AnimationPositionTuple2D
  endPoint: AnimationPositionTuple2D
  saturation_vec: AnimationVelocityVector1D
  color: AnimationHSLPosition
}

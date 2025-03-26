import tgpu from 'typegpu'
import { f32, i32 } from 'typegpu/data'

export const encodeFixedPoint = tgpu['~unstable']
    .fn([f32], i32)
    .does((floatingPoint) => i32(floatingPoint * 1e7))

export const decodeFixedPoint = tgpu['~unstable']
    .fn([i32], f32)
    .does((fixedPoint) => f32(fixedPoint) / 1e7)

import tgpu from "typegpu";
import { f32, i32 } from "typegpu/data";

export const encodeFixedPoint = tgpu["~unstable"]
  .fn([f32], i32)
  .does((floating_point) => i32(floating_point * 1e7));

export const decodeFixedPoint = tgpu["~unstable"]
  .fn([i32], f32)
  .does((fixed_point) => f32(fixed_point) / 1e7);

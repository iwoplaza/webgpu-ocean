import tgpu from 'typegpu';
import { f32, i32 } from 'typegpu/data';

export const encodeFixedPoint = tgpu['~unstable'].fn(
    { value: f32 },
    i32
)((args) => i32(args.value * 1e7));

export const decodeFixedPoint = tgpu['~unstable'].fn(
    { value: i32 },
    f32
)((args) => f32(args.value) / 1e7);

import { arrayOf, f32, struct, u32, vec3f } from "typegpu/data";

export const Particle = struct({
  position: vec3f,
  v: vec3f,
  force: vec3f,
  density: f32,
  nearDensity: f32,
});

export const ParticleArray = (n: number) => arrayOf(Particle, n);

export const SPHParams = struct({
  mass: f32,
  kernelRadius: f32,
  kernelRadiusPow2: f32,
  kernelRadiusPow5: f32,
  kernelRadiusPow6: f32,
  kernelRadiusPow9: f32,
  dt: f32,
  stiffness: f32,
  nearStiffness: f32,
  restDensity: f32,
  viscosity: f32,
  n: u32,
});

export const PosVel = struct({
  position: vec3f,
  v: vec3f,
});

export const PosVelArray = (n: number) => arrayOf(PosVel, n);

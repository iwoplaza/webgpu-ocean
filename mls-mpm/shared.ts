import { arrayOf, mat3x3f, struct, vec3f } from "typegpu/data";

export const Particle = struct({
  position: vec3f,
  v: vec3f,
  C: mat3x3f,
});

export const ParticleArray = (n: number) => arrayOf(Particle, n);

export const PosVel = struct({
  position: vec3f,
  v: vec3f,
});

export const PosVelArray = (n: number) => arrayOf(PosVel, n);

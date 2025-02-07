import { arrayOf, i32, mat3x3f, struct, vec3f } from "typegpu/data";

export const Particle = struct({
  position: vec3f,
  v: vec3f,
  C: mat3x3f,
}).$name("Particle");

export const ParticleArray = (n: number) => arrayOf(Particle, n);

export const Cell = struct({
  vx: i32,
  vy: i32,
  vz: i32,
  mass: i32,
}).$name("Cell");

export const CellArray = (n: number) => arrayOf(Cell, n);

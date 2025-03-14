import tgpu from "typegpu";
import { CellAtomicArray, ParticleArray } from "./shared";
import { builtin, vec3f } from "typegpu/data";
import { encodeFixedPoint } from "./fixedPoint";

export const p2g_1Layout = tgpu
  .bindGroupLayout({
    particles: { storage: ParticleArray },
    cells: { storage: CellAtomicArray, access: "mutable" },
    initBoxSize: { uniform: vec3f },
  })
  .$idx(0);

export const p2g_1Fn = tgpu["~unstable"]
  .computeFn({
    workgroupSize: [64],
    in: { id: builtin.globalInvocationId },
  })
  .does(
    `(input: Input) {
      if (input.id.x < arrayLength(&particles)) {
          var weights: array<vec3f, 3>;

          let particle = particles[input.id.x];
          let cell_idx: vec3f = floor(particle.position);
          let cell_diff: vec3f = particle.position - (cell_idx + 0.5f);
          weights[0] = 0.5f * (0.5f - cell_diff) * (0.5f - cell_diff);
          weights[1] = 0.75f - cell_diff * cell_diff;
          weights[2] = 0.5f * (0.5f + cell_diff) * (0.5f + cell_diff);

          let C: mat3x3f = particle.C;

          for (var gx = 0; gx < 3; gx++) {
              for (var gy = 0; gy < 3; gy++) {
                  for (var gz = 0; gz < 3; gz++) {
                      let weight: f32 = weights[gx].x * weights[gy].y * weights[gz].z;
                      let cell_x: vec3f = vec3f(
                              cell_idx.x + f32(gx) - 1.,
                              cell_idx.y + f32(gy) - 1.,
                              cell_idx.z + f32(gz) - 1.
                          );
                      let cell_dist = (cell_x + 0.5f) - particle.position;

                      let Q: vec3f = C * cell_dist;

                      let mass_contrib: f32 = weight * 1.0; // assuming particle.mass = 1.0
                      let vel_contrib: vec3f = mass_contrib * (particle.v + Q);
                      let cell_index: i32 =
                          i32(cell_x.x) * i32(initBoxSize.y) * i32(initBoxSize.z) +
                          i32(cell_x.y) * i32(initBoxSize.z) +
                          i32(cell_x.z);
                      atomicAdd(&cells[cell_index].mass, encodeFixedPoint(mass_contrib));
                      atomicAdd(&cells[cell_index].vx, encodeFixedPoint(vel_contrib.x));
                      atomicAdd(&cells[cell_index].vy, encodeFixedPoint(vel_contrib.y));
                      atomicAdd(&cells[cell_index].vz, encodeFixedPoint(vel_contrib.z));
                  }
              }
          }
      }
    }`,
  )
  .$uses({ ...p2g_1Layout.bound, encodeFixedPoint });

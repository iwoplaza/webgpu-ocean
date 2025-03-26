import tgpu from 'typegpu';
import { floor } from 'typegpu/std';
import { arrayOf, f32, i32, u32, vec3f, vec3i } from 'typegpu/data';
import { Environment, ParticleArray, SPHParams } from './shared';

const densityLayout = tgpu
    .bindGroupLayout({
        particles: { storage: ParticleArray, access: 'mutable' },
        sortedParticles: { storage: ParticleArray },
        prefixSum: { storage: (n: number) => arrayOf(u32, n) },
        env: { uniform: Environment },
        params: { uniform: SPHParams },
    })
    .$idx(0);

const { params, env } = densityLayout.bound;

const nearDensityKernel = tgpu['~unstable'].fn([f32], f32).does((r) => {
    const scale = 15.0 / (3.1415926535 * params.value.kernelRadiusPow6);
    const d = params.value.kernelRadius - r;
    return scale * d * d * d;
});

const densityKernel = tgpu['~unstable'].fn([f32], f32).does((r) => {
    const scale = 315.0 / (64 * 3.1415926535 * params.value.kernelRadiusPow9);
    const dd = params.value.kernelRadiusPow2 - r * r;
    return scale * dd * dd * dd;
});

const cellPosition = tgpu['~unstable'].fn([vec3f], vec3i).does((v) => {
    const xi = i32(
        floor((v.x + env.value.xHalf + env.value.offset) / env.value.cellSize)
    );
    const yi = i32(
        floor((v.y + env.value.yHalf + env.value.offset) / env.value.cellSize)
    );
    const zi = i32(
        floor((v.z + env.value.zHalf + env.value.offset) / env.value.cellSize)
    );
    return vec3i(xi, yi, zi);
});

const cellNumberFromId = tgpu['~unstable']
    .fn([i32, i32, i32], i32)
    .does(
        (xi, yi, zi) =>
            xi +
            yi * env.value.xGrids +
            zi * env.value.xGrids * env.value.yGrids
    );

export const densityShader = tgpu.resolve({
    template: `
    @compute @workgroup_size(64)
    fn computeDensity(@builtin(global_invocation_id) id: vec3<u32>) {
      if (id.x < params.n) {
          particles[id.x].density = 0.0;
          particles[id.x].nearDensity = 0.0;
          let pos_i = particles[id.x].position;
          let n = params.n;

          let v = cellPosition(pos_i);
          if (v.x < env.xGrids && 0 <= v.x &&
              v.y < env.yGrids && 0 <= v.y &&
              v.z < env.zGrids && 0 <= v.z)
          {
              for (var dz = max(-1, -v.z); dz <= min(1, env.zGrids - v.z - 1); dz++) {
                  for (var dy = max(-1, -v.y); dy <= min(1, env.yGrids - v.y - 1); dy++) {
                      let dxMin = max(-1, -v.x);
                      let dxMax = min(1, env.xGrids - v.x - 1);
                      let startCellNum = cellNumberFromId(v.x + dxMin, v.y + dy, v.z + dz);
                      let endCellNum = cellNumberFromId(v.x + dxMax, v.y + dy, v.z + dz);
                      let start = prefixSum[startCellNum];
                      let end = prefixSum[endCellNum + 1];
                      for (var j = start; j < end; j++) {
                          let pos_j = sortedParticles[j].position;
                          let r2 = dot(pos_i - pos_j, pos_i - pos_j);
                          if (r2 < params.kernelRadiusPow2) {
                              particles[id.x].density += params.mass * densityKernel(sqrt(r2));
                              particles[id.x].nearDensity += params.mass * nearDensityKernel(sqrt(r2));
                          }
                      }
                  }
              }
          }


          // for (var j = 0u; j < n; j = j + 1) {
          //     let pos_j = particles[j].position;
          //     let r2 = dot(pos_i - pos_j, pos_i - pos_j);
          //     if (r2 < params.kernelRadiusPow2) {
          //         particles[id.x].density += params.mass * densityKernel(sqrt(r2));
          //         particles[id.x].nearDensity += params.mass * nearDensityKernel(sqrt(r2));
          //     }
          // }
      }
    }
  `,
    externals: {
        ...densityLayout.bound,
        nearDensityKernel,
        densityKernel,
        cellPosition,
        cellNumberFromId,
    },
});

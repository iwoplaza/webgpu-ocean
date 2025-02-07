import tgpu from "typegpu";
import { ParticleArray } from "./shared";
import { PosVelArray } from "../common";

export const copyPositionLayout = tgpu
  .bindGroupLayout({
    particles: { storage: ParticleArray, access: "readonly" },
    posvel: { storage: PosVelArray, access: "mutable" },
  })
  .$idx(0);

export const copyPositionShader = tgpu.resolve({
  template: /* wgsl */ `
    @compute @workgroup_size(64)
    fn copyPosition(@builtin(global_invocation_id) id: vec3<u32>) {
      if (id.x < arrayLength(&_EXT_.particles)) { // 変える
        _EXT_.posvel[id.x].position = _EXT_.particles[id.x].position;
        _EXT_.posvel[id.x].v = _EXT_.particles[id.x].v;
      }
    }
  `,
  externals: { _EXT_: copyPositionLayout.bound },
});

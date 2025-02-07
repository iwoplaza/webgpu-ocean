import tgpu from "typegpu";
import { ParticleArray } from "./shared";
import { PosVelArray } from "../common";
import { builtin } from "typegpu/data";

export const copyPositionLayout = tgpu
  .bindGroupLayout({
    particles: { storage: ParticleArray, access: "readonly" },
    posvel: { storage: PosVelArray, access: "mutable" },
  })
  .$idx(0);

export const copyPositionFn = tgpu["~unstable"]
  .computeFn(
    {
      gid: builtin.globalInvocationId,
    },
    { workgroupSize: [64] },
  )
  .does(
    /* wgsl */ `(input: Input) {
      if (input.gid.x < arrayLength(&particles)) { // 変える
        posvel[input.gid.x].position = particles[input.gid.x].position;
        posvel[input.gid.x].v = particles[input.gid.x].v;
      }
    }`,
  )
  .$uses({ ...copyPositionLayout.bound });

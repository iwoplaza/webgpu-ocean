import tgpu from "typegpu";
import { builtin } from "typegpu/data";
import { CellArray } from "./shared";

export const clearGridLayout = tgpu
  .bindGroupLayout({
    cells: { storage: CellArray, access: "mutable" },
  })
  .$idx(0);

export const clearGridFn = tgpu["~unstable"]
  .computeFn({ gid: builtin.globalInvocationId }, { workgroupSize: [64] })
  .does(
    `(input: Input) {
      if (input.gid.x < arrayLength(&cells)) {
        cells[input.gid.x].mass = 0;
        cells[input.gid.x].vx = 0;
        cells[input.gid.x].vy = 0;
        cells[input.gid.x].vz = 0;
      }
    }`,
  )
  .$uses({ ...clearGridLayout.bound });

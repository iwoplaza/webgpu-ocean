import tgpu from "typegpu";
import { CellArray } from "./shared";

export const clearGridLayout = tgpu
  .bindGroupLayout({
    cells: { storage: CellArray, access: "mutable" },
  })
  .$idx(0);

export const clearGridShader = tgpu.resolve({
  template: /* wgsl */ `
    @compute @workgroup_size(64)
    fn clearGrid(@builtin(global_invocation_id) id: vec3<u32>) {
      if (id.x < arrayLength(&_EXT_.cells)) {
        _EXT_.cells[id.x].mass = 0;
        _EXT_.cells[id.x].vx = 0;
        _EXT_.cells[id.x].vy = 0;
        _EXT_.cells[id.x].vz = 0;
      }
    }
  `,
  externals: { _EXT_: clearGridLayout.bound },
});

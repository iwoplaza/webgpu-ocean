import tgpu from "typegpu";
import { ParticleArray, SPHParams } from "./shared";
import { f32, struct } from "typegpu/data";

export const RealBoxSize = struct({
  xHalf: f32,
  yHalf: f32,
  zHalf: f32,
});

export const integrateLayout = tgpu
  .bindGroupLayout({
    particles: { storage: ParticleArray, access: "mutable" },
    realBoxSize: { uniform: RealBoxSize },
    params: { uniform: SPHParams },
  })
  .$idx(0);

export const integrateShader = tgpu.resolve({
  template: /* wgsl */ `
  @compute @workgroup_size(64)
  fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < _EXT_.params.n) {
      // avoid zero division
      if (_EXT_.particles[id.x].density != 0.) {
        var a = _EXT_.particles[id.x].force / _EXT_.particles[id.x].density;

        let xPlusDist = _EXT_.realBoxSize.xHalf - _EXT_.particles[id.x].position.x;
        let xMinusDist = _EXT_.realBoxSize.xHalf + _EXT_.particles[id.x].position.x;
        let yPlusDist = _EXT_.realBoxSize.yHalf - _EXT_.particles[id.x].position.y;
        let yMinusDist = _EXT_.realBoxSize.yHalf + _EXT_.particles[id.x].position.y;
        let zPlusDist = _EXT_.realBoxSize.zHalf - _EXT_.particles[id.x].position.z;
        let zMinusDist = _EXT_.realBoxSize.zHalf + _EXT_.particles[id.x].position.z;

        let wallStiffness = 8000.;

        let xPlusForce = vec3f(1., 0., 0.) * wallStiffness * min(xPlusDist, 0.);
        let xMinusForce = vec3f(-1., 0., 0.) * wallStiffness * min(xMinusDist, 0.);
        let yPlusForce = vec3f(0., 1., 0.) * wallStiffness * min(yPlusDist, 0.);
        let yMinusForce = vec3f(0., -1., 0.) * wallStiffness * min(yMinusDist, 0.);
        let zPlusForce = vec3f(0., 0., 1.) * wallStiffness * min(zPlusDist, 0.);
        let zMinusForce = vec3f(0., 0., -1.) * wallStiffness * min(zMinusDist, 0.);

        let xForce = xPlusForce + xMinusForce;
        let yForce = yPlusForce + yMinusForce;
        let zForce = zPlusForce + zMinusForce;

        a += xForce + yForce + zForce;
        _EXT_.particles[id.x].v += _EXT_.params.dt * a;
        _EXT_.particles[id.x].position += _EXT_.params.dt * _EXT_.particles[id.x].v;
      }
    }
  }`,
  externals: { _EXT_: integrateLayout.bound },
});

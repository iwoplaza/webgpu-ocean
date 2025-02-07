import tgpu, { TgpuBindGroup, TgpuBuffer, TgpuRoot, Uniform } from "typegpu";
import { arrayOf, f32u32, u32 } from "typegpu/data";
import { PrefixSumKernel } from "webgpu-radix-sort";

import gridClear from "./grid/gridClear.wgsl";
import gridBuild from "./grid/gridBuild.wgsl";
import reorderParticles from "./grid/reorderParticles.wgsl";
import { densityShader } from "./density";
import force from "./force.wgsl";
import { integrateShader, integrateLayout, RealBoxSize } from "./integrate";
import { copyPositionShader, copyPositionLayout } from "./copyPosition";

import { renderUniformsViews, numParticlesMax } from "../common";
import { Environment, SPHParams } from "./shared";

export const sphParticleStructSize = 64;

const gridClearLayout = tgpu
  .bindGroupLayout({
    cellParticleCount: {
      storage: (n: number) => arrayOf(u32, n),
      access: "mutable",
    },
  })
  .$idx(0);

export class SPHSimulator {
  device: GPUDevice;

  gridClearPipeline: GPUComputePipeline;
  gridBuildPipeline: GPUComputePipeline;
  reorderPipeline: GPUComputePipeline;
  densityPipeline: GPUComputePipeline;
  forcePipeline: GPUComputePipeline;
  integratePipeline: GPUComputePipeline;
  copyPositionPipeline: GPUComputePipeline;

  gridClearBindGroup: TgpuBindGroup<(typeof gridClearLayout)["entries"]>;
  gridBuildBindGroup: GPUBindGroup;
  reorderBindGroup: GPUBindGroup;
  densityBindGroup: GPUBindGroup;
  forceBindGroup: GPUBindGroup;
  integrateBindGroup: TgpuBindGroup<(typeof integrateLayout)["entries"]>;
  copyPositionBindGroup: TgpuBindGroup<(typeof copyPositionLayout)["entries"]>;

  cellParticleCountBuffer: GPUBuffer;
  particleBuffer: GPUBuffer;
  realBoxSizeBuffer: TgpuBuffer<typeof RealBoxSize> & Uniform;
  sphParamsBuffer: TgpuBuffer<typeof SPHParams> & Uniform;

  prefixSumKernel: any;

  kernelRadius = 0.07;
  numParticles = 0;
  gridCount = 0;

  renderDiameter: number;

  constructor(
    particleBuffer: GPUBuffer,
    posvelBuffer: GPUBuffer,
    renderDiameter: number,
    private root: TgpuRoot,
  ) {
    const device = root.device;
    this.device = device;
    this.renderDiameter = renderDiameter;
    const densityModule = device.createShaderModule({ code: densityShader });
    const forceModule = device.createShaderModule({ code: force });
    const integrateModule = device.createShaderModule({
      code: integrateShader,
    });
    const gridBuildModule = device.createShaderModule({ code: gridBuild });
    const gridClearModule = device.createShaderModule({ code: gridClear });
    const reorderParticlesModule = device.createShaderModule({
      code: reorderParticles,
    });
    const copyPositionModule = device.createShaderModule({
      code: copyPositionShader,
    });

    const cellSize = 1.0 * this.kernelRadius;
    const xHalfMax = 2.0;
    const yHalfMax = 2.0;
    const zHalfMax = 2.0;
    const xLen = 2.0 * xHalfMax;
    const yLen = 2.0 * yHalfMax;
    const zLen = 2.0 * zHalfMax;
    const sentinel = 4 * cellSize;
    const xGrids = Math.ceil((xLen + sentinel) / cellSize);
    const yGrids = Math.ceil((yLen + sentinel) / cellSize);
    const zGrids = Math.ceil((zLen + sentinel) / cellSize);
    this.gridCount = xGrids * yGrids * zGrids;
    const offset = sentinel / 2;

    const stiffness = 20;
    const nearStiffness = 1.0;
    const mass = 1.0;
    const restDensity = 15000;
    const viscosity = 100;
    const dt = 0.006;

    this.gridClearPipeline = device.createComputePipeline({
      label: "grid clear pipeline",
      layout: device.createPipelineLayout({
        bindGroupLayouts: [root.unwrap(gridClearLayout)],
      }),
      compute: {
        module: gridClearModule,
      },
    });
    this.gridBuildPipeline = device.createComputePipeline({
      label: "grid build pipeline",
      layout: "auto",
      compute: {
        module: gridBuildModule,
      },
    });
    this.reorderPipeline = device.createComputePipeline({
      label: "reorder pipeline",
      layout: "auto",
      compute: {
        module: reorderParticlesModule,
      },
    });
    this.densityPipeline = device.createComputePipeline({
      label: "density pipeline",
      layout: "auto",
      compute: {
        module: densityModule,
      },
    });
    this.forcePipeline = device.createComputePipeline({
      label: "force pipeline",
      layout: "auto",
      compute: {
        module: forceModule,
      },
    });
    this.integratePipeline = device.createComputePipeline({
      label: "integrate pipeline",
      layout: device.createPipelineLayout({
        bindGroupLayouts: [root.unwrap(integrateLayout)],
      }),
      compute: {
        module: integrateModule,
      },
    });
    this.copyPositionPipeline = device.createComputePipeline({
      label: "copy position pipeline",
      layout: device.createPipelineLayout({
        bindGroupLayouts: [root.unwrap(copyPositionLayout)],
      }),
      compute: {
        module: copyPositionModule,
      },
    });

    this.cellParticleCountBuffer = device.createBuffer({
      // 累積和はここに保存
      label: "cell particle count buffer",
      size: 4 * (this.gridCount + 1), // 1 要素余分にとっておく
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const targetParticlesBuffer = device.createBuffer({
      label: "target particles buffer",
      size: sphParticleStructSize * numParticlesMax,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const particleCellOffsetBuffer = device.createBuffer({
      label: "particle cell offset buffer",
      size: 4 * numParticlesMax,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.realBoxSizeBuffer = root
      .createBuffer(RealBoxSize)
      .$usage("uniform")
      .$name("real box size buffer");
    const environmentBuffer = root
      .createBuffer(Environment, {
        xGrids,
        yGrids,
        zGrids,
        cellSize,
        xHalf: xHalfMax,
        yHalf: yHalfMax,
        zHalf: zHalfMax,
        offset,
      })
      .$usage("uniform")
      .$name("environment buffer");
    this.sphParamsBuffer = root
      .createBuffer(SPHParams, {
        mass,
        kernelRadius: this.kernelRadius,
        kernelRadiusPow2: this.kernelRadius ** 2,
        kernelRadiusPow5: this.kernelRadius ** 5,
        kernelRadiusPow6: this.kernelRadius ** 6,
        kernelRadiusPow9: this.kernelRadius ** 9,
        dt,
        stiffness,
        nearStiffness,
        restDensity,
        viscosity,
        n: 0, // n はあとで
      })
      .$usage("uniform")
      .$name("sph params buffer");

    // BindGroup
    this.gridClearBindGroup = root.createBindGroup(gridClearLayout, {
      cellParticleCount: this.cellParticleCountBuffer,
    });
    this.gridBuildBindGroup = device.createBindGroup({
      layout: this.gridBuildPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.cellParticleCountBuffer } },
        { binding: 1, resource: { buffer: particleCellOffsetBuffer } },
        { binding: 2, resource: { buffer: particleBuffer } },
        { binding: 3, resource: { buffer: root.unwrap(environmentBuffer) } },
        { binding: 4, resource: { buffer: root.unwrap(this.sphParamsBuffer) } },
      ],
    });
    this.reorderBindGroup = device.createBindGroup({
      layout: this.reorderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: particleBuffer } },
        { binding: 1, resource: { buffer: targetParticlesBuffer } },
        { binding: 2, resource: { buffer: this.cellParticleCountBuffer } },
        { binding: 3, resource: { buffer: particleCellOffsetBuffer } },
        { binding: 4, resource: { buffer: root.unwrap(environmentBuffer) } },
        { binding: 5, resource: { buffer: root.unwrap(this.sphParamsBuffer) } },
      ],
    });

    this.densityBindGroup = device.createBindGroup({
      layout: this.densityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: particleBuffer } },
        { binding: 1, resource: { buffer: targetParticlesBuffer } },
        { binding: 2, resource: { buffer: this.cellParticleCountBuffer } },
        { binding: 3, resource: { buffer: root.unwrap(environmentBuffer) } },
        { binding: 4, resource: { buffer: root.unwrap(this.sphParamsBuffer) } },
      ],
    });
    this.forceBindGroup = device.createBindGroup({
      layout: this.forcePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: particleBuffer } },
        { binding: 1, resource: { buffer: targetParticlesBuffer } },
        { binding: 2, resource: { buffer: this.cellParticleCountBuffer } },
        { binding: 3, resource: { buffer: root.unwrap(environmentBuffer) } },
        { binding: 4, resource: { buffer: root.unwrap(this.sphParamsBuffer) } },
      ],
    });
    this.integrateBindGroup = root.createBindGroup(integrateLayout, {
      params: this.sphParamsBuffer,
      particles: particleBuffer,
      realBoxSize: this.realBoxSizeBuffer,
    });
    this.copyPositionBindGroup = root.createBindGroup(copyPositionLayout, {
      particles: particleBuffer,
      posvel: posvelBuffer,
      env: this.sphParamsBuffer,
    });

    this.particleBuffer = particleBuffer;
  }

  reset(numParticles: number, initHalfBoxSize: number[]) {
    renderUniformsViews.sphere_size.set([this.renderDiameter]);
    const particleData = this.initDambreak(initHalfBoxSize, numParticles);
    const numParticleValue = new Float32Array(1);
    numParticleValue[0] = this.numParticles;
    console.log(this.numParticles);
    this.device.queue.writeBuffer(
      this.root.unwrap(this.sphParamsBuffer),
      44,
      numParticleValue,
    ); // TODO : avoid hardcoding
    this.device.queue.writeBuffer(this.particleBuffer, 0, particleData);
    this.realBoxSizeBuffer.write({
      xHalf: initHalfBoxSize[0],
      yHalf: initHalfBoxSize[1],
      zHalf: initHalfBoxSize[2],
    });
  }

  execute(commandEncoder: GPUCommandEncoder) {
    const computePass = commandEncoder.beginComputePass();
    for (let i = 0; i < 2; i++) {
      computePass.setBindGroup(0, this.root.unwrap(this.gridClearBindGroup));
      computePass.setPipeline(this.gridClearPipeline);
      computePass.dispatchWorkgroups(Math.ceil((this.gridCount + 1) / 64));
      computePass.setBindGroup(0, this.gridBuildBindGroup);
      computePass.setPipeline(this.gridBuildPipeline);
      computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
      this.prefixSumKernel = new PrefixSumKernel({
        device: this.device,
        data: this.cellParticleCountBuffer,
        count: this.gridCount + 1,
      });
      this.prefixSumKernel.dispatch(computePass);
      computePass.setBindGroup(0, this.reorderBindGroup);
      computePass.setPipeline(this.reorderPipeline);
      computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));

      computePass.setBindGroup(0, this.densityBindGroup);
      computePass.setPipeline(this.densityPipeline);
      computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
      computePass.setBindGroup(0, this.reorderBindGroup);
      computePass.setPipeline(this.reorderPipeline);
      computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
      computePass.setBindGroup(0, this.forceBindGroup);
      computePass.setPipeline(this.forcePipeline);
      computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
      computePass.setBindGroup(0, this.root.unwrap(this.integrateBindGroup));
      computePass.setPipeline(this.integratePipeline);
      computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
      computePass.setBindGroup(0, this.root.unwrap(this.copyPositionBindGroup));
      computePass.setPipeline(this.copyPositionPipeline);
      computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
    }

    computePass.end();
  }

  initDambreak(initHalfBoxSize: number[], numParticles: number) {
    let particlesBuf = new ArrayBuffer(sphParticleStructSize * numParticles);
    this.numParticles = 0;
    const DIST_FACTOR = 0.5;

    for (
      var y = -initHalfBoxSize[1] * 0.95;
      this.numParticles < numParticles;
      y += DIST_FACTOR * this.kernelRadius
    ) {
      for (
        var x = -0.95 * initHalfBoxSize[0];
        x < 0.95 * initHalfBoxSize[0] && this.numParticles < numParticles;
        x += DIST_FACTOR * this.kernelRadius
      ) {
        for (
          var z = -0.95 * initHalfBoxSize[2];
          z < 0 * initHalfBoxSize[2] && this.numParticles < numParticles;
          z += DIST_FACTOR * this.kernelRadius
        ) {
          let jitter = 0.001 * Math.random();
          const offset = sphParticleStructSize * this.numParticles;
          const particleViews = {
            position: new Float32Array(particlesBuf, offset + 0, 3),
            v: new Float32Array(particlesBuf, offset + 16, 3),
            force: new Float32Array(particlesBuf, offset + 32, 3),
            density: new Float32Array(particlesBuf, offset + 44, 1),
            nearDensity: new Float32Array(particlesBuf, offset + 48, 1),
          };
          particleViews.position.set([x + jitter, y + jitter, z + jitter]);
          this.numParticles++;
        }
      }
    }

    console.log(this.numParticles);
    return particlesBuf;
  }

  changeBoxSize(realBoxSize: number[]) {
    this.realBoxSizeBuffer.write({
      xHalf: realBoxSize[0],
      yHalf: realBoxSize[1],
      zHalf: realBoxSize[2],
    });
  }
}

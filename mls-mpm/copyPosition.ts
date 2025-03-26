import tgpu from 'typegpu';
import { ParticleArray } from './shared';
import { PosVelArray } from '../common';
import { builtin } from 'typegpu/data';
import { arrayLength } from 'typegpu/std';

export const copyPositionLayout = tgpu.bindGroupLayout({
    particles: { storage: ParticleArray, access: 'readonly' },
    posvel: { storage: PosVelArray, access: 'mutable' },
});

const { particles, posvel } = copyPositionLayout.bound;

export const copyPositionFn = tgpu['~unstable']
    .computeFn({
        workgroupSize: [64],
        in: {
            gid: builtin.globalInvocationId,
        },
    })
    .does((input) => {
        if (input.gid.x < arrayLength(particles.value)) {
            // 変える
            posvel.value[input.gid.x].position =
                particles.value[input.gid.x].position;
            posvel.value[input.gid.x].v = particles.value[input.gid.x].v;
        }
    });

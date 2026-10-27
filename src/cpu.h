/*
 * Copyright © 2026, Sungjoon Moon
 * All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FCVVDP_CPU_H
#define FCVVDP_CPU_H

enum CvvdpCpuFlags {
    CVVDP_CPU_FLAG_AVX2 = 1 << 0, // AVX2 + FMA3
    CVVDP_CPU_FLAG_NEON = 1 << 1,
};

extern unsigned cvvdp_cpu_flags;
extern unsigned cvvdp_cpu_flags_mask;

void cvvdp_init_cpu(void);
void cvvdp_set_cpu_flags_mask(unsigned mask);

static inline unsigned cvvdp_get_cpu_flags(void) {
    return cvvdp_cpu_flags & cvvdp_cpu_flags_mask;
}

#endif /* FCVVDP_CPU_H */

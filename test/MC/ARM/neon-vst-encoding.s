@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7-apple-darwin -show-encoding < %s | FileCheck %s

	vst1.8	{d16}, [r0, :64]
	vst1.16	{d16}, [r0]
	vst1.32	{d16}, [r0]
	vst1.64	{d16}, [r0]
	vst1.8	{d16, d17}, [r0, :64]
	vst1.16	{d16, d17}, [r0, :128]
	vst1.32	{d16, d17}, [r0]
	vst1.64	{d16, d17}, [r0]
        vst1.8  {d16, d17, d18}, [r0, :64]
        vst1.8  {d16, d17, d18}, [r0, :64]!
        vst1.8  {d16, d17, d18}, [r0], r3
        vst1.8  {d16, d17, d18, d19}, [r0, :64]
        vst1.16  {d16, d17, d18, d19}, [r1, :64]!
        vst1.64  {d16, d17, d18, d19}, [r3], r2

@ CHECK: vst1.8	{d16}, [r0, :64]        @ encoding: [0x1f,0x07,0x40,0xf4]
@ CHECK: vst1.16 {d16}, [r0]            @ encoding: [0x4f,0x07,0x40,0xf4]
@ CHECK: vst1.32 {d16}, [r0]            @ encoding: [0x8f,0x07,0x40,0xf4]
@ CHECK: vst1.64 {d16}, [r0]            @ encoding: [0xcf,0x07,0x40,0xf4]
@ CHECK: vst1.8	{d16, d17}, [r0, :64]   @ encoding: [0x1f,0x0a,0x40,0xf4]
@ CHECK: vst1.16 {d16, d17}, [r0, :128] @ encoding: [0x6f,0x0a,0x40,0xf4]
@ CHECK: vst1.32 {d16, d17}, [r0]       @ encoding: [0x8f,0x0a,0x40,0xf4]
@ CHECK: vst1.64 {d16, d17}, [r0]       @ encoding: [0xcf,0x0a,0x40,0xf4]
@ CHECK: vst1.8	{d16, d17, d18}, [r0, :64] @ encoding: [0x1f,0x06,0x40,0xf4]
@ CHECK: vst1.8	{d16, d17, d18}, [r0, :64]! @ encoding: [0x1d,0x06,0x40,0xf4]
@ CHECK: vst1.8	{d16, d17, d18}, [r0], r3 @ encoding: [0x03,0x06,0x40,0xf4]
@ CHECK: vst1.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x02,0x40,0xf4]
@ CHECK: vst1.16 {d16, d17, d18, d19}, [r1, :64]! @ encoding: [0x5d,0x02,0x41,0xf4]
@ CHECK: vst1.64 {d16, d17, d18, d19}, [r3], r2 @ encoding: [0xc2,0x02,0x43,0xf4]


	vst2.8	{d16, d17}, [r0, :64]
	vst2.16	{d16, d17}, [r0, :128]
	vst2.32	{d16, d17}, [r0]
	vst2.8	{d16, d17, d18, d19}, [r0, :64]
	vst2.16	{d16, d17, d18, d19}, [r0, :128]
	vst2.32	{d16, d17, d18, d19}, [r0, :256]
	vst2.8	{d16, d17}, [r0, :64]!
	vst2.16	{q15}, [r0, :128]!
	vst2.32	{d14, d15}, [r0]!
	vst2.8	{d16, d17, d18, d19}, [r0, :64]!
	vst2.16	{d18-d21}, [r0, :128]!
	vst2.32	{q4, q5}, [r0, :256]!

@ CHECK: vst2.8	{d16, d17}, [r0, :64]   @ encoding: [0x1f,0x08,0x40,0xf4]
@ CHECK: vst2.16 {d16, d17}, [r0, :128] @ encoding: [0x6f,0x08,0x40,0xf4]
@ CHECK: vst2.32 {d16, d17}, [r0]       @ encoding: [0x8f,0x08,0x40,0xf4]
@ CHECK: vst2.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x03,0x40,0xf4]
@ CHECK: vst2.16 {d16, d17, d18, d19}, [r0, :128] @ encoding: [0x6f,0x03,0x40,0xf4]
@ CHECK: vst2.32 {d16, d17, d18, d19}, [r0, :256] @ encoding: [0xbf,0x03,0x40,0xf4]
@ CHECK: vst2.8	{d16, d17}, [r0, :64]!  @ encoding: [0x1d,0x08,0x40,0xf4]
@ CHECK: vst2.16	{d30, d31}, [r0, :128]! @ encoding: [0x6d,0xe8,0x40,0xf4]
@ CHECK: vst2.32	{d14, d15}, [r0]!       @ encoding: [0x8d,0xe8,0x00,0xf4]
@ CHECK: vst2.8	{d16, d17, d18, d19}, [r0, :64]! @ encoding: [0x1d,0x03,0x40,0xf4]
@ CHECK: vst2.16	{d18, d19, d20, d21}, [r0, :128]! @ encoding: [0x6d,0x23,0x40,0xf4]
@ CHECK: vst2.32	{d8, d9, d10, d11}, [r0, :256]! @ encoding: [0xbd,0x83,0x00,0xf4]


	vst3.8 {d16, d17, d18}, [r1]
	vst3.16 {d6, d7, d8}, [r2]
	vst3.32 {d1, d2, d3}, [r3]
	vst3.8 {d16, d18, d20}, [r0, :64]
	vst3.u16 {d27, d29, d31}, [r4]
	vst3.i32 {d6, d8, d10}, [r5]

	vst3.i8 {d12, d13, d14}, [r6], r1
	vst3.i16 {d11, d12, d13}, [r7], r2
	vst3.u32 {d2, d3, d4}, [r8], r3
	vst3.8 {d4, d6, d8}, [r9], r4
	vst3.u16 {d14, d16, d18}, [r9], r4
	vst3.i32 {d16, d18, d20}, [r10], r5

	vst3.p8 {d6, d7, d8}, [r8]!
	vst3.16 {d9, d10, d11}, [r7]!
	vst3.f32 {d1, d2, d3}, [r6]!
	vst3.8 {d16, d18, d20}, [r0, :64]!
	vst3.p16 {d20, d22, d24}, [r5]!
	vst3.32 {d5, d7, d9}, [r4]!

@ CHECK: vst3.8	{d16, d17, d18}, [r1]   @ encoding: [0x0f,0x04,0x41,0xf4]
@ CHECK: vst3.16	{d6, d7, d8}, [r2]      @ encoding: [0x4f,0x64,0x02,0xf4]
@ CHECK: vst3.32	{d1, d2, d3}, [r3]      @ encoding: [0x8f,0x14,0x03,0xf4]
@ CHECK: vst3.8	{d16, d18, d20}, [r0, :64] @ encoding: [0x1f,0x05,0x40,0xf4]
@ CHECK: vst3.16	{d27, d29, d31}, [r4]   @ encoding: [0x4f,0xb5,0x44,0xf4]
@ CHECK: vst3.32	{d6, d8, d10}, [r5]     @ encoding: [0x8f,0x65,0x05,0xf4]
@ CHECK: vst3.8	{d12, d13, d14}, [r6], r1 @ encoding: [0x01,0xc4,0x06,0xf4]
@ CHECK: vst3.16	{d11, d12, d13}, [r7], r2 @ encoding: [0x42,0xb4,0x07,0xf4]
@ CHECK: vst3.32	{d2, d3, d4}, [r8], r3  @ encoding: [0x83,0x24,0x08,0xf4]
@ CHECK: vst3.8	{d4, d6, d8}, [r9], r4  @ encoding: [0x04,0x45,0x09,0xf4]
@ CHECK: vst3.16	{d14, d16, d18}, [r9], r4 @ encoding: [0x44,0xe5,0x09,0xf4]
@ CHECK: vst3.32	{d16, d18, d20}, [r10], r5 @ encoding: [0x85,0x05,0x4a,0xf4]
@ CHECK: vst3.8	{d6, d7, d8}, [r8]!     @ encoding: [0x0d,0x64,0x08,0xf4]
@ CHECK: vst3.16	{d9, d10, d11}, [r7]!   @ encoding: [0x4d,0x94,0x07,0xf4]
@ CHECK: vst3.32	{d1, d2, d3}, [r6]!     @ encoding: [0x8d,0x14,0x06,0xf4]
@ CHECK: vst3.8	{d16, d18, d20}, [r0, :64]! @ encoding: [0x1d,0x05,0x40,0xf4]
@ CHECK: vst3.16	{d20, d22, d24}, [r5]!  @ encoding: [0x4d,0x45,0x45,0xf4]
@ CHECK: vst3.32	{d5, d7, d9}, [r4]!     @ encoding: [0x8d,0x55,0x04,0xf4]


@	vst4.8	{d16, d17, d18, d19}, [r0, :64]
@	vst4.16	{d16, d17, d18, d19}, [r0, :128]
@	vst4.8	{d16, d18, d20, d22}, [r0, :256]!
@	vst4.8	{d17, d19, d21, d23}, [r0, :256]!
@	vst4.16	{d16, d18, d20, d22}, [r0]!
@	vst4.16	{d17, d19, d21, d23}, [r0]!
@	vst4.32	{d16, d18, d20, d22}, [r0]!
@	vst4.32	{d17, d19, d21, d23}, [r0]!

@ FIXME: vst4.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x00,0x40,0xf4]
@ FIXME: vst4.16 {d16, d17, d18, d19}, [r0, :128] @ encoding: [0x6f,0x00,0x40,0xf4]
@ FIXME: vst4.8	{d16, d18, d20, d22}, [r0, :256]! @ encoding: [0x3d,0x01,0x40,0xf4]
@ FIXME: vst4.8	{d17, d19, d21, d23}, [r0, :256]! @ encoding: [0x3d,0x11,0x40,0xf4]
@ FIXME: vst4.16 {d16, d18, d20, d22}, [r0]! @ encoding: [0x4d,0x01,0x40,0xf4]
@ FIXME: vst4.16 {d17, d19, d21, d23}, [r0]! @ encoding: [0x4d,0x11,0x40,0xf4]
@ FIXME: vst4.32 {d16, d18, d20, d22}, [r0]! @ encoding: [0x8d,0x01,0x40,0xf4]
@ FIXME: vst4.32 {d17, d19, d21, d23}, [r0]! @ encoding: [0x8d,0x11,0x40,0xf4]


	vst2.8	{d16[1], d17[1]}, [r0, :16]
	vst2.p16	{d16[1], d17[1]}, [r0, :32]
	vst2.i32	{d16[1], d17[1]}, [r0]
	vst2.u16	{d17[1], d19[1]}, [r0]
	vst2.f32	{d17[0], d19[0]}, [r0, :64]

        vst2.8 {d2[4], d3[4]}, [r2], r3
        vst2.u8 {d2[4], d3[4]}, [r2]!
        vst2.p8 {d2[4], d3[4]}, [r2]

        vst2.16 {d17[1], d19[1]}, [r0]
        vst2.32 {d17[0], d19[0]}, [r0, :64]
        vst2.i16 {d7[1], d9[1]}, [r1]!
        vst2.32 {d6[0], d8[0]}, [r2, :64]!
        vst2.16 {d2[1], d4[1]}, [r3], r5
        vst2.u32 {d5[0], d7[0]}, [r4, :64], r7

@ CHECK: vst2.8	{d16[1], d17[1]}, [r0, :16] @ encoding: [0x3f,0x01,0xc0,0xf4]
@ CHECK: vst2.16 {d16[1], d17[1]}, [r0, :32] @ encoding: [0x5f,0x05,0xc0,0xf4]
@ CHECK: vst2.32 {d16[1], d17[1]}, [r0]  @ encoding: [0x8f,0x09,0xc0,0xf4]
@ CHECK: vst2.16 {d17[1], d19[1]}, [r0]  @ encoding: [0x6f,0x15,0xc0,0xf4]
@ CHECK: vst2.32 {d17[0], d19[0]}, [r0, :64] @ encoding: [0x5f,0x19,0xc0,0xf4]

@ CHECK: vst2.8	{d2[4], d3[4]}, [r2], r3 @ encoding: [0x83,0x21,0x82,0xf4]
@ CHECK: vst2.8	{d2[4], d3[4]}, [r2]!   @ encoding: [0x8d,0x21,0x82,0xf4]
@ CHECK: vst2.8	{d2[4], d3[4]}, [r2]    @ encoding: [0x8f,0x21,0x82,0xf4]

@ CHECK: vst2.16 {d17[1], d19[1]}, [r0]  @ encoding: [0x6f,0x15,0xc0,0xf4]
@ CHECK: vst2.32 {d17[0], d19[0]}, [r0, :64] @ encoding: [0x5f,0x19,0xc0,0xf4]
@ CHECK: vst2.16 {d7[1], d9[1]}, [r1]!   @ encoding: [0x6d,0x75,0x81,0xf4]
@ CHECK: vst2.32 {d6[0], d8[0]}, [r2, :64]! @ encoding: [0x5d,0x69,0x82,0xf4]
@ CHECK: vst2.16 {d2[1], d4[1]}, [r3], r5 @ encoding: [0x65,0x25,0x83,0xf4]
@ CHECK: vst2.32 {d5[0], d7[0]}, [r4, :64], r7 @ encoding: [0x57,0x59,0x84,0xf4]


@	vst4.8	{d16[1], d17[1], d18[1], d19[1]}, [r0, :32]
@	vst4.16	{d16[1], d17[1], d18[1], d19[1]}, [r0]
@	vst4.32	{d16[1], d17[1], d18[1], d19[1]}, [r0, :128]
@	vst4.16	{d17[3], d19[3], d21[3], d23[3]}, [r0, :64]
@	vst4.32	{d17[0], d19[0], d21[0], d23[0]}, [r0]

@ FIXME: vst4.8	{d16[1], d17[1], d18[1], d19[1]}, [r0, :32] @ encoding: [0x3f,0x03,0xc0,0xf4]
@ FIXME: vst4.16 {d16[1], d17[1], d18[1], d19[1]}, [r0] @ encoding: [0x4f,0x07,0xc0,0xf4]
@ FIXME: vst4.32 {d16[1], d17[1], d18[1], d19[1]}, [r0, :128] @ encoding: [0xaf,0x0b,0xc0,0xf4]
@ FIXME: vst4.16 {d17[3], d19[3], d21[3], d23[3]}, [r0, :64] @ encoding: [0xff,0x17,0xc0,0xf4]
@ FIXME: vst4.32 {d17[0], d19[0], d21[0], d23[0]}, [r0] @ encoding: [0x4f,0x1b,0xc0,0xf4]


@ Spot-check additional size-suffix aliases.

        vst1.8 {d2}, [r2]
        vst1.p8 {d2}, [r2]
        vst1.u8 {d2}, [r2]

        vst1.8 {q2}, [r2]
        vst1.p8 {q2}, [r2]
        vst1.u8 {q2}, [r2]
        vst1.f32 {q2}, [r2]

@ CHECK: vst1.8	{d2}, [r2]              @ encoding: [0x0f,0x27,0x02,0xf4]
@ CHECK: vst1.8	{d2}, [r2]              @ encoding: [0x0f,0x27,0x02,0xf4]
@ CHECK: vst1.8	{d2}, [r2]              @ encoding: [0x0f,0x27,0x02,0xf4]

@ CHECK: vst1.8	{d4, d5}, [r2]          @ encoding: [0x0f,0x4a,0x02,0xf4]
@ CHECK: vst1.8	{d4, d5}, [r2]          @ encoding: [0x0f,0x4a,0x02,0xf4]
@ CHECK: vst1.8	{d4, d5}, [r2]          @ encoding: [0x0f,0x4a,0x02,0xf4]
@ CHECK: vst1.32 {d4, d5}, [r2]         @ encoding: [0x8f,0x4a,0x02,0xf4]

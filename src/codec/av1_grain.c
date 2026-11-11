/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * AV1 film grain synthesis, written from the AV1 specification, section
 * 7.18.3 ("Film grain synthesis process"): the random number process, the
 * grain templates with their auto-regressive filter, the scaling lookup and
 * the noise added in 32-row stripes of 32x32 blocks.
 *
 * Section 7.18.3's noiseStripe and noiseImage are never stored: each row of
 * noiseImage is put together when its picture row is reached, from the
 * templates at the offsets of its stripe's blocks (and, for the first rows of
 * a stripe, of the stripe above), which gives the same samples. The noise is
 * then added in passes over the row that vectorise: the scaling values looked
 * up per sample (the only gather), then the multiply, rounding and clipping.
 * On AArch64 these passes are NEON; elsewhere plain C for the compiler.
 *
 * The decoded picture (the reference, without grain) is read from one
 * buffer and the picture with grain written into another, each row once and
 * whole (the bytes past the picture copied), with memcpy() through a row
 * buffer: a CAPTURE buffer of a decoder that is not DMA-coherent (Apple's
 * AVD) may be mapped uncached, where a load per sample would cost a bus read
 * each.
 */

#include <stdlib.h>
#include <string.h>

/* NEON on AArch64; AV1_GRAIN_NO_NEON (tests) takes the C passes there. */
#if defined(__aarch64__) && !defined(AV1_GRAIN_NO_NEON)
#define GRAIN_NEON 1
#include <arm_neon.h>
#endif

#include "av1_grain.h"

/* The grain templates: LumaGrain is 82 x 73, CbGrain and CrGrain at most. */
#define GRAIN_W		82
#define GRAIN_H		73

/* Room past each row buffer: the vector passes run in whole vectors of up
 * to 32 samples and may read and write past the row's end. */
#define ROW_PAD		128

/* Gaussian_Sequence (the spec's additional tables, section 10). */
static const int16_t gaussian_sequence[2048] = {
	56, 568, -180, 172, 124, -84, 172, -64, -900, 24, 820, 224,
	1248, 996, 272, -8, -916, -388, -732, -104, -188, 800, 112, -652,
	-320, -376, 140, -252, 492, -168, 44, -788, 588, -584, 500, -228,
	12, 680, 272, -476, 972, -100, 652, 368, 432, -196, -720, -192,
	1000, -332, 652, -136, -552, -604, -4, 192, -220, -136, 1000, -52,
	372, -96, -624, 124, -24, 396, 540, -12, -104, 640, 464, 244,
	-208, -84, 368, -528, -740, 248, -968, -848, 608, 376, -60, -292,
	-40, -156, 252, -292, 248, 224, -280, 400, -244, 244, -60, 76,
	-80, 212, 532, 340, 128, -36, 824, -352, -60, -264, -96, -612,
	416, -704, 220, -204, 640, -160, 1220, -408, 900, 336, 20, -336,
	-96, -792, 304, 48, -28, -1232, -1172, -448, 104, -292, -520, 244,
	60, -948, 0, -708, 268, 108, 356, -548, 488, -344, -136, 488,
	-196, -224, 656, -236, -1128, 60, 4, 140, 276, -676, -376, 168,
	-108, 464, 8, 564, 64, 240, 308, -300, -400, -456, -136, 56,
	120, -408, -116, 436, 504, -232, 328, 844, -164, -84, 784, -168,
	232, -224, 348, -376, 128, 568, 96, -1244, -288, 276, 848, 832,
	-360, 656, 464, -384, -332, -356, 728, -388, 160, -192, 468, 296,
	224, 140, -776, -100, 280, 4, 196, 44, -36, -648, 932, 16,
	1428, 28, 528, 808, 772, 20, 268, 88, -332, -284, 124, -384,
	-448, 208, -228, -1044, -328, 660, 380, -148, -300, 588, 240, 540,
	28, 136, -88, -436, 256, 296, -1000, 1400, 0, -48, 1056, -136,
	264, -528, -1108, 632, -484, -592, -344, 796, 124, -668, -768, 388,
	1296, -232, -188, -200, -288, -4, 308, 100, -168, 256, -500, 204,
	-508, 648, -136, 372, -272, -120, -1004, -552, -548, -384, 548, -296,
	428, -108, -8, -912, -324, -224, -88, -112, -220, -100, 996, -796,
	548, 360, -216, 180, 428, -200, -212, 148, 96, 148, 284, 216,
	-412, -320, 120, -300, -384, -604, -572, -332, -8, -180, -176, 696,
	116, -88, 628, 76, 44, -516, 240, -208, -40, 100, -592, 344,
	-308, -452, -228, 20, 916, -1752, -136, -340, -804, 140, 40, 512,
	340, 248, 184, -492, 896, -156, 932, -628, 328, -688, -448, -616,
	-752, -100, 560, -1020, 180, -800, -64, 76, 576, 1068, 396, 660,
	552, -108, -28, 320, -628, 312, -92, -92, -472, 268, 16, 560,
	516, -672, -52, 492, -100, 260, 384, 284, 292, 304, -148, 88,
	-152, 1012, 1064, -228, 164, -376, -684, 592, -392, 156, 196, -524,
	-64, -884, 160, -176, 636, 648, 404, -396, -436, 864, 424, -728,
	988, -604, 904, -592, 296, -224, 536, -176, -920, 436, -48, 1176,
	-884, 416, -776, -824, -884, 524, -548, -564, -68, -164, -96, 692,
	364, -692, -1012, -68, 260, -480, 876, -1116, 452, -332, -352, 892,
	-1088, 1220, -676, 12, -292, 244, 496, 372, -32, 280, 200, 112,
	-440, -96, 24, -644, -184, 56, -432, 224, -980, 272, -260, 144,
	-436, 420, 356, 364, -528, 76, 172, -744, -368, 404, -752, -416,
	684, -688, 72, 540, 416, 92, 444, 480, -72, -1416, 164, -1172,
	-68, 24, 424, 264, 1040, 128, -912, -524, -356, 64, 876, -12,
	4, -88, 532, 272, -524, 320, 276, -508, 940, 24, -400, -120,
	756, 60, 236, -412, 100, 376, -484, 400, -100, -740, -108, -260,
	328, -268, 224, -200, -416, 184, -604, -564, -20, 296, 60, 892,
	-888, 60, 164, 68, -760, 216, -296, 904, -336, -28, 404, -356,
	-568, -208, -1480, -512, 296, 328, -360, -164, -1560, -776, 1156, -428,
	164, -504, -112, 120, -216, -148, -264, 308, 32, 64, -72, 72,
	116, 176, -64, -272, 460, -536, -784, -280, 348, 108, -752, -132,
	524, -540, -776, 116, -296, -1196, -288, -560, 1040, -472, 116, -848,
	-1116, 116, 636, 696, 284, -176, 1016, 204, -864, -648, -248, 356,
	972, -584, -204, 264, 880, 528, -24, -184, 116, 448, -144, 828,
	524, 212, -212, 52, 12, 200, 268, -488, -404, -880, 824, -672,
	-40, 908, -248, 500, 716, -576, 492, -576, 16, 720, -108, 384,
	124, 344, 280, 576, -500, 252, 104, -308, 196, -188, -8, 1268,
	296, 1032, -1196, 436, 316, 372, -432, -200, -660, 704, -224, 596,
	-132, 268, 32, -452, 884, 104, -1008, 424, -1348, -280, 4, -1168,
	368, 476, 696, 300, -8, 24, 180, -592, -196, 388, 304, 500,
	724, -160, 244, -84, 272, -256, -420, 320, 208, -144, -156, 156,
	364, 452, 28, 540, 316, 220, -644, -248, 464, 72, 360, 32,
	-388, 496, -680, -48, 208, -116, -408, 60, -604, -392, 548, -840,
	784, -460, 656, -544, -388, -264, 908, -800, -628, -612, -568, 572,
	-220, 164, 288, -16, -308, 308, -112, -636, -760, 280, -668, 432,
	364, 240, -196, 604, 340, 384, 196, 592, -44, -500, 432, -580,
	-132, 636, -76, 392, 4, -412, 540, 508, 328, -356, -36, 16,
	-220, -64, -248, -60, 24, -192, 368, 1040, 92, -24, -1044, -32,
	40, 104, 148, 192, -136, -520, 56, -816, -224, 732, 392, 356,
	212, -80, -424, -1008, -324, 588, -1496, 576, 460, -816, -848, 56,
	-580, -92, -1372, -112, -496, 200, 364, 52, -140, 48, -48, -60,
	84, 72, 40, 132, -356, -268, -104, -284, -404, 732, -520, 164,
	-304, -540, 120, 328, -76, -460, 756, 388, 588, 236, -436, -72,
	-176, -404, -316, -148, 716, -604, 404, -72, -88, -888, -68, 944,
	88, -220, -344, 960, 472, 460, -232, 704, 120, 832, -228, 692,
	-508, 132, -476, 844, -748, -364, -44, 1116, -1104, -1056, 76, 428,
	552, -692, 60, 356, 96, -384, -188, -612, -576, 736, 508, 892,
	352, -1132, 504, -24, -352, 324, 332, -600, -312, 292, 508, -144,
	-8, 484, 48, 284, -260, -240, 256, -100, -292, -204, -44, 472,
	-204, 908, -188, -1000, -256, 92, 1164, -392, 564, 356, 652, -28,
	-884, 256, 484, -192, 760, -176, 376, -524, -452, -436, 860, -736,
	212, 124, 504, -476, 468, 76, -472, 552, -692, -944, -620, 740,
	-240, 400, 132, 20, 192, -196, 264, -668, -1012, -60, 296, -316,
	-828, 76, -156, 284, -768, -448, -832, 148, 248, 652, 616, 1236,
	288, -328, -400, -124, 588, 220, 520, -696, 1032, 768, -740, -92,
	-272, 296, 448, -464, 412, -200, 392, 440, -200, 264, -152, -260,
	320, 1032, 216, 320, -8, -64, 156, -1016, 1084, 1172, 536, 484,
	-432, 132, 372, -52, -256, 84, 116, -352, 48, 116, 304, -384,
	412, 924, -300, 528, 628, 180, 648, 44, -980, -220, 1320, 48,
	332, 748, 524, -268, -720, 540, -276, 564, -344, -208, -196, 436,
	896, 88, -392, 132, 80, -964, -288, 568, 56, -48, -456, 888,
	8, 552, -156, -292, 948, 288, 128, -716, -292, 1192, -152, 876,
	352, -600, -260, -812, -468, -28, -120, -32, -44, 1284, 496, 192,
	464, 312, -76, -516, -380, -456, -1012, -48, 308, -156, 36, 492,
	-156, -808, 188, 1652, 68, -120, -116, 316, 160, -140, 352, 808,
	-416, 592, 316, -480, 56, 528, -204, -568, 372, -232, 752, -344,
	744, -4, 324, -416, -600, 768, 268, -248, -88, -132, -420, -432,
	80, -288, 404, -316, -1216, -588, 520, -108, 92, -320, 368, -480,
	-216, -92, 1688, -300, 180, 1020, -176, 820, -68, -228, -260, 436,
	-904, 20, 40, -508, 440, -736, 312, 332, 204, 760, -372, 728,
	96, -20, -632, -520, -560, 336, 1076, -64, -532, 776, 584, 192,
	396, -728, -520, 276, -188, 80, -52, -612, -252, -48, 648, 212,
	-688, 228, -52, -260, 428, -412, -272, -404, 180, 816, -796, 48,
	152, 484, -88, -216, 988, 696, 188, -528, 648, -116, -180, 316,
	476, 12, -564, 96, 476, -252, -364, -376, -392, 556, -256, -576,
	260, -352, 120, -16, -136, -260, -492, 72, 556, 660, 580, 616,
	772, 436, 424, -32, -324, -1268, 416, -324, -80, 920, 160, 228,
	724, 32, -516, 64, 384, 68, -128, 136, 240, 248, -204, -68,
	252, -932, -120, -480, -628, -84, 192, 852, -404, -288, -132, 204,
	100, 168, -68, -196, -868, 460, 1080, 380, -80, 244, 0, 484,
	-888, 64, 184, 352, 600, 460, 164, 604, -196, 320, -64, 588,
	-184, 228, 12, 372, 48, -848, -344, 224, 208, -200, 484, 128,
	-20, 272, -468, -840, 384, 256, -720, -520, -464, -580, 112, -120,
	644, -356, -208, -608, -528, 704, 560, -424, 392, 828, 40, 84,
	200, -152, 0, -144, 584, 280, -120, 80, -556, -972, -196, -472,
	724, 80, 168, -32, 88, 160, -688, 0, 160, 356, 372, -776,
	740, -128, 676, -248, -480, 4, -364, 96, 544, 232, -1032, 956,
	236, 356, 20, -40, 300, 24, -676, -596, 132, 1120, -104, 532,
	-1096, 568, 648, 444, 508, 380, 188, -376, -604, 1488, 424, 24,
	756, -220, -192, 716, 120, 920, 688, 168, 44, -460, 568, 284,
	1144, 1160, 600, 424, 888, 656, -356, -320, 220, 316, -176, -724,
	-188, -816, -628, -348, -228, -380, 1012, -452, -660, 736, 928, 404,
	-696, -72, -268, -892, 128, 184, -344, -780, 360, 336, 400, 344,
	428, 548, -112, 136, -228, -216, -820, -516, 340, 92, -136, 116,
	-300, 376, -244, 100, -316, -520, -284, -12, 824, 164, -548, -180,
	-128, 116, -924, -828, 268, -368, -580, 620, 192, 160, 0, -1676,
	1068, 424, -56, -360, 468, -156, 720, 288, -528, 556, -364, 548,
	-148, 504, 316, 152, -648, -620, -684, -24, -376, -384, -108, -920,
	-1032, 768, 180, -264, -508, -1268, -260, -60, 300, -240, 988, 724,
	-376, -576, -212, -736, 556, 192, 1092, -620, -880, 376, -56, -4,
	-216, -32, 836, 268, 396, 1332, 864, -600, 100, 56, -412, -92,
	356, 180, 884, -468, -436, 292, -388, -804, -704, -840, 368, -348,
	140, -724, 1536, 940, 372, 112, -372, 436, -480, 1136, 296, -32,
	-228, 132, -48, -220, 868, -1016, -60, -1044, -464, 328, 916, 244,
	12, -736, -296, 360, 468, -376, -108, -92, 788, 368, -56, 544,
	400, -672, -420, 728, 16, 320, 44, -284, -380, -796, 488, 132,
	204, -596, -372, 88, -152, -908, -636, -572, -624, -116, -692, -200,
	-56, 276, -88, 484, -324, 948, 864, 1000, -456, -184, -276, 292,
	-296, 156, 676, 320, 160, 908, -84, -1236, -288, -116, 260, -372,
	-644, 732, -756, -96, 84, 344, -520, 348, -688, 240, -84, 216,
	-1044, -136, -676, -396, -1500, 960, -40, 176, 168, 1516, 420, -504,
	-344, -364, -360, 1216, -940, -380, -212, 252, -660, -708, 484, -444,
	-152, 928, -120, 1112, 476, -260, 560, -148, -344, 108, -196, 228,
	-288, 504, 560, -328, -88, 288, -1008, 460, -228, 468, -836, -196,
	76, 388, 232, 412, -1168, -716, -644, 756, -172, -356, -504, 116,
	432, 528, 48, 476, -168, -608, 448, 160, -532, -272, 28, -676,
	-12, 828, 980, 456, 520, 104, -104, 256, -344, -4, -28, -368,
	-52, -524, -572, -556, -200, 768, 1124, -208, -512, 176, 232, 248,
	-148, -888, 604, -600, -304, 804, -156, -212, 488, -192, -804, -256,
	368, -360, -916, -328, 228, -240, -448, -472, 856, -556, -364, 572,
	-12, -156, -368, -340, 432, 252, -752, -152, 288, 268, -580, -848,
	-592, 108, -76, 244, 312, -716, 592, -80, 436, 360, 4, -248,
	160, 516, 584, 732, 44, -468, -280, -292, -156, -588, 28, 308,
	912, 24, 124, 156, 180, -252, 944, -924, -772, -520, -428, -624,
	300, -212, -1144, 32, -724, 800, -1128, -212, -1288, -848, 180, -416,
	440, 192, -576, -792, -76, -1080, 80, -532, -352, -132, 380, -820,
	148, 1112, 128, 164, 456, 700, -924, 144, -668, -384, 648, -832,
	508, 552, -52, -100, -656, 208, -568, 748, -88, 680, 232, 300,
	192, -408, -1012, -152, -252, -268, 272, -876, -664, -648, -332, -136,
	16, 12, 1152, -28, 332, -536, 320, -672, -460, -316, 532, -260,
	228, -40, 1052, -816, 180, 88, -496, -556, -672, -368, 428, 92,
	356, 404, -408, 252, 196, -176, -556, 792, 268, 32, 372, 40,
	96, -332, 328, 120, 372, -900, -40, 472, -264, -592, 952, 128,
	656, 112, 664, -232, 420, 4, -344, -464, 556, 244, -416, -32,
	252, 0, -412, 188, -696, 508, -476, 324, -1096, 656, -312, 560,
	264, -136, 304, 160, -64, -580, 248, 336, -720, 560, -348, -288,
	-276, -196, -500, 852, -544, -236, -1128, -992, -776, 116, 56, 52,
	860, 884, 212, -12, 168, 1020, 512, -552, 924, -148, 716, 188,
	164, -340, -520, -184, 880, -152, -680, -208, -1156, -300, -528, -472,
	364, 100, -744, -1056, -32, 540, 280, 144, -676, -32, -232, -280,
	-224, 96, 568, -76, 172, 148, 148, 104, 32, -296, -32, 788,
	-80, 32, -16, 280, 288, 944, 428, -484,
};

struct av1_grain_state {
	/* RandomRegister */
	uint16_t random;

	int16_t luma_grain[GRAIN_H][GRAIN_W];
	int16_t cb_grain[GRAIN_H][GRAIN_W];
	int16_t cr_grain[GRAIN_H][GRAIN_W];

	/* scale_lut( plane, index ) at every sample value of the bit depth;
	 * it lies between two of the points' 8-bit scaling values */
	uint8_t scale[3][1024];

	/*
	 * The random offset of each 32x32 block (the 8 bits of section
	 * 7.18.3.5: offsetX in the high nibble, offsetY in the low), for the
	 * current stripe and the stripe above.
	 */
	uint8_t *offsets[2];

	/* A row of noiseImage per plane, and a row of the stripe above's
	 * noiseStripe for the overlap between stripes. */
	int16_t *noise[3];
	int16_t *above;

	/* The blend's values per sample: the scaling looked up (luma, or Cb
	 * and Cr), chroma's average luma and its merged values (the index
	 * into the scaling lookup). */
	uint8_t *scaling[2];
	uint16_t *average;
	uint16_t *merged[2];

	/* A row of the source and of the result, the whole pitch. */
	uint8_t *luma_in, *luma_out, *chroma_in, *chroma_out;

	uint8_t *buffer;
	size_t buffer_size;
};

/* Round2() and Clip3() of the spec's conventions (section 4.7); >> of a
 * negative value is arithmetic, as GCC and Clang define it. */
static inline int round2(int x, unsigned int n)
{
	if (!n)
		return x;
	return (x + (1 << (n - 1))) >> n;
}

static inline int clip3(int low, int high, int x)
{
	return x < low ? low : x > high ? high : x;
}

/* The random number process, section 7.18.3.2. */
static unsigned int get_random_number(struct av1_grain_state *state,
				      unsigned int bits)
{
	unsigned int r = state->random;
	unsigned int bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;

	r = (r >> 1) | (bit << 15);
	state->random = r;

	return (r >> (16 - bits)) & ((1u << bits) - 1);
}

struct av1_grain_state *av1_grain_state_new(void)
{
	return calloc(1, sizeof(struct av1_grain_state));
}

void av1_grain_state_free(struct av1_grain_state *state)
{
	if (!state)
		return;

	free(state->buffer);
	free(state);
}

static bool points_increase(const uint8_t *value, unsigned int count)
{
	for (unsigned int i = 1; i < count; i++)
		if (value[i] <= value[i - 1])
			return false;

	return true;
}

bool av1_grain_supported(const struct av1_grain_params *p)
{
	if (p->bit_depth != 8 && p->bit_depth != 10)
		return false;
	if (p->subsampling_x != 1 || p->subsampling_y != 1)
		return false;
	if (p->num_y_points > 14 || p->num_cb_points > 10 ||
	    p->num_cr_points > 10)
		return false;
	if (p->ar_coeff_lag > 3 || p->grain_scaling_minus_8 > 3 ||
	    p->ar_coeff_shift_minus_6 > 3 || p->grain_scale_shift > 3)
		return false;

	/* The scaling lookup divides by the distance between two points:
	 * the spec requires each point's value above the previous one. */
	if (!points_increase(p->point_y_value, p->num_y_points))
		return false;
	if (!p->mono_chrome && !p->chroma_scaling_from_luma &&
	    (!points_increase(p->point_cb_value, p->num_cb_points) ||
	     !points_increase(p->point_cr_value, p->num_cr_points)))
		return false;

	return true;
}

/*
 * The auto-regressive filter of section 7.18.3.3 over one template, rows
 * and cols its size. The taps on the rows above are final when a row
 * starts, so they are summed for the whole row first (a loop that
 * vectorises); the row's own taps, which depend on the samples just
 * filtered, follow in order. With luma (chroma's filter), the last
 * coefficient weighs the luma grain at the same position.
 */
static void ar_filter(int16_t (*g)[GRAIN_W], int rows, int cols,
		      const int8_t *coeffs, int lag, unsigned int ar_shift,
		      const int16_t (*luma)[GRAIN_W], unsigned int subx,
		      unsigned int suby, int grain_min, int grain_max)
{
	int span = 2 * lag + 1;
	const int8_t *own = coeffs + lag * span;
	int sum[GRAIN_W];

	for (int y = 3; y < rows; y++) {
		int16_t *row = g[y];

		for (int x = 3; x < cols - 3; x++)
			sum[x] = 0;
		for (int dr = -lag; dr < 0; dr++) {
			for (int dc = -lag; dc <= lag; dc++) {
				int c = coeffs[(dr + lag) * span + dc + lag];
				const int16_t *r = g[y + dr] + dc;

				for (int x = 3; x < cols - 3; x++)
					sum[x] += r[x] * c;
			}
		}
		if (luma) {
			int c = own[lag];
			const int16_t *l0 = luma[((y - 3) << suby) + 3];
			const int16_t *l1 = luma[((y - 3) << suby) + 3 + suby];

			for (int x = 3; x < cols - 3; x++) {
				int lx = ((x - 3) << subx) + 3;
				int l = l0[lx];

				if (subx)
					l += l0[lx + 1];
				if (suby)
					l += l1[lx] + (subx ? l1[lx + 1] : 0);
				sum[x] += round2(l, subx + suby) * c;
			}
		}

		for (int x = 3; x < cols - 3; x++) {
			int s = sum[x];

			for (int dc = -lag; dc < 0; dc++)
				s += row[x + dc] * own[dc + lag];
			row[x] = clip3(grain_min, grain_max, row[x] + round2(s, ar_shift));
		}
	}
}

/* The generate grain process, section 7.18.3.3. */
static void generate_grain(struct av1_grain_state *state,
			   const struct av1_grain_params *p,
			   int grain_min, int grain_max)
{
	unsigned int shift = 12 - p->bit_depth + p->grain_scale_shift;
	unsigned int ar_shift = p->ar_coeff_shift_minus_6 + 6;
	int lag = p->ar_coeff_lag;
	unsigned int subx = p->subsampling_x, suby = p->subsampling_y;
	int chroma_w, chroma_h;

	/* White noise for luma, then the auto-regressive filter over it
	 * (which leaves a template of zeros as it is). */
	if (p->num_y_points > 0) {
		state->random = p->grain_seed;
		for (int y = 0; y < GRAIN_H; y++)
			for (int x = 0; x < GRAIN_W; x++)
				state->luma_grain[y][x] =
					round2(gaussian_sequence[get_random_number(state, 11)],
					       shift);
		ar_filter(state->luma_grain, GRAIN_H, GRAIN_W, p->ar_coeffs_y, lag,
			  ar_shift, NULL, 0, 0, grain_min, grain_max);
	} else {
		memset(state->luma_grain, 0, sizeof(state->luma_grain));
	}

	if (p->mono_chrome)
		return;

	chroma_w = subx ? 44 : 82;
	chroma_h = suby ? 38 : 73;

	for (unsigned int plane = 1; plane < 3; plane++) {
		int16_t (*g)[GRAIN_W] = plane == 1 ? state->cb_grain : state->cr_grain;
		unsigned int points = plane == 1 ? p->num_cb_points : p->num_cr_points;

		state->random = p->grain_seed ^ (plane == 1 ? 0xb524 : 0x49d8);
		for (int y = 0; y < chroma_h; y++) {
			for (int x = 0; x < chroma_w; x++) {
				int r = 0;

				if (points > 0 || p->chroma_scaling_from_luma)
					r = gaussian_sequence[get_random_number(state, 11)];
				g[y][x] = round2(r, shift);
			}
		}
		ar_filter(g, chroma_h, chroma_w,
			  plane == 1 ? p->ar_coeffs_cb : p->ar_coeffs_cr, lag, ar_shift,
			  p->num_y_points > 0 ? (const int16_t (*)[GRAIN_W])state->luma_grain : NULL,
			  subx, suby, grain_min, grain_max);
	}
}

/* The scaling lookup initialization process, section 7.18.3.4, then
 * scale_lut() (section 7.18.3.5) tabulated for every sample value. */
static void init_scaling(struct av1_grain_state *state,
			 const struct av1_grain_params *p, unsigned int planes)
{
	unsigned int shift = p->bit_depth - 8;
	unsigned int values = 1u << p->bit_depth;

	for (unsigned int plane = 0; plane < planes; plane++) {
		const uint8_t *px, *py;
		unsigned int n;
		uint8_t lut[256];

		if (plane == 0 || p->chroma_scaling_from_luma) {
			px = p->point_y_value;
			py = p->point_y_scaling;
			n = p->num_y_points;
		} else if (plane == 1) {
			px = p->point_cb_value;
			py = p->point_cb_scaling;
			n = p->num_cb_points;
		} else {
			px = p->point_cr_value;
			py = p->point_cr_scaling;
			n = p->num_cr_points;
		}

		if (n == 0) {
			memset(lut, 0, sizeof(lut));
		} else {
			for (unsigned int x = 0; x < px[0]; x++)
				lut[x] = py[0];
			for (unsigned int i = 0; i + 1 < n; i++) {
				int delta_y = py[i + 1] - py[i];
				int delta_x = px[i + 1] - px[i];
				int delta = delta_y * ((65536 + (delta_x >> 1)) / delta_x);

				for (int x = 0; x < delta_x; x++)
					lut[px[i] + x] = py[i] + ((x * delta + 32768) >> 16);
			}
			for (unsigned int x = px[n - 1]; x < 256; x++)
				lut[x] = py[n - 1];
		}

		for (unsigned int index = 0; index < values; index++) {
			unsigned int x = index >> shift;
			unsigned int rem = index - (x << shift);

			if (p->bit_depth == 8 || x == 255) {
				state->scale[plane][index] = lut[x];
			} else {
				int start = lut[x], end = lut[x + 1];

				state->scale[plane][index] =
					start + round2((end - start) * (int)rem, shift);
			}
		}
	}
}

/* The row buffers, carved out of one allocation in 64-byte steps. */
static int alloc_rows(struct av1_grain_state *state, unsigned int blocks,
		      size_t luma_pitch, size_t chroma_pitch)
{
	/* A noise row holds a whole number of blocks (32 * blocks luma
	 * samples, at least the width), with room for the vector passes. */
	size_t wide = 32 * (size_t)blocks + ROW_PAD;
	size_t sizes[] = {
		blocks, blocks,					/* offsets */
		2 * wide, 2 * wide, 2 * wide, 2 * wide,		/* noise, above */
		wide, wide,					/* scaling */
		2 * wide, 2 * wide, 2 * wide,			/* average, merged */
		luma_pitch + ROW_PAD, luma_pitch + ROW_PAD,	/* luma_in, _out */
		chroma_pitch + ROW_PAD, chroma_pitch + ROW_PAD,	/* chroma_in, _out */
	};
	enum { N = sizeof(sizes) / sizeof(sizes[0]) };
	uint8_t *at[N];
	size_t total = 0;

	for (unsigned int i = 0; i < N; i++)
		total += (sizes[i] + 63) & ~(size_t)63;

	if (total > state->buffer_size) {
		void *buffer;

		if (posix_memalign(&buffer, 64, total))
			return -1;
		free(state->buffer);
		state->buffer = buffer;
		state->buffer_size = total;
		/* The vector passes read past a row's end, never using what
		 * they read there: zeroed once, so that it is defined. */
		memset(buffer, 0, total);
	}

	total = 0;
	for (unsigned int i = 0; i < N; i++) {
		at[i] = state->buffer + total;
		total += (sizes[i] + 63) & ~(size_t)63;
	}
	state->offsets[0] = at[0];
	state->offsets[1] = at[1];
	state->noise[0] = (int16_t *)at[2];
	state->noise[1] = (int16_t *)at[3];
	state->noise[2] = (int16_t *)at[4];
	state->above = (int16_t *)at[5];
	state->scaling[0] = at[6];
	state->scaling[1] = at[7];
	state->average = (uint16_t *)at[8];
	state->merged[0] = (uint16_t *)at[9];
	state->merged[1] = (uint16_t *)at[10];
	state->luma_in = at[11];
	state->luma_out = at[12];
	state->chroma_in = at[13];
	state->chroma_out = at[14];

	return 0;
}

/* Each block's offset in stripe luma_num (section 7.18.3.5). */
static void stripe_offsets(struct av1_grain_state *state,
			   const struct av1_grain_params *p, unsigned int luma_num,
			   uint8_t *offsets, unsigned int blocks)
{
	state->random = p->grain_seed;
	state->random ^= ((luma_num * 37 + 178) & 255) << 8;
	state->random ^= ((luma_num * 173 + 105) & 255);

	for (unsigned int block = 0; block < blocks; block++)
		offsets[block] = get_random_number(state, 8);
}

/*
 * Row r of noiseStripe for one plane (section 7.18.3.5), from its blocks'
 * offsets: each block's row of the template, and with overlap_flag the
 * first columns of a block blended with the two (luma) or one (chroma)
 * columns the block on its left wrote past its edge.
 */
static inline __attribute__((always_inline)) void
stripe_row(int16_t *out, const int16_t (*grain)[GRAIN_W],
	   const uint8_t *offsets, unsigned int blocks, unsigned int r,
	   bool chroma, bool overlap, int grain_min, int grain_max)
{
	const unsigned int bw = chroma ? 16 : 32;

	for (unsigned int block = 0; block < blocks; block++) {
		unsigned int ox = offsets[block] >> 4, oy = offsets[block] & 15;
		const int16_t *g = chroma ? &grain[6 + oy + r][6 + ox] :
					    &grain[9 + 2 * oy + r][9 + 2 * ox];

		memcpy(out + block * bw, g, bw * sizeof(*g));
	}

	if (!overlap)
		return;

	for (unsigned int block = 1; block < blocks; block++) {
		unsigned int ox = offsets[block - 1] >> 4, oy = offsets[block - 1] & 15;
		const int16_t *left = chroma ? &grain[6 + oy + r][6 + ox + 16] :
					       &grain[9 + 2 * oy + r][9 + 2 * ox + 32];
		int16_t *o = out + block * bw;

		if (chroma) {
			o[0] = clip3(grain_min, grain_max, round2(left[0] * 23 + o[0] * 22, 5));
		} else {
			o[0] = clip3(grain_min, grain_max, round2(left[0] * 27 + o[0] * 17, 5));
			o[1] = clip3(grain_min, grain_max, round2(left[1] * 17 + o[1] * 27, 5));
		}
	}
}

/* The overlap between stripes: a row of noiseImage from the stripe above's
 * row (old) and this stripe's (cur). */
static void blend_rows(int16_t *restrict cur, const int16_t *restrict old,
		       unsigned int n, int w_old, int w_cur, int grain_min,
		       int grain_max)
{
	for (unsigned int x = 0; x < n; x++)
		cur[x] = clip3(grain_min, grain_max,
			       round2(old[x] * w_old + cur[x] * w_cur, 5));
}

/* What the blend needs of the parameters. */
struct blend {
	unsigned int shift;		/* ScalingShift */
	int max_sample;
	int min_luma, max_luma;
	/* Cb's and Cr's clipping range: the whole range for a plane without
	 * noise, whose scaling is 0 and samples then stay as they are. */
	int min_chroma[2], max_chroma[2];
	/* (cb|cr)_luma_mult - 128, (cb|cr)_mult - 128, the scaled offsets */
	int luma_mult[2], mult[2], offset[2];
	bool from_luma;			/* chroma_scaling_from_luma */
};

#ifndef GRAIN_NEON
/* The last step of section 7.18.3.5 for one luma row: luma_out from
 * luma_in and noise[ 0 ], w samples (and maybe some past them). */
static inline __attribute__((always_inline)) void
blend_luma_c(const struct av1_grain_state *state, const struct blend *b,
	     unsigned int w, bool p010)
{
	const int16_t *restrict noise = state->noise[0];
	const uint8_t *restrict scale = state->scale[0];
	uint8_t *restrict scaling = state->scaling[0];
	const unsigned int shift = b->shift;
	const int round = 1 << (shift - 1), lo = b->min_luma, hi = b->max_luma;

	if (p010) {
		const uint16_t *restrict in = (const uint16_t *)state->luma_in;
		uint16_t *restrict out = (uint16_t *)state->luma_out;

		for (unsigned int x = 0; x < w; x++)
			scaling[x] = scale[in[x] >> 6];
		for (unsigned int x = 0; x < w; x++) {
			int v = (in[x] >> 6) + ((scaling[x] * noise[x] + round) >> shift);

			out[x] = clip3(lo, hi, v) << 6;
		}
	} else {
		const uint8_t *restrict in = state->luma_in;
		uint8_t *restrict out = state->luma_out;

		for (unsigned int x = 0; x < w; x++)
			scaling[x] = scale[in[x]];
		for (unsigned int x = 0; x < w; x++) {
			int v = in[x] + ((scaling[x] * noise[x] + round) >> shift);

			out[x] = clip3(lo, hi, v);
		}
	}
}

/* The same for a row of the interleaved chroma (chroma_in to chroma_out,
 * cw = (w + 1) / 2 pairs), from the luma row it sits on as decoded
 * (luma_in, with the last sample of an odd width repeated past it). */
static inline __attribute__((always_inline)) void
blend_chroma_c(const struct av1_grain_state *state, const struct blend *b,
	       unsigned int w, bool p010)
{
	unsigned int cw = (w + 1) >> 1;
	uint16_t *restrict average = state->average;
	uint16_t *restrict m0 = state->merged[0], *restrict m1 = state->merged[1];
	uint8_t *restrict s0 = state->scaling[0], *restrict s1 = state->scaling[1];
	const int16_t *restrict n0 = state->noise[1], *restrict n1 = state->noise[2];
	const unsigned int shift = b->shift;
	const int round = 1 << (shift - 1), max = b->max_sample;
	const int lm0 = b->luma_mult[0], mu0 = b->mult[0], off0 = b->offset[0];
	const int lm1 = b->luma_mult[1], mu1 = b->mult[1], off1 = b->offset[1];
	const int lo0 = b->min_chroma[0], hi0 = b->max_chroma[0];
	const int lo1 = b->min_chroma[1], hi1 = b->max_chroma[1];

	if (p010) {
		const uint16_t *restrict luma = (const uint16_t *)state->luma_in;

		for (unsigned int x = 0; x < cw; x++)
			average[x] = ((luma[2 * x] >> 6) + (luma[2 * x + 1] >> 6) + 1) >> 1;
	} else {
		const uint8_t *restrict luma = state->luma_in;

		for (unsigned int x = 0; x < cw; x++)
			average[x] = (luma[2 * x] + luma[2 * x + 1] + 1) >> 1;
	}

	if (b->from_luma) {
		m0 = m1 = average;
	} else if (p010) {
		const uint16_t *restrict in = (const uint16_t *)state->chroma_in;

		for (unsigned int x = 0; x < cw; x++) {
			int a = average[x], u = in[2 * x] >> 6, v = in[2 * x + 1] >> 6;

			m0[x] = clip3(0, max, ((a * lm0 + u * mu0) >> 6) + off0);
			m1[x] = clip3(0, max, ((a * lm1 + v * mu1) >> 6) + off1);
		}
	} else {
		const uint8_t *restrict in = state->chroma_in;

		for (unsigned int x = 0; x < cw; x++) {
			int a = average[x], u = in[2 * x], v = in[2 * x + 1];

			m0[x] = clip3(0, max, ((a * lm0 + u * mu0) >> 6) + off0);
			m1[x] = clip3(0, max, ((a * lm1 + v * mu1) >> 6) + off1);
		}
	}

	for (unsigned int x = 0; x < cw; x++) {
		s0[x] = state->scale[1][m0[x]];
		s1[x] = state->scale[2][m1[x]];
	}

	if (p010) {
		const uint16_t *restrict in = (const uint16_t *)state->chroma_in;
		uint16_t *restrict out = (uint16_t *)state->chroma_out;

		for (unsigned int x = 0; x < cw; x++) {
			int u = (in[2 * x] >> 6) + ((s0[x] * n0[x] + round) >> shift);
			int v = (in[2 * x + 1] >> 6) + ((s1[x] * n1[x] + round) >> shift);

			out[2 * x] = clip3(lo0, hi0, u) << 6;
			out[2 * x + 1] = clip3(lo1, hi1, v) << 6;
		}
	} else {
		const uint8_t *restrict in = state->chroma_in;
		uint8_t *restrict out = state->chroma_out;

		for (unsigned int x = 0; x < cw; x++) {
			int u = in[2 * x] + ((s0[x] * n0[x] + round) >> shift);
			int v = in[2 * x + 1] + ((s1[x] * n1[x] + round) >> shift);

			out[2 * x] = clip3(lo0, hi0, u);
			out[2 * x + 1] = clip3(lo1, hi1, v);
		}
	}
}

#else /* GRAIN_NEON */
/*
 * NEON. Two identities carry the passes:
 *
 * - Round2(scale * noise, ScalingShift) is SQRDMULH(noise, scale <<
 *   (15 - ScalingShift)): that gives (2 * noise * (scale << k) + (1 << 15))
 *   >> 16 with k = 15 - ScalingShift, the same floor division, in 16-bit
 *   lanes (scale << k stays below 1 << 15, and no saturation happens).
 * - An 8-bit scaling lookup of 256 entries is four TBL/TBX of 64 bytes
 *   each, the table held in sixteen registers. Ten bits index 1024
 *   entries, looked up one at a time (a plain loop).
 */

/* scale << (15 - shift) for 8 lookups. */
static inline int16x8_t scale_factor(uint8x8_t scale, int16x8_t k)
{
	return vreinterpretq_s16_u16(vshlq_u16(vmovl_u8(scale), k));
}

/* 16 lookups in a 256-entry table. */
static inline uint8x16_t lookup256(const uint8x16x4_t t[4], uint8x16_t i)
{
	uint8x16_t k64 = vdupq_n_u8(64);
	uint8x16_t i1 = vsubq_u8(i, k64), i2 = vsubq_u8(i1, k64), i3 = vsubq_u8(i2, k64);
	uint8x16_t r = vqtbl4q_u8(t[0], i);

	r = vqtbx4q_u8(r, t[1], i1);
	r = vqtbx4q_u8(r, t[2], i2);
	return vqtbx4q_u8(r, t[3], i3);
}

static inline void load_table(uint8x16x4_t t[4], const uint8_t *table)
{
	for (int i = 0; i < 4; i++)
		t[i] = vld1q_u8_x4(table + 64 * i);
}

static void blend_luma_neon8(const struct av1_grain_state *state,
			     const struct blend *b, unsigned int w)
{
	const uint8_t *in = state->luma_in;
	uint8_t *out = state->luma_out;
	const int16_t *noise = state->noise[0];
	int16x8_t k = vdupq_n_s16(15 - b->shift);
	uint8x16_t lo = vdupq_n_u8(b->min_luma), hi = vdupq_n_u8(b->max_luma);
	uint8x16x4_t t[4];

	load_table(t, state->scale[0]);

	for (unsigned int x = 0; x < w; x += 16) {
		uint8x16_t s = vld1q_u8(in + x);
		uint8x16_t sc = lookup256(t, s);
		int16x8_t n0 = vqrdmulhq_s16(vld1q_s16(noise + x),
					     scale_factor(vget_low_u8(sc), k));
		int16x8_t n1 = vqrdmulhq_s16(vld1q_s16(noise + x + 8),
					     scale_factor(vget_high_u8(sc), k));
		/* sample + noise, exact in 16 bits */
		int16x8_t v0 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(n0),
							      vget_low_u8(s)));
		int16x8_t v1 = vreinterpretq_s16_u16(vaddw_high_u8(vreinterpretq_u16_s16(n1), s));
		uint8x16_t r = vcombine_u8(vqmovun_s16(v0), vqmovun_s16(v1));

		vst1q_u8(out + x, vminq_u8(vmaxq_u8(r, lo), hi));
	}
}

static void blend_luma_neon10(const struct av1_grain_state *state,
			      const struct blend *b, unsigned int w)
{
	const uint16_t *in = (const uint16_t *)state->luma_in;
	uint16_t *out = (uint16_t *)state->luma_out;
	const int16_t *noise = state->noise[0];
	const uint8_t *scale = state->scale[0];
	uint8_t *scaling = state->scaling[0];
	int16x8_t k = vdupq_n_s16(15 - b->shift);
	int16x8_t lo = vdupq_n_s16(b->min_luma), hi = vdupq_n_s16(b->max_luma);

	for (unsigned int x = 0; x < w; x++)
		scaling[x] = scale[in[x] >> 6];

	for (unsigned int x = 0; x < w; x += 8) {
		int16x8_t s = vreinterpretq_s16_u16(vshrq_n_u16(vld1q_u16(in + x), 6));
		int16x8_t n = vqrdmulhq_s16(vld1q_s16(noise + x),
					    scale_factor(vld1_u8(scaling + x), k));
		int16x8_t v = vminq_s16(vmaxq_s16(vaddq_s16(s, n), lo), hi);

		vst1q_u16(out + x, vshlq_n_u16(vreinterpretq_u16_s16(v), 6));
	}
}

/*
 * Chroma at 8 bits: the merged values (the scaling's index) of 16 pairs at
 * a time into scaling[ 0 ] and [ 1 ], each looked up there in its plane's
 * table, then the noise added.
 */
static void blend_chroma_neon8(const struct av1_grain_state *state,
			       const struct blend *b, unsigned int w)
{
	unsigned int cw = (w + 1) >> 1;
	const uint8_t *luma = state->luma_in;
	const uint8_t *in = state->chroma_in;
	uint8_t *out = state->chroma_out;
	uint8_t *s0 = state->scaling[0], *s1 = state->scaling[1];
	const int16_t *n0 = state->noise[1], *n1 = state->noise[2];
	int16x8_t k = vdupq_n_s16(15 - b->shift);
	uint8x16x4_t t[4];

	if (b->from_luma) {
		for (unsigned int x = 0; x < cw; x += 16) {
			uint8x16x2_t l = vld2q_u8(luma + 2 * x);

			vst1q_u8(s0 + x, vrhaddq_u8(l.val[0], l.val[1]));
		}
		/* Both planes look up the luma points' table. */
		load_table(t, state->scale[1]);
		for (unsigned int x = 0; x < cw; x += 16)
			vst1q_u8(s0 + x, lookup256(t, vld1q_u8(s0 + x)));
		s1 = s0;
	} else {
		int16x8_t lm0 = vdupq_n_s16(b->luma_mult[0]), m0 = vdupq_n_s16(b->mult[0]);
		int16x8_t lm1 = vdupq_n_s16(b->luma_mult[1]), m1 = vdupq_n_s16(b->mult[1]);
		int16x8_t o0 = vdupq_n_s16(b->offset[0]), o1 = vdupq_n_s16(b->offset[1]);

		for (unsigned int x = 0; x < cw; x += 16) {
			uint8x16x2_t l = vld2q_u8(luma + 2 * x);
			uint8x16x2_t uv = vld2q_u8(in + 2 * x);
			uint8x16_t a = vrhaddq_u8(l.val[0], l.val[1]);
			int16x8_t al = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a)));
			int16x8_t ah = vreinterpretq_s16_u16(vmovl_high_u8(a));
			int16x8_t ul = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(uv.val[0])));
			int16x8_t uh = vreinterpretq_s16_u16(vmovl_high_u8(uv.val[0]));
			int16x8_t vl = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(uv.val[1])));
			int16x8_t vh = vreinterpretq_s16_u16(vmovl_high_u8(uv.val[1]));
			/*
			 * (a * luma_mult + u * mult) >> 6: each product fits
			 * 16 bits (255 * 128), their sum does not; SHADD
			 * halves the sum without losing it, and >> 5 more
			 * floors as one >> 6 would.
			 */
			int16x8_t c0l = vhaddq_s16(vmulq_s16(al, lm0), vmulq_s16(ul, m0));
			int16x8_t c0h = vhaddq_s16(vmulq_s16(ah, lm0), vmulq_s16(uh, m0));
			int16x8_t c1l = vhaddq_s16(vmulq_s16(al, lm1), vmulq_s16(vl, m1));
			int16x8_t c1h = vhaddq_s16(vmulq_s16(ah, lm1), vmulq_s16(vh, m1));

			/* + offset, clipped to 0..255 by the narrowing */
			vst1q_u8(s0 + x, vcombine_u8(vqmovun_s16(vsraq_n_s16(o0, c0l, 5)),
						     vqmovun_s16(vsraq_n_s16(o0, c0h, 5))));
			vst1q_u8(s1 + x, vcombine_u8(vqmovun_s16(vsraq_n_s16(o1, c1l, 5)),
						     vqmovun_s16(vsraq_n_s16(o1, c1h, 5))));
		}
		load_table(t, state->scale[1]);
		for (unsigned int x = 0; x < cw; x += 16)
			vst1q_u8(s0 + x, lookup256(t, vld1q_u8(s0 + x)));
		load_table(t, state->scale[2]);
		for (unsigned int x = 0; x < cw; x += 16)
			vst1q_u8(s1 + x, lookup256(t, vld1q_u8(s1 + x)));
	}

	uint8x16_t lo0 = vdupq_n_u8(b->min_chroma[0]), hi0 = vdupq_n_u8(b->max_chroma[0]);
	uint8x16_t lo1 = vdupq_n_u8(b->min_chroma[1]), hi1 = vdupq_n_u8(b->max_chroma[1]);

	for (unsigned int x = 0; x < cw; x += 16) {
		uint8x16x2_t uv = vld2q_u8(in + 2 * x);
		uint8x16_t sc0 = vld1q_u8(s0 + x), sc1 = vld1q_u8(s1 + x);
		int16x8_t u0 = vqrdmulhq_s16(vld1q_s16(n0 + x), scale_factor(vget_low_u8(sc0), k));
		int16x8_t u1 = vqrdmulhq_s16(vld1q_s16(n0 + x + 8), scale_factor(vget_high_u8(sc0), k));
		int16x8_t v0 = vqrdmulhq_s16(vld1q_s16(n1 + x), scale_factor(vget_low_u8(sc1), k));
		int16x8_t v1 = vqrdmulhq_s16(vld1q_s16(n1 + x + 8), scale_factor(vget_high_u8(sc1), k));
		uint8x16x2_t r;

		u0 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(u0), vget_low_u8(uv.val[0])));
		u1 = vreinterpretq_s16_u16(vaddw_high_u8(vreinterpretq_u16_s16(u1), uv.val[0]));
		v0 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(v0), vget_low_u8(uv.val[1])));
		v1 = vreinterpretq_s16_u16(vaddw_high_u8(vreinterpretq_u16_s16(v1), uv.val[1]));
		r.val[0] = vminq_u8(vmaxq_u8(vcombine_u8(vqmovun_s16(u0), vqmovun_s16(u1)), lo0), hi0);
		r.val[1] = vminq_u8(vmaxq_u8(vcombine_u8(vqmovun_s16(v0), vqmovun_s16(v1)), lo1), hi1);
		vst2q_u8(out + 2 * x, r);
	}
}

/*
 * Chroma at 10 bits: the merged values of 8 pairs at a time (in 32 bits:
 * 1023 * 128 does not fit 16) into merged[ 0 ] and [ 1 ], looked up one at
 * a time, then the noise added.
 */
static void blend_chroma_neon10(const struct av1_grain_state *state,
				const struct blend *b, unsigned int w)
{
	unsigned int cw = (w + 1) >> 1;
	const uint16_t *luma = (const uint16_t *)state->luma_in;
	const uint16_t *in = (const uint16_t *)state->chroma_in;
	uint16_t *out = (uint16_t *)state->chroma_out;
	uint16_t *m0 = state->merged[0], *m1 = state->merged[1];
	uint8_t *s0 = state->scaling[0], *s1 = state->scaling[1];
	const int16_t *n0 = state->noise[1], *n1 = state->noise[2];
	int16x8_t k = vdupq_n_s16(15 - b->shift);

	if (b->from_luma) {
		for (unsigned int x = 0; x < cw; x += 8) {
			uint16x8x2_t l = vld2q_u16(luma + 2 * x);

			vst1q_u16(m0 + x, vrhaddq_u16(vshrq_n_u16(l.val[0], 6),
						      vshrq_n_u16(l.val[1], 6)));
		}
		for (unsigned int x = 0; x < cw; x++)
			s0[x] = state->scale[1][m0[x]];
		s1 = s0;
	} else {
		int16x4_t lm0 = vdup_n_s16(b->luma_mult[0]), mu0 = vdup_n_s16(b->mult[0]);
		int16x4_t lm1 = vdup_n_s16(b->luma_mult[1]), mu1 = vdup_n_s16(b->mult[1]);
		int16x8_t o0 = vdupq_n_s16(b->offset[0]), o1 = vdupq_n_s16(b->offset[1]);
		int16x8_t zero = vdupq_n_s16(0), max = vdupq_n_s16(b->max_sample);

		for (unsigned int x = 0; x < cw; x += 8) {
			uint16x8x2_t l = vld2q_u16(luma + 2 * x);
			uint16x8x2_t uv = vld2q_u16(in + 2 * x);
			int16x8_t a = vreinterpretq_s16_u16(vrhaddq_u16(vshrq_n_u16(l.val[0], 6),
									vshrq_n_u16(l.val[1], 6)));
			int16x8_t u = vreinterpretq_s16_u16(vshrq_n_u16(uv.val[0], 6));
			int16x8_t v = vreinterpretq_s16_u16(vshrq_n_u16(uv.val[1], 6));
			int32x4_t c0l = vmlal_s16(vmull_s16(vget_low_s16(a), lm0), vget_low_s16(u), mu0);
			int32x4_t c0h = vmlal_s16(vmull_s16(vget_high_s16(a), lm0), vget_high_s16(u), mu0);
			int32x4_t c1l = vmlal_s16(vmull_s16(vget_low_s16(a), lm1), vget_low_s16(v), mu1);
			int32x4_t c1h = vmlal_s16(vmull_s16(vget_high_s16(a), lm1), vget_high_s16(v), mu1);
			/* >> 6 fits 16 bits (1023 * 128 * 2 >> 6), then the
			 * offset and the clip to the sample range */
			int16x8_t r0 = vaddq_s16(vcombine_s16(vshrn_n_s32(c0l, 6), vshrn_n_s32(c0h, 6)), o0);
			int16x8_t r1 = vaddq_s16(vcombine_s16(vshrn_n_s32(c1l, 6), vshrn_n_s32(c1h, 6)), o1);

			vst1q_u16(m0 + x, vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(r0, zero), max)));
			vst1q_u16(m1 + x, vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(r1, zero), max)));
		}
		for (unsigned int x = 0; x < cw; x++) {
			s0[x] = state->scale[1][m0[x]];
			s1[x] = state->scale[2][m1[x]];
		}
	}

	int16x8_t lo0 = vdupq_n_s16(b->min_chroma[0]), hi0 = vdupq_n_s16(b->max_chroma[0]);
	int16x8_t lo1 = vdupq_n_s16(b->min_chroma[1]), hi1 = vdupq_n_s16(b->max_chroma[1]);

	for (unsigned int x = 0; x < cw; x += 8) {
		uint16x8x2_t uv = vld2q_u16(in + 2 * x);
		int16x8_t u = vreinterpretq_s16_u16(vshrq_n_u16(uv.val[0], 6));
		int16x8_t v = vreinterpretq_s16_u16(vshrq_n_u16(uv.val[1], 6));
		int16x8_t nu = vqrdmulhq_s16(vld1q_s16(n0 + x), scale_factor(vld1_u8(s0 + x), k));
		int16x8_t nv = vqrdmulhq_s16(vld1q_s16(n1 + x), scale_factor(vld1_u8(s1 + x), k));
		uint16x8x2_t r;

		u = vminq_s16(vmaxq_s16(vaddq_s16(u, nu), lo0), hi0);
		v = vminq_s16(vmaxq_s16(vaddq_s16(v, nv), lo1), hi1);
		r.val[0] = vshlq_n_u16(vreinterpretq_u16_s16(u), 6);
		r.val[1] = vshlq_n_u16(vreinterpretq_u16_s16(v), 6);
		vst2q_u16(out + 2 * x, r);
	}
}
#endif /* GRAIN_NEON */

static void blend_luma(const struct av1_grain_state *state,
		       const struct blend *b, unsigned int w, bool p010)
{
#ifdef GRAIN_NEON
	if (p010)
		blend_luma_neon10(state, b, w);
	else
		blend_luma_neon8(state, b, w);
#else
	if (p010)
		blend_luma_c(state, b, w, true);
	else
		blend_luma_c(state, b, w, false);
#endif
}

static void blend_chroma(const struct av1_grain_state *state,
			 const struct blend *b, unsigned int w, bool p010)
{
#ifdef GRAIN_NEON
	if (p010)
		blend_chroma_neon10(state, b, w);
	else
		blend_chroma_neon8(state, b, w);
#else
	if (p010)
		blend_chroma_c(state, b, w, true);
	else
		blend_chroma_c(state, b, w, false);
#endif
}

/* A chroma row, the average luma of an odd width's last pair taken from
 * its one luma sample: that sample repeated past it while the row blends
 * (the luma row buffer's own sample there goes back after). */
static void blend_chroma_row(const struct av1_grain_state *state,
			     const struct blend *b, unsigned int w, bool p010)
{
	uint8_t *past = state->luma_in + ((size_t)w << p010);
	uint8_t kept[2];

	if (!(w & 1)) {
		blend_chroma(state, b, w, p010);
		return;
	}
	memcpy(kept, past, (size_t)1 << p010);
	memcpy(past, past - ((size_t)1 << p010), (size_t)1 << p010);
	blend_chroma(state, b, w, p010);
	memcpy(past, kept, (size_t)1 << p010);
}

/*
 * The add noise synthesis process, section 7.18.3.5, row by row, for 4:2:0
 * (av1_grain_supported()): each chroma row with the luma row it sits on,
 * whose source samples it reads. Every row is written whole: past the
 * picture, the source's bytes.
 */
static void add_noise(struct av1_grain_state *state,
		      const struct av1_grain_params *p,
		      const struct av1_grain_image *src,
		      const struct av1_grain_image *dst,
		      unsigned int w, unsigned int h, unsigned int blocks,
		      unsigned int planes, int grain_min, int grain_max)
{
	unsigned int chroma_w = (w + 1) >> 1;
	unsigned int stripes = ((h + 1) / 2 + 15) / 16;
	bool p010 = p->bit_depth > 8;
	size_t luma_bytes = (size_t)w << p010;
	size_t chroma_bytes = ((size_t)chroma_w * 2) << p010;
	size_t luma_pitch = dst->luma_pitch, chroma_pitch = dst->chroma_pitch;
	bool luma_noise = p->num_y_points > 0;
	bool do_cb = p->num_cb_points > 0 || p->chroma_scaling_from_luma;
	bool do_cr = p->num_cr_points > 0 || p->chroma_scaling_from_luma;
	bool chroma_noise = planes > 1 && (do_cb || do_cr);
	bool overlap = p->overlap_flag;
	int offset_scale = 1 << (p->bit_depth - 8);
	struct blend b = {
		.shift = p->grain_scaling_minus_8 + 8,
		.max_sample = (1 << p->bit_depth) - 1,
		.luma_mult = { p->cb_luma_mult - 128, p->cr_luma_mult - 128 },
		.mult = { p->cb_mult - 128, p->cr_mult - 128 },
		.offset = { (p->cb_offset - 256) * offset_scale,
			    (p->cr_offset - 256) * offset_scale },
		.from_luma = p->chroma_scaling_from_luma,
	};
	int max_chroma;

	if (p->clip_to_restricted_range) {
		b.min_luma = 16 << (p->bit_depth - 8);
		b.max_luma = 235 << (p->bit_depth - 8);
		max_chroma = p->mc_identity ? b.max_luma : 240 << (p->bit_depth - 8);
	} else {
		b.min_luma = 0;
		b.max_luma = b.max_sample;
		max_chroma = b.max_sample;
	}
	b.min_chroma[0] = do_cb ? b.min_luma : 0;
	b.max_chroma[0] = do_cb ? max_chroma : b.max_sample;
	b.min_chroma[1] = do_cr ? b.min_luma : 0;
	b.max_chroma[1] = do_cr ? max_chroma : b.max_sample;

	for (unsigned int s = 0; s < stripes; s++) {
		const uint8_t *cur = state->offsets[s & 1];
		const uint8_t *prev = state->offsets[(s & 1) ^ 1];
		bool top = overlap && s > 0;

		stripe_offsets(state, p, s, state->offsets[s & 1], blocks);

		for (unsigned int i = 0; i < 32 && 32 * s + i < h; i++) {
			unsigned int y = 32 * s + i;
			uint8_t *out = dst->luma + y * luma_pitch;

			memcpy(state->luma_in, src->luma + y * src->luma_pitch, luma_pitch);

			/* Chroma row y / 2 sits on luma row y. */
			if (!(i & 1)) {
				uint8_t *cout = dst->chroma + (y / 2) * chroma_pitch;
				unsigned int ci = i / 2;

				memcpy(state->chroma_in, src->chroma + (y / 2) * src->chroma_pitch,
				       chroma_pitch);
				if (!chroma_noise) {
					/* mono_chrome, or no noise for either: the row as decoded */
					memcpy(cout, state->chroma_in, chroma_pitch);
				} else {
					stripe_row(state->noise[1], state->cb_grain, cur, blocks, ci,
						   true, overlap, grain_min, grain_max);
					stripe_row(state->noise[2], state->cr_grain, cur, blocks, ci,
						   true, overlap, grain_min, grain_max);
					if (top && ci == 0) {
						stripe_row(state->above, state->cb_grain, prev, blocks, 16,
							   true, overlap, grain_min, grain_max);
						blend_rows(state->noise[1], state->above, chroma_w, 23, 22,
							   grain_min, grain_max);
						stripe_row(state->above, state->cr_grain, prev, blocks, 16,
							   true, overlap, grain_min, grain_max);
						blend_rows(state->noise[2], state->above, chroma_w, 23, 22,
							   grain_min, grain_max);
					}
					blend_chroma_row(state, &b, w, p010);
					memcpy(state->chroma_out + chroma_bytes,
					       state->chroma_in + chroma_bytes, chroma_pitch - chroma_bytes);
					memcpy(cout, state->chroma_out, chroma_pitch);
				}
			}

			if (!luma_noise) {
				memcpy(out, state->luma_in, luma_pitch);
				continue;
			}
			stripe_row(state->noise[0], state->luma_grain, cur, blocks, i, false,
				   overlap, grain_min, grain_max);
			if (top && i < 2) {
				stripe_row(state->above, state->luma_grain, prev, blocks, 32 + i,
					   false, overlap, grain_min, grain_max);
				blend_rows(state->noise[0], state->above, w, i ? 17 : 27,
					   i ? 27 : 17, grain_min, grain_max);
			}
			blend_luma(state, &b, w, p010);
			memcpy(state->luma_out + luma_bytes, state->luma_in + luma_bytes,
			       luma_pitch - luma_bytes);
			memcpy(out, state->luma_out, luma_pitch);
		}
	}
}

int av1_grain_apply(struct av1_grain_state *state,
		    const struct av1_grain_params *p,
		    const struct av1_grain_image *src,
		    const struct av1_grain_image *dst,
		    unsigned int w, unsigned int h)
{
	unsigned int planes = p->mono_chrome ? 1 : 3;
	unsigned int blocks = ((w + 1) / 2 + 15) / 16;
	size_t sample = p->bit_depth > 8 ? 2 : 1;
	int grain_center, grain_min, grain_max;

	if (!w || !h || !av1_grain_supported(p))
		return -1;
	if (src->luma_pitch != dst->luma_pitch ||
	    src->chroma_pitch != dst->chroma_pitch ||
	    w * sample > dst->luma_pitch ||
	    (size_t)(w + 1) / 2 * 2 * sample > dst->chroma_pitch)
		return -1;

	if (alloc_rows(state, blocks, dst->luma_pitch, dst->chroma_pitch) < 0)
		return -1;

	/* Section 7.18.3.1, steps 1 to 6. */
	grain_center = 128 << (p->bit_depth - 8);
	grain_min = -grain_center;
	grain_max = (256 << (p->bit_depth - 8)) - 1 - grain_center;

	generate_grain(state, p, grain_min, grain_max);
	init_scaling(state, p, planes);

	/* Step 7. */
	add_noise(state, p, src, dst, w, h, blocks, planes, grain_min,
		  grain_max);

	return 0;
}

/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * AV1 film grain synthesis (AV1 specification, section 7.18.3), on the CPU,
 * for decoders without it: Apple's AVD leaves the grain to software.
 */

#ifndef AV1_GRAIN_H
#define AV1_GRAIN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * The film grain parameters as the spec names them (film_grain_params(),
 * section 5.9.30, after load_grain_params), and the sequence values the
 * synthesis reads. The AR coefficients are the coded values minus 128.
 */
struct av1_grain_params {
	uint16_t grain_seed;
	uint8_t num_y_points;
	uint8_t point_y_value[14];
	uint8_t point_y_scaling[14];
	uint8_t num_cb_points;
	uint8_t point_cb_value[10];
	uint8_t point_cb_scaling[10];
	uint8_t num_cr_points;
	uint8_t point_cr_value[10];
	uint8_t point_cr_scaling[10];
	bool chroma_scaling_from_luma;
	uint8_t grain_scaling_minus_8;
	uint8_t ar_coeff_lag;
	int8_t ar_coeffs_y[24];
	int8_t ar_coeffs_cb[25];
	int8_t ar_coeffs_cr[25];
	uint8_t ar_coeff_shift_minus_6;
	uint8_t grain_scale_shift;
	uint8_t cb_mult;
	uint8_t cb_luma_mult;
	uint16_t cb_offset;
	uint8_t cr_mult;
	uint8_t cr_luma_mult;
	uint16_t cr_offset;
	bool overlap_flag;
	bool clip_to_restricted_range;

	unsigned int bit_depth;		/* 8 or 10 */
	unsigned int subsampling_x;
	unsigned int subsampling_y;
	bool mono_chrome;
	bool mc_identity;		/* matrix_coefficients == MC_IDENTITY */
};

/*
 * A semi-planar 4:2:0 picture: NV12 (8 bits), or P010 (10 bits in the high
 * bits of 16-bit samples). The chroma plane interleaves Cb and Cr.
 */
struct av1_grain_image {
	uint8_t *luma;
	uint8_t *chroma;
	size_t luma_pitch;
	size_t chroma_pitch;
};

struct av1_grain_state;

struct av1_grain_state *av1_grain_state_new(void);
void av1_grain_state_free(struct av1_grain_state *state);

/* Whether av1_grain_apply() can synthesise grain with these parameters:
 * 4:2:0 at 8 or 10 bits, the point lists within bounds and increasing. */
bool av1_grain_supported(const struct av1_grain_params *params);

/*
 * Write src with grain added into dst, a picture w samples wide and h
 * high (UpscaledWidth and FrameHeight), src and dst with the same pitches:
 * the h luma rows and (h + 1) / 2 chroma rows are written whole, the
 * samples the grain leaves as they are and the bytes past the picture (up
 * to the pitch) copied from src. Returns 0, or -1 when the parameters or
 * the layout are not supported or memory runs out (dst then unwritten).
 */
int av1_grain_apply(struct av1_grain_state *state,
		    const struct av1_grain_params *params,
		    const struct av1_grain_image *src,
		    const struct av1_grain_image *dst,
		    unsigned int w, unsigned int h);

#endif /* AV1_GRAIN_H */

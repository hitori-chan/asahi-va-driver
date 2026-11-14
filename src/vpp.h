/*
 * Copyright (C) 2026 Ondrej Jirman <megi@xff.cz>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef VPP_H
#define VPP_H

#include <linux/videodev2.h>
#include <stdbool.h>

#include <va/va.h>
#include <va/va_backend.h>
#include <va/va_backend_vpp.h>
#include <va/va_vpp.h>

struct v4l2r_context;
struct v4l2r_buffer;
struct v4l2r_surface;

struct vpp_ctx {
	int fd;				/* configured converter, -1 until first blit */
	struct v4l2_format output_format;	/* source */
	struct v4l2_format capture_format;	/* destination */

	/* Geometry the converter is currently configured for; a change in
	 * any of these forces a reconfiguration (V4L2 formats cannot change
	 * while buffers are allocated). */
	uint32_t src_pixelformat, dst_pixelformat;
	uint32_t src_width, src_height, src_pitch;
	uint32_t dst_width, dst_height, dst_pitch;

	/* Pipeline parameters of the picture being assembled. */
	struct v4l2r_surface *src_surface;
	VARectangle src_region;
	VARectangle dst_region;
	uint32_t rotation;		/* VA_ROTATION_* */
	uint32_t mirror;		/* VA_MIRROR_* mask */
	bool have_params;
};

VAStatus vpp_ctx_create(struct v4l2r_context *ctx);
void vpp_ctx_destroy(struct v4l2r_context *ctx);

VAStatus vpp_ctx_begin_picture(struct v4l2r_context *ctx,
			       struct v4l2r_surface *target);
VAStatus vpp_ctx_render_buffer(struct v4l2r_context *ctx,
			       struct v4l2r_buffer *buf);
VAStatus vpp_ctx_end_picture(struct v4l2r_context *ctx);

void vpp_fill_vtable(struct VADriverVTableVPP *vtable);

#endif /* VPP_H */

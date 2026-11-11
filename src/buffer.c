/*
 * VA buffer objects.
 *
 * Parameter buffers are plain heap allocations; bitstream data is copied
 * exactly once, straight into the mmap()ed V4L2 OUTPUT buffer, when the
 * codec backend processes it at RenderPicture time.
 *
 * Copyright (C) 2026 Ondrej Jirman <megi@xff.cz>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <stdlib.h>
#include <string.h>

#include <linux/dma-buf.h>

#include "v4l2_request.h"

VAStatus v4l2r_CreateBuffer(VADriverContextP va_ctx, VAContextID context_id,
			    VABufferType type, unsigned int size,
			    unsigned int num_elements, void *data,
			    VABufferID *buf_id)
{
	struct v4l2r_driver *drv = v4l2r_driver(va_ctx);
	struct v4l2r_buffer *buffer;
	VABufferID id;

	switch (type) {
	case VAPictureParameterBufferType:
	case VAIQMatrixBufferType:
	case VASliceParameterBufferType:
	case VASliceDataBufferType:
	case VAProbabilityBufferType:
	case VAImageBufferType:
	case VAProcPipelineParameterBufferType:
		break;
	default:
		return VA_STATUS_ERROR_UNSUPPORTED_BUFFERTYPE;
	}

	(void)context_id;

	pthread_mutex_lock(&drv->mutex);
	id = v4l2r_handles_alloc(&drv->buffers, sizeof(*buffer));
	buffer = V4L2R_BUFFER(drv, id);
	pthread_mutex_unlock(&drv->mutex);
	if (!buffer)
		return VA_STATUS_ERROR_ALLOCATION_FAILED;

	buffer->type = type;
	buffer->element_size = size;
	buffer->nb_elements = num_elements;

	buffer->data = malloc((size_t)size * num_elements);
	if (!buffer->data) {
		pthread_mutex_lock(&drv->mutex);
		v4l2r_handles_free(&drv->buffers, id);
		pthread_mutex_unlock(&drv->mutex);
		return VA_STATUS_ERROR_ALLOCATION_FAILED;
	}

	if (data)
		memcpy(buffer->data, data, (size_t)size * num_elements);

	*buf_id = id;
	return VA_STATUS_SUCCESS;
}

VAStatus v4l2r_BufferSetNumElements(VADriverContextP va_ctx, VABufferID buf_id,
				    unsigned int num_elements)
{
	struct v4l2r_driver *drv = v4l2r_driver(va_ctx);
	struct v4l2r_buffer *buffer;

	buffer = V4L2R_BUFFER_GET(drv, buf_id);
	if (!buffer)
		return VA_STATUS_ERROR_INVALID_BUFFER;

	if (buffer->derived)
		return VA_STATUS_ERROR_INVALID_BUFFER;

	if (num_elements != buffer->nb_elements) {
		void *data = realloc(buffer->data,
				     (size_t)buffer->element_size * num_elements);
		if (!data)
			return VA_STATUS_ERROR_ALLOCATION_FAILED;
		buffer->data = data;
		buffer->nb_elements = num_elements;
	}

	return VA_STATUS_SUCCESS;
}

/*
 * A derived image's data is the surface's mapped backing: the client reads or
 * writes it between the first map and the last unmap, which bracket the
 * CPU's access to a cached backing (v4l2r_surface_cpu_begin()).
 */
static VAStatus buffer_map(struct v4l2r_driver *drv, VABufferID buf_id,
			   void **pbuf, uint64_t access)
{
	struct v4l2r_surface *begin = NULL;
	struct v4l2r_buffer *buffer;

	pthread_mutex_lock(&drv->mutex);
	buffer = V4L2R_BUFFER(drv, buf_id);
	if (!buffer) {
		pthread_mutex_unlock(&drv->mutex);
		return VA_STATUS_ERROR_INVALID_BUFFER;
	}
	if (buffer->derived && buffer->surface) {
		/* A later map that also writes: its end cleans, the
		 * first map's begin invalidated already. */
		if (!buffer->map_count++)
			begin = buffer->surface;
		buffer->map_access |= access;
	}
	*pbuf = buffer->data;
	pthread_mutex_unlock(&drv->mutex);

	if (begin)
		v4l2r_surface_cpu_begin(drv, begin, access);

	return VA_STATUS_SUCCESS;
}

VAStatus v4l2r_MapBuffer(VADriverContextP va_ctx, VABufferID buf_id, void **pbuf)
{
	return buffer_map(v4l2r_driver(va_ctx), buf_id, pbuf, DMA_BUF_SYNC_RW);
}

#if VA_CHECK_VERSION(1, 21, 0)
VAStatus v4l2r_MapBuffer2(VADriverContextP va_ctx, VABufferID buf_id,
			  void **pbuf, uint32_t flags)
{
	uint64_t access = 0;

	if (flags & VA_MAPBUFFER_FLAG_READ)
		access |= DMA_BUF_SYNC_READ;
	if (flags & VA_MAPBUFFER_FLAG_WRITE)
		access |= DMA_BUF_SYNC_WRITE;

	/* VA_MAPBUFFER_FLAG_DEFAULT: both */
	return buffer_map(v4l2r_driver(va_ctx), buf_id, pbuf,
			  access ? access : DMA_BUF_SYNC_RW);
}
#endif

/* The last unmap of a derived image (or its destruction while mapped): the
 * surface to end the CPU's access to, and how; under the driver mutex. */
static struct v4l2r_surface *buffer_unmap_locked(struct v4l2r_buffer *buffer,
						 bool all, uint64_t *access)
{
	if (!buffer->derived || !buffer->map_count)
		return NULL;

	buffer->map_count = all ? 0 : buffer->map_count - 1;
	if (buffer->map_count)
		return NULL;

	*access = buffer->map_access;
	buffer->map_access = 0;
	return buffer->surface;
}

VAStatus v4l2r_UnmapBuffer(VADriverContextP va_ctx, VABufferID buf_id)
{
	struct v4l2r_driver *drv = v4l2r_driver(va_ctx);
	struct v4l2r_surface *end;
	struct v4l2r_buffer *buffer;
	uint64_t access = 0;

	pthread_mutex_lock(&drv->mutex);
	buffer = V4L2R_BUFFER(drv, buf_id);
	if (!buffer) {
		pthread_mutex_unlock(&drv->mutex);
		return VA_STATUS_ERROR_INVALID_BUFFER;
	}
	end = buffer_unmap_locked(buffer, false, &access);
	pthread_mutex_unlock(&drv->mutex);

	if (end)
		v4l2r_surface_cpu_end(drv, end, access);

	return VA_STATUS_SUCCESS;
}

VAStatus v4l2r_DestroyBuffer(VADriverContextP va_ctx, VABufferID buf_id)
{
	struct v4l2r_driver *drv = v4l2r_driver(va_ctx);
	struct v4l2r_surface *end;
	struct v4l2r_buffer *buffer;
	uint64_t access = 0;

	pthread_mutex_lock(&drv->mutex);
	buffer = V4L2R_BUFFER(drv, buf_id);
	if (!buffer) {
		pthread_mutex_unlock(&drv->mutex);
		return VA_STATUS_ERROR_INVALID_BUFFER;
	}

	end = buffer_unmap_locked(buffer, true, &access);

	/* Derived image buffers point into the CAPTURE buffer mapping. */
	if (!buffer->derived)
		free(buffer->data);
	else if (buffer->surface)
		buffer->surface->derived--;

	v4l2r_handles_free(&drv->buffers, buf_id);
	pthread_mutex_unlock(&drv->mutex);

	/* Destroyed while mapped: the access ends here. */
	if (end)
		v4l2r_surface_cpu_end(drv, end, access);

	return VA_STATUS_SUCCESS;
}

VAStatus v4l2r_BufferInfo(VADriverContextP va_ctx, VABufferID buf_id,
			  VABufferType *type, unsigned int *size,
			  unsigned int *num_elements)
{
	struct v4l2r_driver *drv = v4l2r_driver(va_ctx);
	struct v4l2r_buffer *buffer;

	buffer = V4L2R_BUFFER_GET(drv, buf_id);
	if (!buffer)
		return VA_STATUS_ERROR_INVALID_BUFFER;

	if (type)
		*type = buffer->type;
	if (size)
		*size = buffer->element_size;
	if (num_elements)
		*num_elements = buffer->nb_elements;

	return VA_STATUS_SUCCESS;
}

VAStatus v4l2r_AcquireBufferHandle(VADriverContextP va_ctx, VABufferID buf_id,
				   VABufferInfo *buf_info)
{
	(void)va_ctx; (void)buf_id; (void)buf_info;
	return VA_STATUS_ERROR_UNIMPLEMENTED;
}

VAStatus v4l2r_ReleaseBufferHandle(VADriverContextP va_ctx, VABufferID buf_id)
{
	(void)va_ctx; (void)buf_id;
	return VA_STATUS_ERROR_UNIMPLEMENTED;
}

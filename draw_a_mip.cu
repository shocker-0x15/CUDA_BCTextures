#include <cuda.h>

extern "C" __global__ void drawMip(
	CUtexObject texture, int32_t mipLevel,
	uint32_t* dstImg, int2 dstImgSize) {
	const int2 pixIdx(
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y);
	float4 texValue = tex2DLod<float4>(
		texture, (pixIdx.x + 0.5f) / dstImgSize.x, (pixIdx.y + 0.5f) / dstImgSize.y, mipLevel);
	//printf("%4u, %4u: (%g, %g, %g, %g)\n", pixIdx.x, pixIdx.y, texValue.x, texValue.y, texValue.z, texValue.w);
	dstImg[pixIdx.y * dstImgSize.x + pixIdx.x] =
		(min(static_cast<uint32_t>(255 * texValue.x), 255u) << 0)
		| (min(static_cast<uint32_t>(255 * texValue.y), 255u) << 8)
		| (min(static_cast<uint32_t>(255 * texValue.z), 255u) << 16)
		| (min(static_cast<uint32_t>(255 * texValue.w), 255u) << 24);
}

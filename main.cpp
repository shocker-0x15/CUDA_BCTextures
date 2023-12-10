#define NOMINMAX
#define _USE_MATH_DEFINES
#include <Windows.h>
#undef near
#undef far
#undef RGB

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>

#include <cuda.h>
#include <vector_types.h>
#include "dds_loader.h"
#include "stb_image_write.h"

#define CUDADRV_CHECK(call) \
    do { \
        CUresult error = call; \
        if (error != CUDA_SUCCESS) { \
            std::stringstream ss; \
            const char* errMsg = "failed to get an error message."; \
            cuGetErrorString(error, &errMsg); \
            ss << "CUDA call (" << #call << " ) failed with error: '" \
               << errMsg \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)

void devPrintf(const char* fmt, ...);
std::filesystem::path getExecutableDirectory();



int32_t mainFunc(int32_t argc, const char* argv[]) {
    CUcontext cuContext;
    CUdevice cuDevice;
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuDeviceGet(&cuDevice, 0));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));

    // Initialize a kernel.
    CUmodule cuModule;
    CUDADRV_CHECK(cuModuleLoad(
        &cuModule,
        (getExecutableDirectory() / "CUDA_BCTextures/ptxes/draw_a_mip.ptx").string().c_str()));
    CUfunction kernel;
    CUDADRV_CHECK(cuModuleGetFunction(&kernel, cuModule, "drawMip"));

    // Read DDS texture data.
    int32_t width, height, numMipLevels;
    size_t* mipSizes;
    dds::Format format;
    // 1024x1024, 512x512, 256x256, 128x128, 64x64, 32x32, 16x16, 8x8, 4x4, 2x2, 1x1
    uint8_t** const data = dds::load("data/example.dds", &width, &height, &numMipLevels, &mipSizes, &format);
    // 37x27, 18x13, 9x6, 4x3, 2x1, 1x1
    //uint8_t** const data = dds::load("data/mip_test_37x27.dds", &width, &height, &numMipLevels, &mipSizes, &format);
    //uint8_t** const data = dds::load("data/mip_test_37x27_purple.dds", &width, &height, &numMipLevels, &mipSizes, &format);

    const bool use_sRGB_degamma =
        format == dds::Format::BC1_UNorm_sRGB ||
        format == dds::Format::BC2_UNorm_sRGB ||
        format == dds::Format::BC3_UNorm_sRGB ||
        format == dds::Format::BC7_UNorm_sRGB;

    /*
    JP: このプログラムはMIPレベルごとに画像を生成する。
        このフラグを0にセットすると、プログラムはCUDA ArrayをUint要素と1/4サイズで作成する。
        これはArrayが最後の2つのミップレベルを失うことを意味する。
        フラグを1にセットすると、プログラムはブロック圧縮テクスチャー専用のenumと本来のサイズでArrayを作成する。
        このenumはCUDA 11.5から導入されている。
    EN: This program produces images, each for each mip level.
        With this flag == 0, it creates a CUDA array with UInt element and 1/4 dimensions.
        This means that the array loses the last two mip levels.
        With this flag == 1, it creates the array with a proper dedicated enum for block compressed texture and
        the true dimensions. That enums was introduced at CUDA 11.5.
    */
#define USE_DEDICATED_ENUM 1

    // CUDA until 11.5 requires to specify the image size in the number of blocks (4x4 pixels) instead of pixels.
    // This requires us to limit the maximum mip level by 2, otherwise get a runtime error.
    const int32_t _numMipLevels = USE_DEDICATED_ENUM ? numMipLevels : std::max(numMipLevels - 2, 1);

    // Create a mipmapped CUDA array.
    CUmipmappedArray mipmappedArray;
    std::vector<CUarray> mipArrays(numMipLevels);
    size_t stride;
    {
        if (format == dds::Format::BC1_UNorm ||
            format == dds::Format::BC1_UNorm_sRGB ||
            format == dds::Format::BC4_UNorm ||
            format == dds::Format::BC4_SNorm) {
            stride = 8;
        }
        else {
            stride = 16;
        }

        CUDA_ARRAY3D_DESCRIPTOR arrayDesc = {};
        if constexpr (USE_DEDICATED_ENUM) {
            // JP: CUDA 11.5以降、BCテクスチャーフォーマット専用のenumが用意されている。
            // EN: Since CUDA 11.5, we have dedicated enum values for BC texture formats.
            switch (format) {
            case dds::Format::BC1_UNorm:
                arrayDesc.Format = CU_AD_FORMAT_BC1_UNORM;
                arrayDesc.NumChannels = 4;
                break;
            case dds::Format::BC1_UNorm_sRGB:
                arrayDesc.Format = CU_AD_FORMAT_BC1_UNORM_SRGB;
                arrayDesc.NumChannels = 4;
                break;
            case dds::Format::BC2_UNorm:
                arrayDesc.Format = CU_AD_FORMAT_BC2_UNORM;
                arrayDesc.NumChannels = 4;
                break;
            case dds::Format::BC2_UNorm_sRGB:
                arrayDesc.Format = CU_AD_FORMAT_BC2_UNORM_SRGB;
                arrayDesc.NumChannels = 4;
                break;
            case dds::Format::BC3_UNorm:
                arrayDesc.Format = CU_AD_FORMAT_BC3_UNORM;
                arrayDesc.NumChannels = 4;
                break;
            case dds::Format::BC3_UNorm_sRGB:
                arrayDesc.Format = CU_AD_FORMAT_BC3_UNORM_SRGB;
                arrayDesc.NumChannels = 4;
                break;
            case dds::Format::BC4_UNorm:
                arrayDesc.Format = CU_AD_FORMAT_BC4_UNORM;
                arrayDesc.NumChannels = 1;
                break;
            case dds::Format::BC4_SNorm:
                arrayDesc.Format = CU_AD_FORMAT_BC4_SNORM;
                arrayDesc.NumChannels = 1;
                break;
            case dds::Format::BC5_UNorm:
                arrayDesc.Format = CU_AD_FORMAT_BC5_UNORM;
                arrayDesc.NumChannels = 2;
                break;
            case dds::Format::BC5_SNorm:
                arrayDesc.Format = CU_AD_FORMAT_BC5_SNORM;
                arrayDesc.NumChannels = 2;
                break;
            case dds::Format::BC6H_UF16:
                arrayDesc.Format = CU_AD_FORMAT_BC6H_UF16;
                arrayDesc.NumChannels = 3;
                break;
            case dds::Format::BC6H_SF16:
                arrayDesc.Format = CU_AD_FORMAT_BC6H_SF16;
                arrayDesc.NumChannels = 3;
                break;
            case dds::Format::BC7_UNorm:
                arrayDesc.Format = CU_AD_FORMAT_BC7_UNORM;
                arrayDesc.NumChannels = 4;
                break;
            case dds::Format::BC7_UNorm_sRGB:
                arrayDesc.Format = CU_AD_FORMAT_BC7_UNORM_SRGB;
                arrayDesc.NumChannels = 4;
                break;
            default:
                break;
            }
            arrayDesc.Width = width;
            arrayDesc.Height = height;
        }
        else {
            // JP: CUDA 11.5より前は1/4サイズの非BCテクスチャーのふりをする必要があった。
            // EN: Until CUDA 11.5, we needed to pretend non-BC texture of 1/4 dimensions.
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
            if (format == dds::Format::BC1_UNorm ||
                format == dds::Format::BC1_UNorm_sRGB ||
                format == dds::Format::BC4_UNorm ||
                format == dds::Format::BC4_SNorm) {
                arrayDesc.NumChannels = 2;
            }
            else {
                arrayDesc.NumChannels = 4;
            }
            arrayDesc.Width = (width + 3) / 4;
            arrayDesc.Height = (height + 3) / 4;
        }
        arrayDesc.Depth = 0;

        CUDADRV_CHECK(cuMipmappedArrayCreate(&mipmappedArray, &arrayDesc, _numMipLevels));

        for (int32_t mipLevel = 0; mipLevel < _numMipLevels; ++mipLevel)
            CUDADRV_CHECK(cuMipmappedArrayGetLevel(&mipArrays[mipLevel], mipmappedArray, mipLevel));
    }

    // Transfer data for each mip level.
    for (int32_t mipLevel = 0; mipLevel < _numMipLevels; ++mipLevel) {
        CUDA_ARRAY_DESCRIPTOR arrayDesc;
        CUDADRV_CHECK(cuArrayGetDescriptor(&arrayDesc, mipArrays[mipLevel]));

#if USE_DEDICATED_ENUM
        const size_t deviceBw = (arrayDesc.Width + 3) / 4;
        const size_t deviceBh = (arrayDesc.Height + 3) / 4;
        const size_t deviceSizePerRow = deviceBw * stride;
        const size_t hostBw = deviceBw;
        const size_t hostBh = deviceBh;
        const size_t hostSizePerRow = deviceSizePerRow;
#else
        const size_t deviceBw = arrayDesc.Width;
        const size_t deviceBh = arrayDesc.Height;
        const size_t deviceSizePerRow = deviceBw * stride;
        const size_t hostBw = ((std::max(width >> mipLevel, 1)) + 3) / 4;
        const size_t hostBh = ((std::max(height >> mipLevel, 1)) + 3) / 4;
        const size_t hostSizePerRow = hostBw * stride;
#endif

        CUDA_MEMCPY3D params = {};
        params.WidthInBytes = deviceSizePerRow;
        params.Height = deviceBh;
        params.Depth = 1;

        params.srcMemoryType = CU_MEMORYTYPE_HOST;
        params.srcHost = data[mipLevel];
        params.srcPitch = hostSizePerRow;
        params.srcHeight = hostBh;
        params.srcXInBytes = 0;
        params.srcY = 0;
        params.srcZ = 0;
        // srcArray, srcDevice, srcLOD are not used in this case.

        params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        params.dstArray = mipArrays[mipLevel];
        params.dstXInBytes = 0;
        params.dstY = 0;
        params.dstZ = 0;
        // dstDevice, dstHeight, dstHost, dstLOD, dstPitch are not used in this case.

        CUDADRV_CHECK(cuMemcpy3D(&params));
    }
    dds::free(data, mipSizes);

    // Create a texture object.
    CUtexObject texture;
    {
        CUDA_TEXTURE_DESC texDesc = {};
        texDesc.flags |= CU_TRSF_NORMALIZED_COORDINATES;
        if (use_sRGB_degamma)
            texDesc.flags |= CU_TRSF_SRGB;
        else
            texDesc.flags &= ~CU_TRSF_SRGB;
        texDesc.filterMode = CU_TR_FILTER_MODE_POINT;
        texDesc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
        texDesc.maxMipmapLevelClamp = _numMipLevels - 1;

        CUDA_RESOURCE_DESC resDesc = {};
        resDesc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        resDesc.res.mipmap.hMipmappedArray = mipmappedArray;

        CUDA_RESOURCE_VIEW_DESC resViewDesc = {};
        switch (format) {
        case dds::Format::BC1_UNorm:
        case dds::Format::BC1_UNorm_sRGB:
            resViewDesc.format = CU_RES_VIEW_FORMAT_UNSIGNED_BC1;
            break;
        case dds::Format::BC2_UNorm:
        case dds::Format::BC2_UNorm_sRGB:
            resViewDesc.format = CU_RES_VIEW_FORMAT_UNSIGNED_BC2;
            break;
        case dds::Format::BC3_UNorm:
        case dds::Format::BC3_UNorm_sRGB:
            resViewDesc.format = CU_RES_VIEW_FORMAT_UNSIGNED_BC3;
            break;
        case dds::Format::BC4_UNorm:
            resViewDesc.format = CU_RES_VIEW_FORMAT_UNSIGNED_BC4;
            break;
        case dds::Format::BC4_SNorm:
            resViewDesc.format = CU_RES_VIEW_FORMAT_SIGNED_BC4;
            break;
        case dds::Format::BC5_UNorm:
            resViewDesc.format = CU_RES_VIEW_FORMAT_UNSIGNED_BC5;
            break;
        case dds::Format::BC5_SNorm:
            resViewDesc.format = CU_RES_VIEW_FORMAT_SIGNED_BC5;
            break;
        case dds::Format::BC6H_UF16:
            resViewDesc.format = CU_RES_VIEW_FORMAT_UNSIGNED_BC6H;
            break;
        case dds::Format::BC6H_SF16:
            resViewDesc.format = CU_RES_VIEW_FORMAT_SIGNED_BC6H;
            break;
        case dds::Format::BC7_UNorm:
        case dds::Format::BC7_UNorm_sRGB:
            resViewDesc.format = CU_RES_VIEW_FORMAT_UNSIGNED_BC7;
            break;
        default:
            break;
        }
#if USE_DEDICATED_ENUM
        resViewDesc.width = width;
        resViewDesc.height = height;
#else
        resViewDesc.width = (width + 3) / 4 * 4;
        resViewDesc.height = (height + 3) / 4 * 4;
#endif
        resViewDesc.depth = 0;
        resViewDesc.firstMipmapLevel = 0;
        resViewDesc.lastMipmapLevel = _numMipLevels - 1;

        CUDADRV_CHECK(cuTexObjectCreate(
            &texture, &resDesc, &texDesc,
            USE_DEDICATED_ENUM ?
                nullptr : // new style
                &resViewDesc // old style
        ));
    }

    // Allocate an image buffer.
    const int2 imgSize = { 1024, 1024 };
    CUdeviceptr imgDataOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&imgDataOnDevice, sizeof(uint32_t) * imgSize.x * imgSize.y));

    // Draw mip levels.
    std::vector<uint32_t> img(imgSize.x * imgSize.y);
    for (int32_t mipLevel = 0; mipLevel < _numMipLevels; ++mipLevel) {
        const void* args[] = { &texture, &mipLevel, &imgDataOnDevice, &imgSize };
        CUDADRV_CHECK(cuLaunchKernel(
            kernel, imgSize.x / 32, imgSize.y / 32, 1, 32, 32, 1,
            0, 0,
            const_cast<void**>(args), nullptr));
        CUDADRV_CHECK(cuStreamSynchronize(0));

        CUDADRV_CHECK(cuMemcpyDtoH(img.data(), imgDataOnDevice, sizeof(uint32_t) * imgSize.x * imgSize.y));

        char filename[256];
        sprintf_s(filename, "%smip_%02d.png", USE_DEDICATED_ENUM ? "new_" : "old_", mipLevel);
        stbi_write_png(filename, imgSize.x, imgSize.y, 4, img.data(), imgSize.x * sizeof(uint32_t));
    }

    CUDADRV_CHECK(cuMemFree(imgDataOnDevice));

    CUDADRV_CHECK(cuTexObjectDestroy(texture));

    CUDADRV_CHECK(cuMipmappedArrayDestroy(mipmappedArray));

    CUDADRV_CHECK(cuModuleUnload(cuModule));

    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}

int32_t main(int32_t argc, const char* argv[]) {
    int32_t ret;
    try {
        ret = mainFunc(argc, argv);
    }
    catch (const std::exception &e) {
        devPrintf("[ERROR] %s\n", e.what());
        ret = -1;
    }
    return ret;
}



void devPrintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char str[4096];
    vsnprintf_s(str, sizeof(str), _TRUNCATE, fmt, args);
    va_end(args);
    OutputDebugString(str);
}

std::filesystem::path getExecutableDirectory() {
    static std::filesystem::path ret;

    static bool done = false;
    if (!done) {
        TCHAR filepath[1024];
        GetModuleFileName(NULL, filepath, 1024);
        ret = filepath;
        ret = ret.remove_filename();

        done = true;
    }

    return ret;
}

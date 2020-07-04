/*
**   VapourSynth port by HolyWu
**
**                    dfttest v1.8 for Avisynth 2.5.x
**
**   2D/3D frequency domain denoiser.
**
**   Copyright (C) 2007-2010 Kevin Stone
**
**   This program is free software: you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation, either version 3 of the License, or
**   (at your option) any later version.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>

#include <algorithm>
#include <string>

#include "DFTTest.h"

using namespace std::literals;

#ifdef DFTTEST_X86
#include "VCL2/vectorclass.h"

template<int type> extern void filter_sse2(float * dftc, const float * sigmas, const int ccnt, const float * pmin, const float * pmax, const float * sigmas2) noexcept;
template<int type> extern void filter_avx2(float * dftc, const float * sigmas, const int ccnt, const float * pmin, const float * pmax, const float * sigmas2) noexcept;
template<int type> extern void filter_avx512(float * dftc, const float * sigmas, const int ccnt, const float * pmin, const float * pmax, const float * sigmas2) noexcept;

template<typename pixel_t> extern void func_0_sse2(VSFrameRef * src[3], VSFrameRef * dst, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept;
template<typename pixel_t> extern void func_0_avx2(VSFrameRef * src[3], VSFrameRef * dst, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept;
template<typename pixel_t> extern void func_0_avx512(VSFrameRef * src[3], VSFrameRef * dst, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept;

template<typename pixel_t> extern void func_1_sse2(VSFrameRef * src[15][3], VSFrameRef * dst, const int pos, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept;
template<typename pixel_t> extern void func_1_avx2(VSFrameRef * src[15][3], VSFrameRef * dst, const int pos, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept;
template<typename pixel_t> extern void func_1_avx512(VSFrameRef * src[15][3], VSFrameRef * dst, const int pos, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept;
#endif

#define EXTRA(a,b) (((a) % (b)) ? ((b) - ((a) % (b))) : 0)

template<typename arg_t>
static auto getArg(const VSAPI * vsapi, const VSMap * map, const char * key, const arg_t defaultValue) noexcept {
    arg_t arg{};
    int err{};

    if constexpr (std::is_same_v<arg_t, bool>)
        arg = !!vsapi->propGetInt(map, key, 0, &err);
    else if constexpr (std::is_same_v<arg_t, int>)
        arg = int64ToIntS(vsapi->propGetInt(map, key, 0, &err));
    else if constexpr (std::is_same_v<arg_t, int64_t>)
        arg = vsapi->propGetInt(map, key, 0, &err);
    else if constexpr (std::is_same_v<arg_t, float>)
        arg = static_cast<float>(vsapi->propGetFloat(map, key, 0, &err));
    else if constexpr (std::is_same_v<arg_t, double>)
        arg = vsapi->propGetFloat(map, key, 0, &err);

    if (err)
        arg = defaultValue;

    return arg;
}

template<typename pixel_t>
static auto copyPad(const VSFrameRef * src, VSFrameRef * dst[3], const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept {
    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (d->process[plane]) {
            const int srcWidth = vsapi->getFrameWidth(src, plane);
            const int dstWidth = vsapi->getFrameWidth(dst[plane], 0);
            const int srcHeight = vsapi->getFrameHeight(src, plane);
            const int dstHeight = vsapi->getFrameHeight(dst[plane], 0);
            const int dstStride = vsapi->getStride(dst[plane], 0) / sizeof(pixel_t);

            const int offy = (dstHeight - srcHeight) / 2;
            const int offx = (dstWidth - srcWidth) / 2;

            vs_bitblt(vsapi->getWritePtr(dst[plane], 0) + vsapi->getStride(dst[plane], 0) * offy + offx * sizeof(pixel_t),
                      vsapi->getStride(dst[plane], 0),
                      vsapi->getReadPtr(src, plane),
                      vsapi->getStride(src, plane),
                      srcWidth * sizeof(pixel_t),
                      srcHeight);

            pixel_t * VS_RESTRICT dstp = reinterpret_cast<pixel_t *>(vsapi->getWritePtr(dst[plane], 0)) + dstStride * offy;

            for (int y = offy; y < srcHeight + offy; y++) {
                int w = offx * 2;
                for (int x = 0; x < offx; x++, w--)
                    dstp[x] = dstp[w];

                w = offx + srcWidth - 2;
                for (int x = offx + srcWidth; x < dstWidth; x++, w--)
                    dstp[x] = dstp[w];

                dstp += dstStride;
            }

            int w = offy * 2;
            for (int y = 0; y < offy; y++, w--)
                memcpy(vsapi->getWritePtr(dst[plane], 0) + vsapi->getStride(dst[plane], 0) * y,
                       vsapi->getReadPtr(dst[plane], 0) + vsapi->getStride(dst[plane], 0) * w,
                       dstWidth * sizeof(pixel_t));

            w = offy + srcHeight - 2;
            for (int y = offy + srcHeight; y < dstHeight; y++, w--)
                memcpy(vsapi->getWritePtr(dst[plane], 0) + vsapi->getStride(dst[plane], 0) * y,
                       vsapi->getReadPtr(dst[plane], 0) + vsapi->getStride(dst[plane], 0) * w,
                       dstWidth * sizeof(pixel_t));
        }
    }
}

template<typename pixel_t>
static inline auto proc0(const pixel_t * s0, const float * s1, float * VS_RESTRICT d, const int p0, const int p1, const float srcScale) noexcept {
    for (int u = 0; u < p1; u++) {
        for (int v = 0; v < p1; v++)
            d[v] = s0[v] * srcScale * s1[v];

        s0 += p0;
        s1 += p1;
        d += p1;
    }
}

static inline auto proc1(const float * s0, const float * s1, float * VS_RESTRICT d, const int p0, const int p1) noexcept {
    for (int u = 0; u < p0; u++) {
        for (int v = 0; v < p0; v++)
            d[v] += s0[v] * s1[v];

        s0 += p0;
        s1 += p0;
        d += p1;
    }
}

static inline auto removeMean(float * VS_RESTRICT dftc, const float * dftgc, const int ccnt, float * VS_RESTRICT dftc2) noexcept {
    const float gf = dftc[0] / dftgc[0];

    for (int h = 0; h < ccnt; h += 2) {
        dftc2[h + 0] = gf * dftgc[h + 0];
        dftc2[h + 1] = gf * dftgc[h + 1];
        dftc[h + 0] -= dftc2[h + 0];
        dftc[h + 1] -= dftc2[h + 1];
    }
}

static inline auto addMean(float * VS_RESTRICT dftc, const int ccnt, const float * dftc2) noexcept {
    for (int h = 0; h < ccnt; h += 2) {
        dftc[h + 0] += dftc2[h + 0];
        dftc[h + 1] += dftc2[h + 1];
    }
}

template<int type>
static inline void filter_c(float * VS_RESTRICT dftc, const float * sigmas, const int ccnt, const float * pmin, const float * pmax, const float * sigmas2) noexcept {
    const float beta = pmin[0];

    for (int h = 0; h < ccnt; h += 2) {
        float psd, mult;

        if constexpr (type != 2)
            psd = dftc[h + 0] * dftc[h + 0] + dftc[h + 1] * dftc[h + 1];

        if constexpr (type == 0) {
            mult = std::max((psd - sigmas[h]) / (psd + 1e-15f), 0.0f);
        } else if constexpr (type == 1) {
            if (psd < sigmas[h])
                dftc[h + 0] = dftc[h + 1] = 0.0f;
        } else if constexpr (type == 2) {
            dftc[h + 0] *= sigmas[h];
            dftc[h + 1] *= sigmas[h];
        } else if constexpr (type == 3) {
            if (psd >= pmin[h] && psd <= pmax[h]) {
                dftc[h + 0] *= sigmas[h];
                dftc[h + 1] *= sigmas[h];
            } else {
                dftc[h + 0] *= sigmas2[h];
                dftc[h + 1] *= sigmas2[h];
            }
        } else if constexpr (type == 4) {
            mult = sigmas[h] * std::sqrt(psd * pmax[h] / ((psd + pmin[h]) * (psd + pmax[h]) + 1e-15f));
        } else if constexpr (type == 5) {
            mult = std::pow(std::max((psd - sigmas[h]) / (psd + 1e-15f), 0.0f), beta);
        } else {
            mult = std::sqrt(std::max((psd - sigmas[h]) / (psd + 1e-15f), 0.0f));
        }

        if constexpr (type == 0 || type > 3) {
            dftc[h + 0] *= mult;
            dftc[h + 1] *= mult;
        }
    }
}

template<typename pixel_t>
static auto cast(const float * ebp, pixel_t * VS_RESTRICT dstp, const int dstWidth, const int dstHeight, const int dstStride, const int ebpStride,
                 const float dstScale, const int peak) noexcept {
    for (int y = 0; y < dstHeight; y++) {
        for (int x = 0; x < dstWidth; x++) {
            if constexpr (std::is_integral_v<pixel_t>)
                dstp[x] = std::clamp(static_cast<int>(ebp[x] * dstScale + 0.5f), 0, peak);
            else
                dstp[x] = ebp[x] * dstScale;
        }

        ebp += ebpStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void func_0_c(VSFrameRef * src[3], VSFrameRef * dst, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept {
    const float * hw = d->hw.get();
    const float * sigmas = d->sigmas.get();
    const float * sigmas2 = d->sigmas2.get();
    const float * pmins = d->pmins.get();
    const float * pmaxs = d->pmaxs.get();
    const fftwf_complex * dftgc = d->dftgc.get();
    fftwf_plan ft = d->ft.get();
    fftwf_plan fti = d->fti.get();

    const auto threadId = std::this_thread::get_id();
    float * ebuff = reinterpret_cast<float *>(vsapi->getWritePtr(d->ebuff.at(threadId).get(), 0));
    float * dftr = d->dftr.at(threadId).get();
    fftwf_complex * dftc = d->dftc.at(threadId).get();
    fftwf_complex * dftc2 = d->dftc2.at(threadId).get();

    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (d->process[plane]) {
            const int width = d->padWidth[plane];
            const int height = d->padHeight[plane];
            const int eheight = d->eheight[plane];
            const int srcStride = vsapi->getStride(src[plane], 0) / sizeof(pixel_t);
            const int ebpStride = vsapi->getStride(d->ebuff.at(threadId).get(), 0) / sizeof(float);
            const pixel_t * srcp = reinterpret_cast<const pixel_t *>(vsapi->getReadPtr(src[plane], 0));
            float * ebpSaved = ebuff;

            memset(ebuff, 0, ebpStride * height * sizeof(float));

            for (int y = 0; y < eheight; y += d->inc) {
                for (int x = 0; x <= width - d->sbsize; x += d->inc) {
                    proc0(srcp + x, hw, dftr, srcStride, d->sbsize, d->srcScale);

                    fftwf_execute_dft_r2c(ft, dftr, dftc);
                    if (d->zmean)
                        removeMean(reinterpret_cast<float *>(dftc), reinterpret_cast<const float *>(dftgc), d->ccnt2, reinterpret_cast<float *>(dftc2));

                    d->filterCoeffs(reinterpret_cast<float *>(dftc), sigmas, d->ccnt2, d->uf0b ? &d->f0beta : pmins, pmaxs, sigmas2);

                    if (d->zmean)
                        addMean(reinterpret_cast<float *>(dftc), d->ccnt2, reinterpret_cast<const float *>(dftc2));
                    fftwf_execute_dft_c2r(fti, dftc, dftr);

                    if (d->type & 1) // spatial overlapping
                        proc1(dftr, hw, ebpSaved + x, d->sbsize, ebpStride);
                    else
                        ebpSaved[x + d->sbd1 * ebpStride + d->sbd1] = dftr[d->sbd1 * d->sbsize + d->sbd1] * hw[d->sbd1 * d->sbsize + d->sbd1];
                }

                srcp += srcStride * d->inc;
                ebpSaved += ebpStride * d->inc;
            }

            const int dstWidth = vsapi->getFrameWidth(dst, plane);
            const int dstHeight = vsapi->getFrameHeight(dst, plane);
            const int dstStride = vsapi->getStride(dst, plane) / sizeof(pixel_t);
            pixel_t * dstp = reinterpret_cast<pixel_t *>(vsapi->getWritePtr(dst, plane));
            const float * ebp = ebuff + ebpStride * ((height - dstHeight) / 2) + (width - dstWidth) / 2;
            cast(ebp, dstp, dstWidth, dstHeight, dstStride, ebpStride, d->dstScale, d->peak);
        }
    }
}

template<typename pixel_t>
static void func_1_c(VSFrameRef * src[15][3], VSFrameRef * dst, const int pos, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept {
    const float * hw = d->hw.get();
    const float * sigmas = d->sigmas.get();
    const float * sigmas2 = d->sigmas2.get();
    const float * pmins = d->pmins.get();
    const float * pmaxs = d->pmaxs.get();
    const fftwf_complex * dftgc = d->dftgc.get();
    fftwf_plan ft = d->ft.get();
    fftwf_plan fti = d->fti.get();

    const auto threadId = std::this_thread::get_id();
    float * ebuff = reinterpret_cast<float *>(vsapi->getWritePtr(d->ebuff.at(threadId).get(), 0));
    float * dftr = d->dftr.at(threadId).get();
    fftwf_complex * dftc = d->dftc.at(threadId).get();
    fftwf_complex * dftc2 = d->dftc2.at(threadId).get();

    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (d->process[plane]) {
            const int width = d->padWidth[plane];
            const int height = d->padHeight[plane];
            const int eheight = d->eheight[plane];
            const int srcStride = vsapi->getStride(src[0][plane], 0) / sizeof(pixel_t);
            const int ebpStride = vsapi->getStride(d->ebuff.at(threadId).get(), 0) / sizeof(float);

            const pixel_t * srcp[15] = {};
            for (int i = 0; i < d->tbsize; i++)
                srcp[i] = reinterpret_cast<const pixel_t *>(vsapi->getReadPtr(src[i][plane], 0));

            memset(ebuff, 0, ebpStride * height * sizeof(float));

            for (int y = 0; y < eheight; y += d->inc) {
                for (int x = 0; x <= width - d->sbsize; x += d->inc) {
                    for (int z = 0; z < d->tbsize; z++)
                        proc0(srcp[z] + x, hw + d->barea * z, dftr + d->barea * z, srcStride, d->sbsize, d->srcScale);

                    fftwf_execute_dft_r2c(ft, dftr, dftc);
                    if (d->zmean)
                        removeMean(reinterpret_cast<float *>(dftc), reinterpret_cast<const float *>(dftgc), d->ccnt2, reinterpret_cast<float *>(dftc2));

                    d->filterCoeffs(reinterpret_cast<float *>(dftc), sigmas, d->ccnt2, d->uf0b ? &d->f0beta : pmins, pmaxs, sigmas2);

                    if (d->zmean)
                        addMean(reinterpret_cast<float *>(dftc), d->ccnt2, reinterpret_cast<const float *>(dftc2));
                    fftwf_execute_dft_c2r(fti, dftc, dftr);

                    if (d->type & 1) // spatial overlapping
                        proc1(dftr + pos * d->barea, hw + pos * d->barea, ebuff + y * ebpStride + x, d->sbsize, ebpStride);
                    else
                        ebuff[(y + d->sbd1) * ebpStride + x + d->sbd1] = dftr[pos * d->barea + d->sbd1 * d->sbsize + d->sbd1] * hw[pos * d->barea + d->sbd1 * d->sbsize + d->sbd1];
                }

                for (int q = 0; q < d->tbsize; q++)
                    srcp[q] += srcStride * d->inc;
            }

            const int dstWidth = vsapi->getFrameWidth(dst, plane);
            const int dstHeight = vsapi->getFrameHeight(dst, plane);
            const int dstStride = vsapi->getStride(dst, plane) / sizeof(pixel_t);
            pixel_t * dstp = reinterpret_cast<pixel_t *>(vsapi->getWritePtr(dst, plane));
            const float * ebp = ebuff + ebpStride * ((height - dstHeight) / 2) + (width - dstWidth) / 2;
            cast(ebp, dstp, dstWidth, dstHeight, dstStride, ebpStride, d->dstScale, d->peak);
        }
    }
}

static void VS_CC dfttestInit(VSMap * in, VSMap * out, void ** instanceData, VSNode * node, VSCore * core, const VSAPI * vsapi) {
    DFTTestData * d = static_cast<DFTTestData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef * VS_CC dfttestGetFrame(int n, int activationReason, void ** instanceData, void ** frameData, VSFrameContext * frameCtx, VSCore * core, const VSAPI * vsapi) {
    DFTTestData * d = static_cast<DFTTestData *>(*instanceData);

    if (activationReason == arInitial) {
        if (d->tbsize == 1) {
            vsapi->requestFrameFilter(n, d->node, frameCtx);
        } else {
            const int start = std::max(n - d->tbsize / 2, 0);
            const int stop = std::min(n + d->tbsize / 2, d->vi->numFrames - 1);
            for (int i = start; i <= stop; i++)
                vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        try {
            auto threadId = std::this_thread::get_id();

            if (!d->ebuff.count(threadId)) {
                d->ebuff.emplace(threadId,
                                 unique_VSFrameRef{ vsapi->newVideoFrame(vsapi->registerFormat(cmGray, stFloat, 32, 0, 0, core), d->padWidth[0], d->padHeight[0], nullptr, core),
                                                    vsapi->freeFrame });

                float * dftr = vs_aligned_malloc<float>((d->bvolume + 15) * sizeof(float), 64);
                if (!dftr)
                    throw "malloc failure (dftr)";
                d->dftr.emplace(threadId, unique_float{ dftr, vs_aligned_free });

                fftwf_complex * dftc = vs_aligned_malloc<fftwf_complex>((d->ccnt + 15) * sizeof(fftwf_complex), 64);
                if (!dftc)
                    throw "malloc failure (dftc)";
                d->dftc.emplace(threadId, unique_fftwf_complex{ dftc, vs_aligned_free });

                fftwf_complex * dftc2 = vs_aligned_malloc<fftwf_complex>((d->ccnt + 15) * sizeof(fftwf_complex), 64);
                if (!dftc2)
                    throw "malloc failure (dftc2)";
                d->dftc2.emplace(threadId, unique_fftwf_complex{ dftc2, vs_aligned_free });
            }
        } catch (const char * error) {
            vsapi->setFilterError(("DFTTest: "s + error).c_str(), frameCtx);
            return nullptr;
        }

        const VSFrameRef * src0 = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrameRef * fr[] = { d->process[0] ? nullptr : src0, d->process[1] ? nullptr : src0, d->process[2] ? nullptr : src0 };
        const int pl[] = { 0, 1, 2 };
        VSFrameRef * dst = vsapi->newVideoFrame2(d->vi->format, d->vi->width, d->vi->height, fr, pl, src0, core);
        vsapi->freeFrame(src0);

        if (d->tbsize == 1) {
            const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
            VSFrameRef * pad[3] = {};

            for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
                if (d->process[plane])
                    pad[plane] = vsapi->newVideoFrame(d->padFormat, d->padWidth[plane], d->padHeight[plane], nullptr, core);
            }

            d->copyPad(src, pad, d, vsapi);
            d->func_0(pad, dst, d, vsapi);

            vsapi->freeFrame(src);
            for (int plane = 0; plane < d->vi->format->numPlanes; plane++)
                vsapi->freeFrame(pad[plane]);
        } else {
            const VSFrameRef * src[15] = {};
            VSFrameRef * pad[15][3] = {};

            const int pos = d->tbsize / 2;

            for (int i = n - pos; i <= n + pos; i++) {
                src[i - n + pos] = vsapi->getFrameFilter(std::clamp(i, 0, d->vi->numFrames - 1), d->node, frameCtx);

                for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
                    if (d->process[plane])
                        pad[i - n + pos][plane] = vsapi->newVideoFrame(d->padFormat, d->padWidth[plane], d->padHeight[plane], nullptr, core);
                }

                d->copyPad(src[i - n + pos], pad[i - n + pos], d, vsapi);
            }

            d->func_1(pad, dst, pos, d, vsapi);

            for (int i = n - pos; i <= n + pos; i++) {
                vsapi->freeFrame(src[i - n + pos]);
                for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
                    vsapi->freeFrame(pad[i - n + pos][plane]);
                }
            }
        }

        return dst;
    }

    return nullptr;
}

static void VS_CC dfttestFree(void * instanceData, VSCore * core, const VSAPI * vsapi) {
    DFTTestData * d = static_cast<DFTTestData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

static void VS_CC dfttestCreate(const VSMap * in, VSMap * out, void * userData, VSCore * core, const VSAPI * vsapi) {
    std::unique_ptr<DFTTestData> d = std::make_unique<DFTTestData>();
    int err;

    auto createWindow = [&](unique_float & hw, const int tmode, const int smode) noexcept {
        auto getWinValue = [](const double n, const double size, const int win, const double beta) noexcept {
            auto besselI0 = [](double p) noexcept {
                p /= 2.0;
                double n = 1.0, t = 1.0, d = 1.0;
                int k = 1;
                double v;

                do {
                    n *= p;
                    d *= k;
                    v = n / d;
                    t += v * v;
                } while (++k < 15 && v > 1e-8);

                return t;
            };

            switch (win) {
            case 0: // hanning
                return 0.5 - 0.5 * std::cos(2.0 * M_PI * n / size);
            case 1: // hamming
                return 0.53836 - 0.46164 * std::cos(2.0 * M_PI * n / size);
            case 2: // blackman
                return 0.42 - 0.5 * std::cos(2.0 * M_PI * n / size) + 0.08 * std::cos(4.0 * M_PI * n / size);
            case 3: // 4 term blackman-harris
                return 0.35875 - 0.48829 * std::cos(2.0 * M_PI * n / size) + 0.14128 * std::cos(4.0 * M_PI * n / size) - 0.01168 * std::cos(6.0 * M_PI * n / size);
            case 4: // kaiser-bessel
            {
                const double v = 2.0 * n / size - 1.0;
                return besselI0(M_PI * beta * std::sqrt(1.0 - v * v)) / besselI0(M_PI * beta);
            }
            case 5: // 7 term blackman-harris
                return 0.27105140069342415 -
                       0.433297939234486060 * std::cos(2.0 * M_PI * n / size) +
                       0.218122999543110620 * std::cos(4.0 * M_PI * n / size) -
                       0.065925446388030898 * std::cos(6.0 * M_PI * n / size) +
                       0.010811742098372268 * std::cos(8.0 * M_PI * n / size) -
                       7.7658482522509342e-4 * std::cos(10.0 * M_PI * n / size) +
                       1.3887217350903198e-5 * std::cos(12.0 * M_PI * n / size);
            case 6: // flat top
                return 0.2810639 - 0.5208972 * std::cos(2.0 * M_PI * n / size) + 0.1980399 * std::cos(4.0 * M_PI * n / size);
            case 7: // rectangular
                return 1.0;
            case 8: // Bartlett
                return 2.0 / size * (size / 2.0 - std::abs(n - size / 2.0));
            case 9: // Bartlett-Hann
                return 0.62 - 0.48 * (n / size - 0.5) - 0.38 * std::cos(2.0 * M_PI * n / size);
            case 10: // Nuttall
                return 0.355768 - 0.487396 * std::cos(2.0 * M_PI * n / size) + 0.144232 * std::cos(4.0 * M_PI * n / size) - 0.012604 * std::cos(6.0 * M_PI * n / size);
            case 11: // Blackman-Nuttall
                return 0.3635819 - 0.4891775 * std::cos(2.0 * M_PI * n / size) + 0.1365995 * std::cos(4.0 * M_PI * n / size) - 0.0106411 * std::cos(6.0 * M_PI * n / size);
            default:
                return 0.0;
            }
        };

        auto normalizeForOverlapAdd = [](std::unique_ptr<double[]> & hw, const int bsize, const int osize) noexcept {
            std::unique_ptr<double[]> nw = std::make_unique<double[]>(bsize);
            const int inc = bsize - osize;

            for (int q = 0; q < bsize; q++) {
                for (int h = q; h >= 0; h -= inc)
                    nw[q] += hw[h] * hw[h];
                for (int h = q + inc; h < bsize; h += inc)
                    nw[q] += hw[h] * hw[h];
            }

            for (int q = 0; q < bsize; q++)
                hw[q] /= std::sqrt(nw[q]);
        };

        std::unique_ptr<double[]> tw = std::make_unique<double[]>(d->tbsize);
        for (int j = 0; j < d->tbsize; j++)
            tw[j] = getWinValue(j + 0.5, d->tbsize, d->twin, d->tbeta);
        if (tmode == 1)
            normalizeForOverlapAdd(tw, d->tbsize, d->tosize);

        std::unique_ptr<double[]> sw = std::make_unique<double[]>(d->sbsize);
        for (int j = 0; j < d->sbsize; j++)
            sw[j] = getWinValue(j + 0.5, d->sbsize, d->swin, d->sbeta);
        if (smode == 1)
            normalizeForOverlapAdd(sw, d->sbsize, d->sosize);

        const double nscale = 1.0 / std::sqrt(d->bvolume);
        for (int j = 0; j < d->tbsize; j++)
            for (int k = 0; k < d->sbsize; k++)
                for (int q = 0; q < d->sbsize; q++)
                    hw[(j * d->sbsize + k) * d->sbsize + q] = static_cast<float>(tw[j] * sw[k] * sw[q] * nscale);
    };

    auto interp = [](const float pf, const std::unique_ptr<float[]> & pv, const int cnt) noexcept {
        int lidx = 0;
        for (int i = cnt - 1; i >= 0; i--) {
            if (pv[i * 2] <= pf) {
                lidx = i;
                break;
            }
        }

        int hidx = cnt - 1;
        for (int i = 0; i < cnt; i++) {
            if (pv[i * 2] >= pf) {
                hidx = i;
                break;
            }
        }

        const float d0 = pf - pv[lidx * 2];
        const float d1 = pv[hidx * 2] - pf;

        if (hidx == lidx || d0 <= 0.0f)
            return pv[lidx * 2 + 1];
        if (d1 <= 0.0f)
            return pv[hidx * 2 + 1];

        const float tf = d0 / (d0 + d1);
        return pv[lidx * 2 + 1] * (1.0f - tf) + pv[hidx * 2 + 1] * tf;
    };

    auto getSVal = [&](const int pos, const int len, const std::unique_ptr<float[]> & pv, const int cnt, float & pf) noexcept {
        if (len == 1) {
            pf = 0.0f;
            return 1.0f;
        }

        const int ld2 = len / 2;
        pf = (pos > ld2 ? len - pos : pos) / static_cast<float>(ld2);
        return interp(pf, pv, cnt);
    };

    try {
        d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->node);

        if (!isConstantFormat(d->vi) ||
            (d->vi->format->sampleType == stInteger && d->vi->format->bitsPerSample > 16) ||
            (d->vi->format->sampleType == stFloat && d->vi->format->bitsPerSample != 32))
            throw "only constant format 8-16 bit integer and 32 bit float input supported";

        const int ftype = getArg(vsapi, in, "ftype", 0);
        const float sigma = getArg(vsapi, in, "sigma", 8.0f);
        const float sigma2 = getArg(vsapi, in, "sigma2", 8.0f);
        const float pmin = getArg(vsapi, in, "pmin", 0.0f);
        const float pmax = getArg(vsapi, in, "pmax", 500.0f);
        d->sbsize = getArg(vsapi, in, "sbsize", 16);
        const int smode = getArg(vsapi, in, "smode", 1);
        d->sosize = getArg(vsapi, in, "sosize", 12);
        d->tbsize = getArg(vsapi, in, "tbsize", 3);
        const int tmode = getArg(vsapi, in, "tmode", 0);
        d->tosize = getArg(vsapi, in, "tosize", 0);
        d->swin = getArg(vsapi, in, "swin", 0);
        d->twin = getArg(vsapi, in, "twin", 7);
        d->sbeta = getArg(vsapi, in, "sbeta", 2.5);
        d->tbeta = getArg(vsapi, in, "tbeta", 2.5);
        d->zmean = getArg(vsapi, in, "zmean", true);
        d->f0beta = getArg(vsapi, in, "f0beta", 1.0f);
        const float alpha = getArg(vsapi, in, "alpha", ftype == 0 ? 5.0f : 7.0f);
        const int ssystem = getArg(vsapi, in, "ssystem", 0);
        const int opt = getArg(vsapi, in, "opt", 0);

        const int64_t * nlocation = vsapi->propGetIntArray(in, "nlocation", &err);
        const double * slocation = vsapi->propGetFloatArray(in, "slocation", &err);
        const double * ssx = vsapi->propGetFloatArray(in, "ssx", &err);
        const double * ssy = vsapi->propGetFloatArray(in, "ssy", &err);
        const double * sst = vsapi->propGetFloatArray(in, "sst", &err);

        const int numNlocation = vsapi->propNumElements(in, "nlocation");
        const int numSlocation = vsapi->propNumElements(in, "slocation");
        const int numSsx = vsapi->propNumElements(in, "ssx");
        const int numSsy = vsapi->propNumElements(in, "ssy");
        const int numSst = vsapi->propNumElements(in, "sst");

        {
            const int m = vsapi->propNumElements(in, "planes");

            for (int i = 0; i < 3; i++)
                d->process[i] = (m <= 0);

            for (int i = 0; i < m; i++) {
                const int n = int64ToIntS(vsapi->propGetInt(in, "planes", i, nullptr));

                if (n < 0 || n >= d->vi->format->numPlanes)
                    throw "plane index out of range";

                if (d->process[n])
                    throw "plane specified twice";

                d->process[n] = true;
            }
        }

        if (ftype < 0 || ftype > 4)
            throw "ftype must be 0, 1, 2, 3, or 4";

        if (d->sbsize < 1)
            throw "sbsize must be greater than or equal to 1";

        if (smode < 0 || smode > 1)
            throw "smode must be 0 or 1";

        if (smode == 0 && !(d->sbsize & 1))
            throw "sbsize must be odd when using smode=0";

        if (smode == 0)
            d->sosize = 0;

        if (d->sosize < 0 || d->sosize >= d->sbsize)
            throw "sosize must be between 0 and sbsize-1 (inclusive)";

        if (d->sosize > d->sbsize / 2 && d->sbsize % (d->sbsize - d->sosize) != 0)
            throw "spatial overlap greater than 50% requires that sbsize-sosize is a divisor of sbsize";

        if (d->tbsize < 1 || d->tbsize > 15)
            throw "tbsize must be between 1 and 15 (inclusive)";

        if (tmode != 0)
            throw "tmode must be 0. tmode=1 is not implemented";

        if (tmode == 0 && !(d->tbsize & 1))
            throw "tbsize must be odd when using tmode=0";

        if (tmode == 0)
            d->tosize = 0;

        if (d->tosize < 0 || d->tosize >= d->tbsize)
            throw "tosize must be between 0 and tbsize-1 (inclusive)";

        if (d->tosize > d->tbsize / 2 && d->tbsize % (d->tbsize - d->tosize) != 0)
            throw "temporal overlap greater than 50% requires that tbsize-tosize is a divisor of tbsize";

        if (d->tbsize > d->vi->numFrames)
            throw "tbsize must be less than or equal to the number of frames in the clip";

        if (d->swin < 0 || d->swin > 11)
            throw "swin must be between 0 and 11 (inclusive)";

        if (d->twin < 0 || d->twin > 11)
            throw "twin must be between 0 and 11 (inclusive)";

        if (nlocation && (numNlocation & 3))
            throw "number of elements in nlocation must be a multiple of 4";

        if (alpha <= 0.0f)
            throw "alpha must be greater than 0.0";

        if (slocation && (numSlocation & 1))
            throw "number of elements in slocation must be a multiple of 2";

        if (ssx && (numSsx & 1))
            throw "number of elements in ssx must be a multiple of 2";

        if (ssy && (numSsy & 1))
            throw "number of elements in ssy must be a multiple of 2";

        if (sst && (numSst & 1))
            throw "number of elements in sst must be a multiple of 2";

        if (ssystem < 0 || ssystem > 1)
            throw "ssystem must be 0 or 1";

        if (opt < 0 || opt > 4)
            throw "opt must be 0, 1, 2, 3, or 4";

        {
            if (ftype == 0) {
                if (std::abs(d->f0beta - 1.0f) < 0.00005f)
                    d->filterCoeffs = filter_c<0>;
                else if (std::abs(d->f0beta - 0.5f) < 0.00005f)
                    d->filterCoeffs = filter_c<6>;
                else
                    d->filterCoeffs = filter_c<5>;
            } else if (ftype == 1) {
                d->filterCoeffs = filter_c<1>;
            } else if (ftype == 2) {
                d->filterCoeffs = filter_c<2>;
            } else if (ftype == 3) {
                d->filterCoeffs = filter_c<3>;
            } else {
                d->filterCoeffs = filter_c<4>;
            }

            if (d->vi->format->bytesPerSample == 1) {
                d->copyPad = copyPad<uint8_t>;
                d->func_0 = func_0_c<uint8_t>;
                d->func_1 = func_1_c<uint8_t>;
            } else if (d->vi->format->bytesPerSample == 2) {
                d->copyPad = copyPad<uint16_t>;
                d->func_0 = func_0_c<uint16_t>;
                d->func_1 = func_1_c<uint16_t>;
            } else {
                d->copyPad = copyPad<float>;
                d->func_0 = func_0_c<float>;
                d->func_1 = func_1_c<float>;
            }

#ifdef DFTTEST_X86
            const int iset = instrset_detect();
            if ((opt == 0 && iset >= 10) || opt == 4) {
                if (ftype == 0) {
                    if (std::abs(d->f0beta - 1.0f) < 0.00005f)
                        d->filterCoeffs = filter_avx512<0>;
                    else if (std::abs(d->f0beta - 0.5f) < 0.00005f)
                        d->filterCoeffs = filter_avx512<6>;
                    else
                        d->filterCoeffs = filter_avx512<5>;
                } else if (ftype == 1) {
                    d->filterCoeffs = filter_avx512<1>;
                } else if (ftype == 2) {
                    d->filterCoeffs = filter_avx512<2>;
                } else if (ftype == 3) {
                    d->filterCoeffs = filter_avx512<3>;
                } else {
                    d->filterCoeffs = filter_avx512<4>;
                }

                if (d->vi->format->bytesPerSample == 1) {
                    d->func_0 = func_0_avx512<uint8_t>;
                    d->func_1 = func_1_avx512<uint8_t>;
                } else if (d->vi->format->bytesPerSample == 2) {
                    d->func_0 = func_0_avx512<uint16_t>;
                    d->func_1 = func_1_avx512<uint16_t>;
                } else {
                    d->func_0 = func_0_avx512<float>;
                    d->func_1 = func_1_avx512<float>;
                }
            } else if ((opt == 0 && iset >= 8) || opt == 3) {
                if (ftype == 0) {
                    if (std::abs(d->f0beta - 1.0f) < 0.00005f)
                        d->filterCoeffs = filter_avx2<0>;
                    else if (std::abs(d->f0beta - 0.5f) < 0.00005f)
                        d->filterCoeffs = filter_avx2<6>;
                    else
                        d->filterCoeffs = filter_avx2<5>;
                } else if (ftype == 1) {
                    d->filterCoeffs = filter_avx2<1>;
                } else if (ftype == 2) {
                    d->filterCoeffs = filter_avx2<2>;
                } else if (ftype == 3) {
                    d->filterCoeffs = filter_avx2<3>;
                } else {
                    d->filterCoeffs = filter_avx2<4>;
                }

                if (d->vi->format->bytesPerSample == 1) {
                    d->func_0 = func_0_avx2<uint8_t>;
                    d->func_1 = func_1_avx2<uint8_t>;
                } else if (d->vi->format->bytesPerSample == 2) {
                    d->func_0 = func_0_avx2<uint16_t>;
                    d->func_1 = func_1_avx2<uint16_t>;
                } else {
                    d->func_0 = func_0_avx2<float>;
                    d->func_1 = func_1_avx2<float>;
                }
            } else if ((opt == 0 && iset >= 2) || opt == 2) {
                if (ftype == 0) {
                    if (std::abs(d->f0beta - 1.0f) < 0.00005f)
                        d->filterCoeffs = filter_sse2<0>;
                    else if (std::abs(d->f0beta - 0.5f) < 0.00005f)
                        d->filterCoeffs = filter_sse2<6>;
                    else
                        d->filterCoeffs = filter_sse2<5>;
                } else if (ftype == 1) {
                    d->filterCoeffs = filter_sse2<1>;
                } else if (ftype == 2) {
                    d->filterCoeffs = filter_sse2<2>;
                } else if (ftype == 3) {
                    d->filterCoeffs = filter_sse2<3>;
                } else {
                    d->filterCoeffs = filter_sse2<4>;
                }

                if (d->vi->format->bytesPerSample == 1) {
                    d->func_0 = func_0_sse2<uint8_t>;
                    d->func_1 = func_1_sse2<uint8_t>;
                } else if (d->vi->format->bytesPerSample == 2) {
                    d->func_0 = func_0_sse2<uint16_t>;
                    d->func_1 = func_1_sse2<uint16_t>;
                } else {
                    d->func_0 = func_0_sse2<float>;
                    d->func_1 = func_1_sse2<float>;
                }
            }
#endif
        }

        if (d->vi->format->sampleType == stInteger) {
            d->dstScale = static_cast<float>(1 << (d->vi->format->bitsPerSample - 8));
            d->srcScale = 1.0f / d->dstScale;
            d->peak = (1 << d->vi->format->bitsPerSample) - 1;
        } else {
            d->srcScale = 255.0f;
            d->dstScale = 1.0f / 255.0f;
        }

        if (ftype != 0)
            d->f0beta = 1.0f;

        d->barea = d->sbsize * d->sbsize;
        d->bvolume = d->barea * d->tbsize;
        d->ccnt = (d->sbsize / 2 + 1) * d->sbsize * d->tbsize;
        d->ccnt2 = d->ccnt * 2;
        d->type = tmode * 4 + (d->tbsize > 1 ? 2 : 0) + smode;
        d->sbd1 = d->sbsize / 2;
        d->uf0b = (std::abs(d->f0beta - 1.0f) < 0.00005f) ? false : true;
        d->inc = (d->type & 1) ? d->sbsize - d->sosize : 1;

        d->padFormat = vsapi->registerFormat(cmGray, d->vi->format->sampleType, d->vi->format->bitsPerSample, 0, 0, core);
        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            const int width = d->vi->width >> (plane ? d->vi->format->subSamplingW : 0);
            const int height = d->vi->height >> (plane ? d->vi->format->subSamplingH : 0);

            if (smode == 0) {
                const int ae = (d->sbsize >> 1) << 1;
                d->padWidth[plane] = width + ae;
                d->padHeight[plane] = height + ae;
                d->eheight[plane] = height;
            } else {
                const int ae = std::max(d->sbsize - d->sosize, d->sosize) * 2;
                d->padWidth[plane] = width + EXTRA(width, d->sbsize) + ae;
                d->padHeight[plane] = height + EXTRA(height, d->sbsize) + ae;
                d->eheight[plane] = (d->padHeight[plane] - d->sosize) / (d->sbsize - d->sosize) * (d->sbsize - d->sosize);
            }
        }

        d->hw = { vs_aligned_malloc<float>((d->bvolume + 15) * sizeof(float), 64), vs_aligned_free };
        if (!d->hw)
            throw "malloc failure (hw)";

        createWindow(d->hw, tmode, smode);

        unique_float dftgr{ vs_aligned_malloc<float>((d->bvolume + 15) * sizeof(float), 64), vs_aligned_free };
        d->dftgc = { vs_aligned_malloc<fftwf_complex>((d->ccnt + 15) * sizeof(fftwf_complex), 64), vs_aligned_free };
        if (!dftgr || !d->dftgc)
            throw "malloc failure (dftgr/dftgc)";

        fftwf_make_planner_thread_safe();

        if (d->tbsize > 1) {
            d->ft = { fftwf_plan_dft_r2c_3d(d->tbsize, d->sbsize, d->sbsize, dftgr.get(), d->dftgc.get(), FFTW_PATIENT | FFTW_DESTROY_INPUT), fftwf_destroy_plan };
            d->fti = { fftwf_plan_dft_c2r_3d(d->tbsize, d->sbsize, d->sbsize, d->dftgc.get(), dftgr.get(), FFTW_PATIENT | FFTW_DESTROY_INPUT), fftwf_destroy_plan };
        } else {
            d->ft = { fftwf_plan_dft_r2c_2d(d->sbsize, d->sbsize, dftgr.get(), d->dftgc.get(), FFTW_PATIENT | FFTW_DESTROY_INPUT), fftwf_destroy_plan };
            d->fti = { fftwf_plan_dft_c2r_2d(d->sbsize, d->sbsize, d->dftgc.get(), dftgr.get(), FFTW_PATIENT | FFTW_DESTROY_INPUT), fftwf_destroy_plan };
        }

        float wscale = 0.0f;

        const float * hwT = d->hw.get();
        float * VS_RESTRICT dftgrT = dftgr.get();
        for (int s = 0; s < d->tbsize; s++) {
            for (int i = 0; i < d->sbsize; i++) {
                for (int k = 0; k < d->sbsize; k++) {
                    dftgrT[k] = 255.0f * hwT[k];
                    wscale += hwT[k] * hwT[k];
                }
                hwT += d->sbsize;
                dftgrT += d->sbsize;
            }
        }

        wscale = 1.0f / wscale;
        const float wscalef = (ftype < 2) ? wscale : 1.0f;

        fftwf_execute_dft_r2c(d->ft.get(), dftgr.get(), d->dftgc.get());

        d->sigmas = { vs_aligned_malloc<float>((d->ccnt2 + 15) * sizeof(float), 64), vs_aligned_free };
        d->sigmas2 = { vs_aligned_malloc<float>((d->ccnt2 + 15) * sizeof(float), 64), vs_aligned_free };
        d->pmins = { vs_aligned_malloc<float>((d->ccnt2 + 15) * sizeof(float), 64), vs_aligned_free };
        d->pmaxs = { vs_aligned_malloc<float>((d->ccnt2 + 15) * sizeof(float), 64), vs_aligned_free };
        if (!d->sigmas || !d->sigmas2 || !d->pmins || !d->pmaxs)
            throw "malloc failure (sigmas/sigmas2/pmins/pmaxs)";

        if (slocation || ssx || ssy || sst) {
            auto parseSigmaLocation = [&](const double * s, const int num, int & poscnt, const float pfact) {
                float * parray = nullptr;

                if (!s) {
                    parray = new float[4];
                    parray[0] = 0.0f;
                    parray[2] = 1.0f;
                    parray[1] = parray[3] = std::pow(sigma, pfact);
                    poscnt = 2;
                } else {
                    const double * sT = s;
                    bool found[] = { false, false };
                    poscnt = 0;

                    for (int i = 0; i < num; i += 2) {
                        const float pos = static_cast<float>(sT[i]);

                        if (pos < 0.0f || pos > 1.0f)
                            throw "sigma location - invalid pos (" + std::to_string(pos) + ")";

                        if (pos == 0.0f)
                            found[0] = true;
                        else if (pos == 1.0f)
                            found[1] = true;

                        poscnt++;
                    }

                    if (!found[0] || !found[1])
                        throw "sigma location - one or more end points not provided";

                    parray = new float[poscnt * 2];
                    sT = s;
                    poscnt = 0;

                    for (int i = 0; i < num; i += 2) {
                        parray[poscnt * 2 + 0] = static_cast<float>(sT[i + 0]);
                        parray[poscnt * 2 + 1] = std::pow(static_cast<float>(sT[i + 1]), pfact);

                        poscnt++;
                    }

                    for (int i = 1; i < poscnt; i++) {
                        int j = i;
                        const float t0 = parray[j * 2 + 0];
                        const float t1 = parray[j * 2 + 1];

                        while (j > 0 && parray[(j - 1) * 2] > t0) {
                            parray[j * 2 + 0] = parray[(j - 1) * 2 + 0];
                            parray[j * 2 + 1] = parray[(j - 1) * 2 + 1];
                            j--;
                        }

                        parray[j * 2 + 0] = t0;
                        parray[j * 2 + 1] = t1;
                    }
                }

                return parray;
            };

            int ndim = 3;
            if (d->tbsize == 1)
                ndim -= 1;
            if (d->sbsize == 1)
                ndim -= 2;

            const float ndiv = 1.0f / ndim;
            int tcnt = 0, sycnt = 0, sxcnt = 0;
            std::unique_ptr<float[]> tdata, sydata, sxdata;

            if (slocation) {
                tdata = std::unique_ptr<float[]>{ parseSigmaLocation(slocation, numSlocation, tcnt, ssystem ? 1.0f : ndiv) };
                sydata = std::unique_ptr<float[]>{ parseSigmaLocation(slocation, numSlocation, sycnt, ssystem ? 1.0f : ndiv) };
                sxdata = std::unique_ptr<float[]>{ parseSigmaLocation(slocation, numSlocation, sxcnt, ssystem ? 1.0f : ndiv) };
            } else {
                tdata = std::unique_ptr<float[]>{ parseSigmaLocation(sst, numSst, tcnt, ndiv) };
                sydata = std::unique_ptr<float[]>{ parseSigmaLocation(ssy, numSsy, sycnt, ndiv) };
                sxdata = std::unique_ptr<float[]>{ parseSigmaLocation(ssx, numSsx, sxcnt, ndiv) };
            }

            const int cpx = d->sbsize / 2 + 1;
            float pft, pfy, pfx;

            for (int z = 0; z < d->tbsize; z++) {
                const float tval = getSVal(z, d->tbsize, tdata, tcnt, pft);

                for (int y = 0; y < d->sbsize; y++) {
                    const float syval = getSVal(y, d->sbsize, sydata, sycnt, pfy);

                    for (int x = 0; x < cpx; x++) {
                        const float sxval = getSVal(x, d->sbsize, sxdata, sxcnt, pfx);
                        float val;

                        if (ssystem) {
                            const float dw = std::sqrt((pft * pft + pfy * pfy + pfx * pfx) / ndim);
                            val = interp(dw, tdata, tcnt);
                        } else {
                            val = tval * syval * sxval;
                        }

                        const int pos = ((z * d->sbsize + y) * cpx + x) * 2;
                        d->sigmas[pos + 0] = d->sigmas[pos + 1] = val / wscalef;
                    }
                }
            }
        } else {
            for (int i = 0; i < d->ccnt2; i++)
                d->sigmas[i] = sigma / wscalef;
        }

        for (int i = 0; i < d->ccnt2; i++) {
            d->sigmas2[i] = sigma2 / wscalef;
            d->pmins[i] = pmin / wscale;
            d->pmaxs[i] = pmax / wscale;
        }

        if (nlocation && ftype < 2) {
            struct NPInfo final {
                int fn, b, y, x;
            };

            memset(d->sigmas.get(), 0, d->ccnt2 * sizeof(float));

            unique_float hw2{ vs_aligned_malloc<float>((d->bvolume + 15) * sizeof(float), 64), vs_aligned_free };
            if (!hw2)
                throw "malloc failure (hw2)";

            createWindow(hw2, 0, 0);

            unique_float dftr{ vs_aligned_malloc<float>((d->bvolume + 15) * sizeof(float), 64), vs_aligned_free };
            unique_fftwf_complex dftgc2{ vs_aligned_malloc<fftwf_complex>((d->ccnt + 15) * sizeof(fftwf_complex), 64), vs_aligned_free };
            if (!dftr || !dftgc2)
                throw "malloc failure (dftr/dftgc2)";

            float wscale2 = 0.0f;
            int w = 0;
            for (int s = 0; s < d->tbsize; s++) {
                for (int i = 0; i < d->sbsize; i++) {
                    for (int k = 0; k < d->sbsize; k++, w++) {
                        dftr[w] = 255.0f * hw2[w];
                        wscale2 += hw2[w] * hw2[w];
                    }
                }
            }
            wscale2 = 1.0f / wscale2;
            fftwf_execute_dft_r2c(d->ft.get(), dftr.get(), dftgc2.get());

            int nnpoints = 0;
            std::unique_ptr<NPInfo[]> npts = std::make_unique<NPInfo[]>(500);

            for (int i = 0; i < numNlocation; i += 4) {
                const int fn = int64ToIntS(nlocation[i + 0]);
                const int b = int64ToIntS(nlocation[i + 1]);
                const int y = int64ToIntS(nlocation[i + 2]);
                const int x = int64ToIntS(nlocation[i + 3]);

                if (fn < 0 || fn > d->vi->numFrames - d->tbsize)
                    throw "invalid frame number in nlocation (" + std::to_string(fn) + ")";

                if (b < 0 || b >= d->vi->format->numPlanes)
                    throw "invalid plane number in nlocation (" + std::to_string(b) + ")";

                const int height = d->vi->height >> (b ? d->vi->format->subSamplingH : 0);
                if (y < 0 || y > height - d->sbsize)
                    throw "invalid y pos in nlocation (" + std::to_string(y) + ")";

                const int width = d->vi->width >> (b ? d->vi->format->subSamplingW : 0);
                if (x < 0 || x > width - d->sbsize)
                    throw "invalid x pos in nlocation (" + std::to_string(x) + ")";

                if (nnpoints >= 500)
                    throw "maximum number of entries in nlocation is 500";

                npts[nnpoints].fn = fn;
                npts[nnpoints].b = b;
                npts[nnpoints].y = y;
                npts[nnpoints].x = x;
                nnpoints++;
            }

            for (int ct = 0; ct < nnpoints; ct++) {
                unique_fftwf_complex _dftc{ vs_aligned_malloc<fftwf_complex>((d->ccnt + 15) * sizeof(fftwf_complex), 64), vs_aligned_free };
                unique_fftwf_complex dftc2{ vs_aligned_malloc<fftwf_complex>((d->ccnt + 15) * sizeof(fftwf_complex), 64), vs_aligned_free };
                if (!_dftc || !dftc2)
                    throw "malloc failure (dftc/dftc2)";

                float * dftc = reinterpret_cast<float *>(_dftc.get());

                for (int z = 0; z < d->tbsize; z++) {
                    const VSFrameRef * src = vsapi->getFrame(npts[ct].fn + z, d->node, nullptr, 0);
                    const int stride = vsapi->getStride(src, npts[ct].b) / d->vi->format->bytesPerSample;

                    if (d->vi->format->bytesPerSample == 1) {
                        const uint8_t * srcp = vsapi->getReadPtr(src, npts[ct].b) + stride * npts[ct].y + npts[ct].x;
                        proc0(srcp, hw2.get() + d->barea * z, dftr.get() + d->barea * z, stride, d->sbsize, d->srcScale);
                    } else if (d->vi->format->bytesPerSample == 2) {
                        const uint16_t * srcp = reinterpret_cast<const uint16_t *>(vsapi->getReadPtr(src, npts[ct].b)) + stride * npts[ct].y + npts[ct].x;
                        proc0(srcp, hw2.get() + d->barea * z, dftr.get() + d->barea * z, stride, d->sbsize, d->srcScale);
                    } else {
                        const float * srcp = reinterpret_cast<const float *>(vsapi->getReadPtr(src, npts[ct].b)) + stride * npts[ct].y + npts[ct].x;
                        proc0(srcp, hw2.get() + d->barea * z, dftr.get() + d->barea * z, stride, d->sbsize, d->srcScale);
                    }

                    vsapi->freeFrame(src);
                }

                fftwf_execute_dft_r2c(d->ft.get(), dftr.get(), reinterpret_cast<fftwf_complex *>(dftc));

                if (d->zmean)
                    removeMean(dftc, reinterpret_cast<const float *>(dftgc2.get()), d->ccnt2, reinterpret_cast<float *>(dftc2.get()));

                for (int h = 0; h < d->ccnt2; h += 2) {
                    const float psd = dftc[h + 0] * dftc[h + 0] + dftc[h + 1] * dftc[h + 1];
                    d->sigmas[h + 0] += psd;
                    d->sigmas[h + 1] += psd;
                }
            }

            const float scale = 1.0f / nnpoints;
            for (int h = 0; h < d->ccnt2; h++)
                d->sigmas[h] = d->sigmas[h] * scale * (wscale2 / wscale) * alpha;
        }

        const unsigned numThreads = vsapi->getCoreInfo(core)->numThreads;
        d->ebuff.reserve(numThreads);
        d->dftr.reserve(numThreads);
        d->dftc.reserve(numThreads);
        d->dftc2.reserve(numThreads);
    } catch (const char * error) {
        vsapi->setError(out, ("DFTTest: "s + error).c_str());
        vsapi->freeNode(d->node);
        return;
    } catch (const std::string & error) {
        vsapi->setError(out, ("DFTTest: " + error).c_str());
        vsapi->freeNode(d->node);
        return;
    }

    vsapi->createFilter(in, out, "DFTTest", dfttestInit, dfttestGetFrame, dfttestFree, fmParallel, 0, d.release(), core);
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin * plugin) {
    configFunc("com.holywu.dfttest", "dfttest", "2D/3D frequency domain denoiser", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("DFTTest",
                 "clip:clip;"
                 "ftype:int:opt;"
                 "sigma:float:opt;"
                 "sigma2:float:opt;"
                 "pmin:float:opt;"
                 "pmax:float:opt;"
                 "sbsize:int:opt;"
                 "smode:int:opt;"
                 "sosize:int:opt;"
                 "tbsize:int:opt;"
                 "tmode:int:opt;"
                 "tosize:int:opt;"
                 "swin:int:opt;"
                 "twin:int:opt;"
                 "sbeta:float:opt;"
                 "tbeta:float:opt;"
                 "zmean:int:opt;"
                 "f0beta:float:opt;"
                 "nlocation:int[]:opt;"
                 "alpha:float:opt;"
                 "slocation:float[]:opt;"
                 "ssx:float[]:opt;"
                 "ssy:float[]:opt;"
                 "sst:float[]:opt;"
                 "ssystem:int:opt;"
                 "planes:int[]:opt;"
                 "opt:int:opt;",
                 dfttestCreate, nullptr, plugin);
}

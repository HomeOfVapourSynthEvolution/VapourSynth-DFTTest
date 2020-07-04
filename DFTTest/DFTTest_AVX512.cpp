#ifdef DFTTEST_X86
#include "DFTTest.h"

#include "VCL2/vectormath_exp.h"

template<typename pixel_t>
static inline auto proc0(const pixel_t * _s0, const float * _s1, float * d, const int p0, const int p1, const float srcScale) noexcept {
    for (int u = 0; u < p1; u++) {
        for (int v = 0; v < p1; v += Vec16f().size()) {
            Vec16f s0;

            if constexpr (std::is_same_v<pixel_t, uint8_t>)
                s0 = to_float(Vec16i().load_16uc(_s0 + v));
            else if constexpr (std::is_same_v<pixel_t, uint16_t>)
                s0 = to_float(Vec16i().load_16us(_s0 + v));
            else
                s0 = Vec16f().load(_s0 + v);

            const Vec16f s1 = Vec16f().load(_s1 + v);

            if constexpr (std::is_same_v<pixel_t, uint8_t>)
                (s0 * s1).store(d + v);
            else
                (s0 * srcScale * s1).store(d + v);
        }

        _s0 += p0;
        _s1 += p1;
        d += p1;
    }
}

static inline auto proc1(const float * _s0, const float * _s1, float * _d, const int p0, const int p1) noexcept {
    for (int u = 0; u < p0; u++) {
        for (int v = 0; v < p0; v += Vec16f().size()) {
            const Vec16f s0 = Vec16f().load(_s0 + v);
            const Vec16f s1 = Vec16f().load(_s1 + v);
            const Vec16f d = Vec16f().load(_d + v);
            mul_add(s0, s1, d).store(_d + v);
        }

        _s0 += p0;
        _s1 += p0;
        _d += p1;
    }
}

static inline auto proc1Partial(const float * _s0, const float * _s1, float * _d, const int p0, const int p1) noexcept {
    const int regularPart = p0 & ~(Vec16f().size() - 1);

    for (int u = 0; u < p0; u++) {
        int v;

        for (v = 0; v < regularPart; v += Vec16f().size()) {
            const Vec16f s0 = Vec16f().load(_s0 + v);
            const Vec16f s1 = Vec16f().load(_s1 + v);
            const Vec16f d = Vec16f().load(_d + v);
            mul_add(s0, s1, d).store(_d + v);
        }

        const Vec16f s0 = Vec16f().load(_s0 + v);
        const Vec16f s1 = Vec16f().load(_s1 + v);
        const Vec16f d = Vec16f().load(_d + v);
        mul_add(s0, s1, d).store_partial(p0 - v, _d + v);

        _s0 += p0;
        _s1 += p0;
        _d += p1;
    }
}

static inline auto removeMean(float * _dftc, const float * _dftgc, const int ccnt, float * _dftc2) noexcept {
    const Vec16f gf = _dftc[0] / _dftgc[0];

    for (int h = 0; h < ccnt; h += Vec16f().size()) {
        const Vec16f dftgc = Vec16f().load_a(_dftgc + h);
        const Vec16f dftc = Vec16f().load_a(_dftc + h);
        const Vec16f dftc2 = gf * dftgc;
        dftc2.store_a(_dftc2 + h);
        (dftc - dftc2).store_a(_dftc + h);
    }
}

static inline auto addMean(float * _dftc, const int ccnt, const float * _dftc2) noexcept {
    for (int h = 0; h < ccnt; h += Vec16f().size()) {
        const Vec16f dftc = Vec16f().load_a(_dftc + h);
        const Vec16f dftc2 = Vec16f().load_a(_dftc2 + h);
        (dftc + dftc2).store_a(_dftc + h);
    }
}

template<int type>
inline void filter_avx512(float * _dftc, const float * _sigmas, const int ccnt, const float * _pmin, const float * _pmax, const float * _sigmas2) noexcept {
    const Vec16f beta = _pmin[0];

    for (int h = 0; h < ccnt; h += Vec16f().size()) {
        Vec16f dftc, psd, sigmas, pmin, pmax, mult;

        dftc = Vec16f().load_a(_dftc + h);
        sigmas = Vec16f().load_a(_sigmas + h);

        if constexpr (type != 2) {
            const Vec16f dftcSquare = dftc * dftc;
            psd = dftcSquare + permute16<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>(dftcSquare);
        }

        if constexpr (type == 3 || type == 4) {
            pmin = Vec16f().load_a(_pmin + h);
            pmax = Vec16f().load_a(_pmax + h);
        }

        if constexpr (type == 0) {
            mult = max((psd - sigmas) * rcp_nr(psd + 1e-15f), zero_16f());
        } else if constexpr (type == 1) {
            dftc = select(psd < sigmas, zero_16f(), dftc);
        } else if constexpr (type == 2) {
            dftc *= sigmas;
        } else if constexpr (type == 3) {
            const Vec16f sigmas2 = Vec16f().load_a(_sigmas2 + h);
            dftc = select(psd >= pmin && psd <= pmax, dftc * sigmas, dftc * sigmas2);
        } else if constexpr (type == 4) {
            mult = sigmas * sqrt(psd * pmax * rcp_nr(mul_add(psd + pmin, psd + pmax, 1e-15f)));
        } else if constexpr (type == 5) {
            mult = pow(max((psd - sigmas) * rcp_nr(psd + 1e-15f), zero_16f()), beta);
        } else {
            mult = sqrt(max((psd - sigmas) * rcp_nr(psd + 1e-15f), zero_16f()));
        }

        if constexpr (type == 0 || type > 3)
            dftc *= mult;

        dftc.store_a(_dftc + h);
    }
}

template<typename pixel_t>
static auto cast(const float * ebp, pixel_t * dstp, const int dstWidth, const int dstHeight, const int dstStride, const int ebpStride, const float dstScale, const int peak) noexcept {
    for (int y = 0; y < dstHeight; y++) {
        for (int x = 0; x < dstWidth; x += Vec16f().size()) {
            if constexpr (std::is_same_v<pixel_t, uint8_t>) {
                const Vec16i srcp = truncatei(Vec16f().load(ebp + x) + 0.5f);
                const auto result = compress_saturated_s2u(compress_saturated(srcp, zero_si512()), zero_si512()).get_low().get_low();
                result.store_nt(dstp + x);
            } else if constexpr (std::is_same_v<pixel_t, uint16_t>) {
                const Vec16i srcp = truncatei(mul_add(Vec16f().load(ebp + x), dstScale, 0.5f));
                const auto result = compress_saturated_s2u(srcp, zero_si512()).get_low();
                min(result, peak).store_nt(dstp + x);
            } else {
                const Vec16f srcp = Vec16f().load(ebp + x) * dstScale;
                srcp.store_nt(dstp + x);
            }
        }

        ebp += ebpStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
void func_0_avx512(VSFrameRef * src[3], VSFrameRef * dst, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept {
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

                    if (d->type & 1) { // spatial overlapping
                        if (!(d->sbsize & (Vec16f().size() - 1)))
                            proc1(dftr, hw, ebpSaved + x, d->sbsize, ebpStride);
                        else
                            proc1Partial(dftr, hw, ebpSaved + x, d->sbsize, ebpStride);
                    } else {
                        ebpSaved[x + d->sbd1 * ebpStride + d->sbd1] = dftr[d->sbd1 * d->sbsize + d->sbd1] * hw[d->sbd1 * d->sbsize + d->sbd1];
                    }
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
void func_1_avx512(VSFrameRef * src[15][3], VSFrameRef * dst, const int pos, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept {
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

                    if (d->type & 1) { // spatial overlapping
                        if (!(d->sbsize & (Vec16f().size() - 1)))
                            proc1(dftr + pos * d->barea, hw + pos * d->barea, ebuff + y * ebpStride + x, d->sbsize, ebpStride);
                        else
                            proc1Partial(dftr + pos * d->barea, hw + pos * d->barea, ebuff + y * ebpStride + x, d->sbsize, ebpStride);
                    } else {
                        ebuff[(y + d->sbd1) * ebpStride + x + d->sbd1] = dftr[pos * d->barea + d->sbd1 * d->sbsize + d->sbd1] * hw[pos * d->barea + d->sbd1 * d->sbsize + d->sbd1];
                    }
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

template void filter_avx512<0>(float * dftc, const float * sigmas, const int ccnt, const float * pmin, const float * pmax, const float * sigmas2) noexcept;
template void filter_avx512<1>(float * dftc, const float * sigmas, const int ccnt, const float * pmin, const float * pmax, const float * sigmas2) noexcept;
template void filter_avx512<2>(float * dftc, const float * sigmas, const int ccnt, const float * pmin, const float * pmax, const float * sigmas2) noexcept;
template void filter_avx512<3>(float * dftc, const float * sigmas, const int ccnt, const float * pmin, const float * pmax, const float * sigmas2) noexcept;
template void filter_avx512<4>(float * dftc, const float * sigmas, const int ccnt, const float * pmin, const float * pmax, const float * sigmas2) noexcept;
template void filter_avx512<5>(float * dftc, const float * sigmas, const int ccnt, const float * pmin, const float * pmax, const float * sigmas2) noexcept;
template void filter_avx512<6>(float * dftc, const float * sigmas, const int ccnt, const float * pmin, const float * pmax, const float * sigmas2) noexcept;

template void func_0_avx512<uint8_t>(VSFrameRef * src[3], VSFrameRef * dst, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept;
template void func_0_avx512<uint16_t>(VSFrameRef * src[3], VSFrameRef * dst, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept;
template void func_0_avx512<float>(VSFrameRef * src[3], VSFrameRef * dst, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept;

template void func_1_avx512<uint8_t>(VSFrameRef * src[15][3], VSFrameRef * dst, const int pos, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept;
template void func_1_avx512<uint16_t>(VSFrameRef * src[15][3], VSFrameRef * dst, const int pos, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept;
template void func_1_avx512<float>(VSFrameRef * src[15][3], VSFrameRef * dst, const int pos, const DFTTestData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept;
#endif

#include "SGP4Batch.h"
#include "SGP4.h"
#include "Globals.h"
#include <cmath>
#include <immintrin.h>
#include <omp.h>
#include <iostream>
#include <cstring>

namespace libsgp4
{

struct alignas(64) Vec8 {
    __m512d v;
    Vec8() : v(_mm512_setzero_pd()) {}
    Vec8(__m512d val) : v(val) {}
    Vec8(double s) : v(_mm512_set1_pd(s)) {}
    static Vec8 load(const double* p) { return _mm512_load_pd(p); }
    void store(double* p) const { _mm512_store_pd(p, v); }
    Vec8 operator+(Vec8 b) const { return _mm512_add_pd(v, b.v); }
    Vec8 operator-(Vec8 b) const { return _mm512_sub_pd(v, b.v); }
    Vec8 operator*(Vec8 b) const { return _mm512_mul_pd(v, b.v); }
    Vec8 fmadd(Vec8 b, Vec8 c) const { return _mm512_fmadd_pd(v, b.v, c.v); }
};

static inline Vec8 operator-(double a, Vec8 b) { return Vec8(a) - b; }
static inline Vec8 operator*(double a, Vec8 b) { return Vec8(a) * b; }
static inline Vec8 operator/(Vec8 a, Vec8 b) { return _mm512_div_pd(a.v, b.v); }

extern "C" {
    __m512d _ZGVeN8v_sin(__m512d x);
    __m512d _ZGVeN8v_cos(__m512d x);
    __m512d _ZGVeN8vv_atan2(__m512d y, __m512d x);
}

static inline void v_sincos_vector(Vec8 x, Vec8& s, Vec8& c) {
    s.v = _ZGVeN8v_sin(x.v);
    c.v = _ZGVeN8v_cos(x.v);
}

static inline Vec8 v_atan2_vector(Vec8 y, Vec8 x) {
    return Vec8(_ZGVeN8vv_atan2(y.v, x.v));
}

inline void v_sincos_minimax(Vec8 x, Vec8& s, Vec8& c) {
    const double INV_HALFPI = 0.63661977236758134308;
    __m512d n_float = _mm512_roundscale_pd(_mm512_mul_pd(x.v, _mm512_set1_pd(INV_HALFPI)), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
    __m256i n_256 = _mm512_cvtpd_epi32(n_float);
    Vec8 r = x - Vec8(_mm512_mul_pd(n_float, _mm512_set1_pd(1.57079632679489655800)));
    r = r - Vec8(_mm512_mul_pd(n_float, _mm512_set1_pd(6.12323399573676603587e-17)));
    Vec8 r2 = r * r;
    Vec8 sn = r + (r * r2) * Vec8(_mm512_fmadd_pd(r2.v, _mm512_fmadd_pd(r2.v, _mm512_fmadd_pd(r2.v, _mm512_set1_pd(2.75573137070700676789e-6), _mm512_set1_pd(-1.98412698298579493134e-4)), _mm512_set1_pd(8.3333333333322489461245e-3)), _mm512_set1_pd(-1.66666666666664650410e-1)));
    Vec8 cs = Vec8(1.0) + r2 * Vec8(_mm512_fmadd_pd(r2.v, _mm512_fmadd_pd(r2.v, _mm512_fmadd_pd(r2.v, _mm512_set1_pd(2.48015872894767294178e-5), _mm512_set1_pd(-1.38888888887695625410e-3)), _mm512_set1_pd(4.16666666666662260110e-2)), _mm512_set1_pd(-5.00000000000000000000e-1)));
    __m256i n_mod_4 = _mm256_and_si256(n_256, _mm256_set1_epi32(3));
    __mmask8 swap_mask = _mm256_test_epi32_mask(n_mod_4, _mm256_set1_epi32(1));
    __mmask8 neg_sin_mask = _mm256_cmp_epi32_mask(n_mod_4, _mm256_set1_epi32(1), _MM_CMPINT_GT);
    __mmask8 neg_cos_mask = _mm256_cmp_epi32_mask(n_mod_4, _mm256_setzero_si256(), _MM_CMPINT_GT) & _mm256_cmp_epi32_mask(n_mod_4, _mm256_set1_epi32(3), _MM_CMPINT_LT);
    Vec8 res_s = Vec8(_mm512_mask_blend_pd(swap_mask, sn.v, cs.v));
    Vec8 res_c = Vec8(_mm512_mask_blend_pd(swap_mask, cs.v, sn.v));
    s = Vec8(_mm512_mask_sub_pd(res_s.v, neg_sin_mask, _mm512_setzero_pd(), res_s.v));
    c = Vec8(_mm512_mask_sub_pd(res_c.v, neg_cos_mask, _mm512_setzero_pd(), res_c.v));
}

static inline Vec8 v_fmod_accurate(Vec8 a, double b) {
    alignas(64) double a_a[8], a_res[8];
    a.store(a_a);
    for(int i=0; i<8; i++) a_res[i] = fmod(a_a[i], b);
    return Vec8::load(a_res);
}

SGP4Batch::SGP4Batch(const std::vector<Tle>& tles) : total_satellites_(static_cast<int>(tles.size())) {
    padded_satellites_ = (total_satellites_ + 7) & ~7;
    cosio = make_aligned<double>(padded_satellites_); sinio = make_aligned<double>(padded_satellites_);
    eta = make_aligned<double>(padded_satellites_); t2cof = make_aligned<double>(padded_satellites_);
    x1mth2 = make_aligned<double>(padded_satellites_); x3thm1 = make_aligned<double>(padded_satellites_);
    x7thm1 = make_aligned<double>(padded_satellites_); aycof = make_aligned<double>(padded_satellites_);
    xlcof = make_aligned<double>(padded_satellites_); xnodcf = make_aligned<double>(padded_satellites_);
    c1 = make_aligned<double>(padded_satellites_); c4 = make_aligned<double>(padded_satellites_);
    omgdot = make_aligned<double>(padded_satellites_); xnodot = make_aligned<double>(padded_satellites_);
    xmdot = make_aligned<double>(padded_satellites_); c5 = make_aligned<double>(padded_satellites_);
    omgcof = make_aligned<double>(padded_satellites_); xmcof = make_aligned<double>(padded_satellites_);
    delmo = make_aligned<double>(padded_satellites_); sinmo = make_aligned<double>(padded_satellites_);
    d2 = make_aligned<double>(padded_satellites_); d3 = make_aligned<double>(padded_satellites_);
    d4 = make_aligned<double>(padded_satellites_); t3cof = make_aligned<double>(padded_satellites_);
    t4cof = make_aligned<double>(padded_satellites_); t5cof = make_aligned<double>(padded_satellites_);
    xmo = make_aligned<double>(padded_satellites_); nodeo = make_aligned<double>(padded_satellites_);
    omegao = make_aligned<double>(padded_satellites_); ecco = make_aligned<double>(padded_satellites_);
    inclo = make_aligned<double>(padded_satellites_); bstar = make_aligned<double>(padded_satellites_);
    aodp = make_aligned<double>(padded_satellites_); no_kozai = make_aligned<double>(padded_satellites_);
    use_simple_model = make_aligned<double>(padded_satellites_); use_deep_space = make_aligned<double>(padded_satellites_);
    active = make_aligned<double>(padded_satellites_); epochs.resize(padded_satellites_);
    last_tsince = make_aligned<double>(padded_satellites_); memo_e = make_aligned<double>(padded_satellites_);
    memo_a = make_aligned<double>(padded_satellites_); memo_omega = make_aligned<double>(padded_satellites_);
    memo_xl = make_aligned<double>(padded_satellites_); memo_xnode = make_aligned<double>(padded_satellites_);

    for (int i = 0; i < total_satellites_; ++i) {
        SGP4 model(tles[i]);
        cosio[i] = model.common_consts_.cosio; sinio[i] = model.common_consts_.sinio;
        eta[i] = model.common_consts_.eta; t2cof[i] = model.common_consts_.t2cof;
        x1mth2[i] = model.common_consts_.x1mth2; x3thm1[i] = model.common_consts_.x3thm1;
        x7thm1[i] = model.common_consts_.x7thm1; aycof[i] = model.common_consts_.aycof;
        xlcof[i] = model.common_consts_.xlcof; xnodcf[i] = model.common_consts_.xnodcf;
        c1[i] = model.common_consts_.c1; c4[i] = model.common_consts_.c4;
        omgdot[i] = model.common_consts_.omgdot; xnodot[i] = model.common_consts_.xnodot;
        xmdot[i] = model.common_consts_.xmdot; c5[i] = model.nearspace_consts_.c5;
        omgcof[i] = model.nearspace_consts_.omgcof; xmcof[i] = model.nearspace_consts_.xmcof;
        delmo[i] = model.nearspace_consts_.delmo; sinmo[i] = model.nearspace_consts_.sinmo;
        d2[i] = model.nearspace_consts_.d2; d3[i] = model.nearspace_consts_.d3;
        d4[i] = model.nearspace_consts_.d4; t3cof[i] = model.nearspace_consts_.t3cof;
        t4cof[i] = model.nearspace_consts_.t4cof; t5cof[i] = model.nearspace_consts_.t5cof;
        xmo[i] = model.elements_.MeanAnomoly(); nodeo[i] = model.elements_.AscendingNode();
        omegao[i] = model.elements_.ArgumentPerigee(); ecco[i] = model.elements_.Eccentricity();
        inclo[i] = model.elements_.Inclination(); bstar[i] = model.elements_.BStar();
        aodp[i] = model.elements_.RecoveredSemiMajorAxis(); no_kozai[i] = model.elements_.RecoveredMeanMotion();
        use_simple_model[i] = model.use_simple_model_ ? 1.0 : 0.0;
        use_deep_space[i] = model.use_deep_space_ ? 1.0 : 0.0; active[i] = 1.0;
        epochs[i] = model.elements_.Epoch(); last_tsince[i] = -1e9;
    }
    for (int i = total_satellites_; i < padded_satellites_; ++i) {
        active[i] = 0.0; last_tsince[i] = -1e9;
    }
}

void SGP4Batch::Propagate(double tsince, std::vector<Eci>& results, MathMode mode) const {
    results.resize(total_satellites_, Eci(DateTime(), Vector()));
    const double* p_xmdot = xmdot.get(); const double* p_xmo = xmo.get(); const double* p_omgdot = omgdot.get();
    const double* p_omegao = omegao.get(); const double* p_xnodot = xnodot.get(); const double* p_nodeo = nodeo.get();
    const double* p_xnodcf = xnodcf.get(); const double* p_c1 = c1.get(); const double* p_bstar = bstar.get();
    const double* p_c4 = c4.get(); const double* p_t2cof = t2cof.get(); const double* p_eta = eta.get();
    const double* p_xmcof = xmcof.get(); const double* p_delmo = delmo.get(); const double* p_omgcof = omgcof.get();
    const double* p_d2 = d2.get(); const double* p_d3 = d3.get(); const double* p_d4 = d4.get();
    const double* p_c5 = c5.get(); const double* p_sinmo = sinmo.get(); const double* p_t3cof = t3cof.get();
    const double* p_t5cof = t5cof.get(); const double* p_t4cof = t4cof.get(); const double* p_use_simple = use_simple_model.get();
    const double* p_aodp = aodp.get(); const double* p_ecco = ecco.get(); const double* p_no_kozai = no_kozai.get();
    const double* p_xlcof = xlcof.get(); const double* p_aycof = aycof.get(); const double* p_x3thm1 = x3thm1.get();
    const double* p_cosio = cosio.get(); const double* p_inclo = inclo.get(); const double* p_sinio = sinio.get();
    const double* p_x1mth2 = x1mth2.get(); const double* p_x7thm1 = x7thm1.get();
    const double* p_use_deep = use_deep_space.get(); const double* p_active = active.get();
    double* p_last_t = last_tsince.get(); double* p_memo_a = memo_a.get(); double* p_memo_e = memo_e.get();
    double* p_memo_om = memo_omega.get(); double* p_memo_xl = memo_xl.get(); double* p_memo_xnode = memo_xnode.get();

    Vec8 v_tsince(tsince); Vec8 v_tsq = v_tsince * v_tsince; Vec8 v_tcube = v_tsq * v_tsince; Vec8 v_tfour = v_tsince * v_tcube;
    Vec8 v_one(1.0); Vec8 v_zero(0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < padded_satellites_; i += 8) {
        Vec8 v_last_t = Vec8::load(&p_last_t[i]);
        __mmask8 m_memo = _mm512_cmp_pd_mask(v_tsince.v, v_last_t.v, _CMP_EQ_OQ);
        Vec8 a, e, omega, xl, xnode;
        if (m_memo == 0xFF) {
            a = Vec8::load(&p_memo_a[i]); e = Vec8::load(&p_memo_e[i]);
            omega = Vec8::load(&p_memo_om[i]); xl = Vec8::load(&p_memo_xl[i]); xnode = Vec8::load(&p_memo_xnode[i]);
        } else {
            Vec8 xmdf = Vec8::load(&p_xmdot[i]).fmadd(v_tsince, Vec8::load(&p_xmo[i]));
            Vec8 omgadf = Vec8::load(&p_omgdot[i]).fmadd(v_tsince, Vec8::load(&p_omegao[i]));
            Vec8 xnoddf = Vec8::load(&p_xnodot[i]).fmadd(v_tsince, Vec8::load(&p_nodeo[i]));
            xnode = xnoddf + Vec8::load(&p_xnodcf[i]) * v_tsq;
            Vec8 tempa = v_one - Vec8::load(&p_c1[i]) * v_tsince;
            Vec8 v_bstar = Vec8::load(&p_bstar[i]);
            Vec8 tempe = v_bstar * Vec8::load(&p_c4[i]) * v_tsince;
            Vec8 templ = Vec8::load(&p_t2cof[i]) * v_tsq;
            Vec8 xmp = xmdf; omega = omgadf;
            Vec8 sw_xmdf, cw_xmdf;
            if (mode == MathMode::Minimax) v_sincos_minimax(xmdf, sw_xmdf, cw_xmdf); else v_sincos_vector(xmdf, sw_xmdf, cw_xmdf);
            Vec8 etacos = v_one + Vec8::load(&p_eta[i]) * cw_xmdf;
            Vec8 delm = Vec8::load(&p_xmcof[i]) * (etacos * etacos * etacos - Vec8::load(&p_delmo[i]));
            Vec8 delomg = Vec8::load(&p_omgcof[i]) * v_tsince;
            Vec8 temp_ns = delomg + delm;
            Vec8 xmp_ns = xmdf + temp_ns; Vec8 omega_ns = omgadf - temp_ns;
            Vec8 tempa_ns = tempa - Vec8::load(&p_d2[i]) * v_tsq - Vec8::load(&p_d3[i]) * v_tcube - Vec8::load(&p_d4[i]) * v_tfour;
            Vec8 sw_xmp_ns, cw_xmp_ns;
            if (mode == MathMode::Minimax) v_sincos_minimax(xmp_ns, sw_xmp_ns, cw_xmp_ns); else v_sincos_vector(xmp_ns, sw_xmp_ns, cw_xmp_ns);
            Vec8 tempe_ns = tempe + v_bstar * Vec8::load(&p_c5[i]) * (sw_xmp_ns - Vec8::load(&p_sinmo[i]));
            Vec8 templ_ns = templ + Vec8::load(&p_t3cof[i]) * v_tcube + v_tfour * Vec8::load(&p_t5cof[i]).fmadd(v_tsince, Vec8::load(&p_t4cof[i]));
            __mmask8 m_ns = _mm512_cmp_pd_mask(Vec8::load(&p_use_simple[i]).v, v_zero.v, _CMP_EQ_OQ);
            xmp.v = _mm512_mask_blend_pd(m_ns, xmp.v, xmp_ns.v); omega.v = _mm512_mask_blend_pd(m_ns, omega.v, omega_ns.v);
            tempa.v = _mm512_mask_blend_pd(m_ns, tempa.v, tempa_ns.v); tempe.v = _mm512_mask_blend_pd(m_ns, tempe.v, tempe_ns.v);
            templ.v = _mm512_mask_blend_pd(m_ns, templ.v, templ_ns.v);
            a = Vec8::load(&p_aodp[i]) * tempa * tempa; e = Vec8::load(&p_ecco[i]) - tempe;
            e.v = _mm512_max_pd(e.v, _mm512_set1_pd(1e-6)); e.v = _mm512_min_pd(e.v, _mm512_set1_pd(1.0 - 1e-6));
            xl = xmp + omega + xnode + Vec8(p_no_kozai[i]) * templ;
            if (m_memo != 0) {
                a.v = _mm512_mask_blend_pd(m_memo, a.v, Vec8::load(&p_memo_a[i]).v); e.v = _mm512_mask_blend_pd(m_memo, e.v, Vec8::load(&p_memo_e[i]).v);
                omega.v = _mm512_mask_blend_pd(m_memo, omega.v, Vec8::load(&p_memo_om[i]).v); xl.v = _mm512_mask_blend_pd(m_memo, xl.v, Vec8::load(&p_memo_xl[i]).v);
                xnode.v = _mm512_mask_blend_pd(m_memo, xnode.v, Vec8::load(&p_memo_xnode[i]).v);
            }
            v_tsince.store(&p_last_t[i]); a.store(&p_memo_a[i]); e.store(&p_memo_e[i]); omega.store(&p_memo_om[i]); xl.store(&p_memo_xl[i]); xnode.store(&p_memo_xnode[i]);
        }
        Vec8 beta2 = v_one - e * e; Vec8 xn = Vec8(kXKE) / (a * _mm512_sqrt_pd(a.v));
        Vec8 sw_om, cw_om;
        if (mode == MathMode::Minimax) v_sincos_minimax(omega, sw_om, cw_om); else v_sincos_vector(omega, sw_om, cw_om);
        Vec8 axn = e * cw_om; Vec8 temp11 = v_one / (a * beta2);
        Vec8 xll = temp11 * Vec8::load(&p_xlcof[i]) * axn; Vec8 aynl = temp11 * Vec8::load(&p_aycof[i]);
        Vec8 xlt = xl + xll; Vec8 ayn = e * sw_om + aynl; Vec8 elsq = axn * axn + ayn * ayn;
        Vec8 capu = v_fmod_accurate(xlt - xnode, kTWOPI); Vec8 epw = capu; Vec8 sinepw, cosepw, ecose, esine;
        for(int j=0; j<3; j++) {
            if (mode == MathMode::Minimax) v_sincos_minimax(epw, sinepw, cosepw); else v_sincos_vector(epw, sinepw, cosepw);
            ecose = axn * cosepw + ayn * sinepw; esine = axn * sinepw - ayn * cosepw;
            Vec8 f = capu - epw + esine; Vec8 fdot = v_one - ecose;
            epw = epw + f / (fdot + Vec8(0.5) * esine * (f / fdot));
        }
        if (mode == MathMode::Minimax) v_sincos_minimax(epw, sinepw, cosepw); else v_sincos_vector(epw, sinepw, cosepw);
        ecose = axn * cosepw + ayn * sinepw; esine = axn * sinepw - ayn * cosepw;
        Vec8 temp21 = v_one - elsq; Vec8 pl = a * temp21; Vec8 r = a * (v_one - ecose); Vec8 temp31 = v_one / r;
        Vec8 rdot = Vec8(kXKE) * _mm512_sqrt_pd(a.v) * esine * temp31; Vec8 rfdot = Vec8(kXKE) * _mm512_sqrt_pd(pl.v) * temp31;
        Vec8 temp32 = a * temp31; Vec8 betal = _mm512_sqrt_pd(temp21.v); Vec8 v_temp33 = v_one / (v_one + betal);
        Vec8 cosu_un = temp32 * (cosepw - axn + ayn * esine * v_temp33); Vec8 sinu_un = temp32 * (sinepw - ayn - axn * esine * v_temp33);
        Vec8 u = v_atan2_vector(sinu_un, cosu_un); Vec8 sin2u = Vec8(2.0) * sinu_un * cosu_un; Vec8 cos2u = Vec8(2.0) * cosu_un * cosu_un - v_one;
        Vec8 t41 = v_one / pl; Vec8 t42 = Vec8(kCK2) * t41; Vec8 t43 = t42 * t41;
        Vec8 rk = r * (v_one - Vec8(1.5) * t43 * betal * Vec8::load(&p_x3thm1[i])) + Vec8(0.5) * t42 * Vec8::load(&p_x1mth2[i]) * cos2u;
        Vec8 uk = u - Vec8(0.25) * t43 * Vec8::load(&p_x7thm1[i]) * sin2u;
        Vec8 xnodek = xnode + Vec8(1.5) * t43 * Vec8::load(&p_cosio[i]) * sin2u;
        Vec8 xinck = Vec8::load(&p_inclo[i]) + Vec8(1.5) * t43 * Vec8::load(&p_cosio[i]) * Vec8::load(&p_sinio[i]) * cos2u;
        Vec8 rdotk = rdot - Vec8(p_no_kozai[i]) * t42 * Vec8::load(&p_x1mth2[i]) * sin2u;
        Vec8 rfdotk = rfdot + Vec8(p_no_kozai[i]) * t42 * (Vec8::load(&p_x1mth2[i]) * cos2u + Vec8(1.5) * Vec8::load(&p_x3thm1[i]));
        Vec8 snuk, csuk, snik, csik, snnk, csnk;
        if (mode == MathMode::Minimax) { v_sincos_minimax(uk, snuk, csuk); v_sincos_minimax(xinck, snik, csik); v_sincos_minimax(xnodek, snnk, csnk); }
        else { v_sincos_vector(uk, snuk, csuk); v_sincos_vector(xinck, snik, csik); v_sincos_vector(xnodek, snnk, csnk); }
        Vec8 xmx = (v_zero - snnk) * csik; Vec8 xmy = csnk * csik;
        Vec8 ux = xmx * snuk + csnk * csuk; Vec8 uy = xmy * snuk + snnk * csuk; Vec8 uz = snik * snuk;
        Vec8 vx = xmx * csuk - csnk * snuk; Vec8 vy = xmy * csuk - snnk * snuk; Vec8 vz = snik * csuk;
        alignas(64) double a_rk[8], a_ux[8], a_uy[8], a_uz[8], a_rdotk[8], a_rfdotk[8], a_vx[8], a_vy[8], a_vz[8];
        rk.store(a_rk); ux.store(a_ux); uy.store(a_uy); uz.store(a_uz); rdotk.store(a_rdotk); rfdotk.store(a_rfdotk); vx.store(a_vx); vy.store(a_vy); vz.store(a_vz);
        for (int s = 0; s < 8; s++) {
            int idx = i + s; if (idx >= total_satellites_) break;
            if (p_active[idx] == 0.0 || p_use_deep[idx] == 1.0) continue;
            results[idx] = Eci(epochs[idx].AddMinutes(tsince), Vector(a_rk[s] * a_ux[s] * kXKMPER, a_rk[s] * a_uy[s] * kXKMPER, a_rk[s] * a_uz[s] * kXKMPER),
                           Vector((a_rdotk[s] * a_ux[s] + a_rfdotk[s] * a_vx[s]) * kXKMPER / 60.0, (a_rdotk[s] * a_uy[s] + a_rfdotk[s] * a_vy[s]) * kXKMPER / 60.0, (a_rdotk[s] * a_uz[s] + a_rfdotk[s] * a_vz[s]) * kXKMPER / 60.0));
        }
    }
}

} // namespace libsgp4

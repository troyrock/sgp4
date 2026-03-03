#include "SGP4Batch.h"
#include "SGP4.h"
#include "Globals.h"
#include "Util.h"
#include <cmath>
#include <immintrin.h>
#include <omp.h>
#include <iostream>
#include <cstring>
#include <algorithm>

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
    Vec8 operator/(Vec8 b) const { return _mm512_div_pd(v, b.v); }
    Vec8 fmadd(Vec8 b, Vec8 c) const { return _mm512_fmadd_pd(v, b.v, c.v); }
};

static inline Vec8 operator-(double a, Vec8 b) { return Vec8(a) - b; }
static inline Vec8 operator*(double a, Vec8 b) { return Vec8(a) * b; }

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
    Vec8 sn = r + (r * r2) * Vec8(_mm512_fmadd_pd(r2.v, _mm512_fmadd_pd(r2.v, _mm512_fmadd_pd(r2.v, _mm512_set1_pd(2.75573137070700676789e-6), _mm512_set1_pd(-1.98412698298579493134e-4)), _mm512_set1_pd(8.3333333333322489461245e-3)), _mm512_set1_pd(-1.66666666666664650410-1)));
    Vec8 cs = Vec8(1.0) + r2 * Vec8(_mm512_fmadd_pd(r2.v, _mm512_fmadd_pd(r2.v, _mm512_fmadd_pd(r2.v, _mm512_set1_pd(2.48015872894767294178e-5), _mm512_set1_pd(-1.38888888887695625410e-3)), _mm512_set1_pd(4.16666666666662260110e-2)), _mm512_set1_pd(-5.00000000000000000000e-1)));
    __m256i n_mod_4 = _mm256_and_si256(n_256, _mm256_set1_epi32(3));
    __mmask8 swap_mask = _mm256_test_epi32_mask(n_mod_4, _mm256_set1_epi32(1));
    __mmask8 neg_sin_mask = _mm256_test_epi32_mask(n_mod_4, _mm256_set1_epi32(2));
    __mmask8 neg_cos_mask = _mm256_cmp_epi32_mask(n_mod_4, _mm256_set1_epi32(1), _MM_CMPINT_GE) & _mm256_cmp_epi32_mask(n_mod_4, _mm256_set1_epi32(3), _MM_CMPINT_LT);
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

static inline Vec8 v_wrap_twopi(Vec8 x) {
    Vec8 res = v_fmod_accurate(x, kTWOPI);
    __mmask8 m_neg = _mm512_cmp_pd_mask(res.v, _mm512_setzero_pd(), _CMP_LT_OQ);
    return Vec8(_mm512_mask_add_pd(res.v, m_neg, res.v, _mm512_set1_pd(kTWOPI)));
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
    ds_sse = make_aligned<double>(padded_satellites_); ds_ssi = make_aligned<double>(padded_satellites_); ds_ssl = make_aligned<double>(padded_satellites_); 
    ds_ssh = make_aligned<double>(padded_satellites_); ds_ssg = make_aligned<double>(padded_satellites_); ds_zmos = make_aligned<double>(padded_satellites_);
    ds_zmol = make_aligned<double>(padded_satellites_); ds_se2 = make_aligned<double>(padded_satellites_); ds_se3 = make_aligned<double>(padded_satellites_);
    ds_si2 = make_aligned<double>(padded_satellites_); ds_si3 = make_aligned<double>(padded_satellites_); ds_sl2 = make_aligned<double>(padded_satellites_);
    ds_sl3 = make_aligned<double>(padded_satellites_); ds_sl4 = make_aligned<double>(padded_satellites_); ds_sgh2 = make_aligned<double>(padded_satellites_);
    ds_sgh3 = make_aligned<double>(padded_satellites_); ds_sgh4 = make_aligned<double>(padded_satellites_); ds_sh2 = make_aligned<double>(padded_satellites_);
    ds_sh3 = make_aligned<double>(padded_satellites_); ds_ee2 = make_aligned<double>(padded_satellites_); ds_e3 = make_aligned<double>(padded_satellites_);
    ds_xi2 = make_aligned<double>(padded_satellites_); ds_xi3 = make_aligned<double>(padded_satellites_); ds_xl2 = make_aligned<double>(padded_satellites_);
    ds_xl3 = make_aligned<double>(padded_satellites_); ds_xl4 = make_aligned<double>(padded_satellites_); ds_xgh2 = make_aligned<double>(padded_satellites_);
    ds_xgh3 = make_aligned<double>(padded_satellites_); ds_xgh4 = make_aligned<double>(padded_satellites_); ds_xh2 = make_aligned<double>(padded_satellites_);
    ds_xh3 = make_aligned<double>(padded_satellites_); ds_d2201 = make_aligned<double>(padded_satellites_); ds_d2211 = make_aligned<double>(padded_satellites_);
    ds_d3210 = make_aligned<double>(padded_satellites_); ds_d3222 = make_aligned<double>(padded_satellites_); ds_d4410 = make_aligned<double>(padded_satellites_);
    ds_d4422 = make_aligned<double>(padded_satellites_); ds_d5220 = make_aligned<double>(padded_satellites_); ds_d5232 = make_aligned<double>(padded_satellites_);
    ds_d5421 = make_aligned<double>(padded_satellites_); ds_d5433 = make_aligned<double>(padded_satellites_); ds_del1 = make_aligned<double>(padded_satellites_);
    ds_del2 = make_aligned<double>(padded_satellites_); ds_del3 = make_aligned<double>(padded_satellites_); ds_xfact = make_aligned<double>(padded_satellites_);
    ds_xlamo = make_aligned<double>(padded_satellites_); ds_shape = make_aligned<double>(padded_satellites_); ds_gsto = make_aligned<double>(padded_satellites_);
    int_xli = make_aligned<double>(padded_satellites_); int_xni = make_aligned<double>(padded_satellites_); int_atime = make_aligned<double>(padded_satellites_);

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
        if (model.use_deep_space_) {
            const auto& ds = model.deepspace_consts_;
            ds_sse[i] = ds.sse; ds_ssi[i] = ds.ssi; ds_ssl[i] = ds.ssl; ds_ssh[i] = ds.ssh; ds_ssg[i] = ds.ssg;
            ds_zmos[i] = ds.zmos; ds_zmol[i] = ds.zmol; ds_se2[i] = ds.se2; ds_se3[i] = ds.se3;
            ds_si2[i] = ds.si2; ds_si3[i] = ds.si3; ds_sl2[i] = ds.sl2; ds_sl3[i] = ds.sl3; ds_sl4[i] = ds.sl4;
            ds_sgh2[i] = ds.sgh2; ds_sgh3[i] = ds.sgh3; ds_sgh4[i] = ds.sgh4; ds_sh2[i] = ds.sh2; ds_sh3[i] = ds.sh3;
            ds_ee2[i] = ds.ee2; ds_e3[i] = ds.e3; ds_xi2[i] = ds.xi2; ds_xi3[i] = ds.xi3; ds_xl2[i] = ds.xl2;
            ds_xl3[i] = ds.xl3; ds_xl4[i] = ds.xl4; ds_xgh2[i] = ds.xgh2; ds_xgh3[i] = ds.xgh3; ds_xgh4[i] = ds.xgh4;
            ds_xh2[i] = ds.xh2; ds_xh3[i] = ds.xh3; ds_d2201[i] = ds.d2201; ds_d2211[i] = ds.d2211; ds_d3210[i] = ds.d3210;
            ds_d3222[i] = ds.d3222; ds_d4410[i] = ds.d4410; ds_d4422[i] = ds.d4422; ds_d5220[i] = ds.d5220;
            ds_d5232[i] = ds.d5232; ds_d5421[i] = ds.d5421; ds_d5433[i] = ds.d5433; ds_del1[i] = ds.del1;
            ds_del2[i] = ds.del2; ds_del3[i] = ds.del3; ds_xfact[i] = ds.xfact; ds_xlamo[i] = ds.xlamo;
            ds_shape[i] = static_cast<double>(ds.shape); ds_gsto[i] = ds.gsto;
            int_xli[i] = model.integrator_params_.xli; int_xni[i] = model.integrator_params_.xni; int_atime[i] = model.integrator_params_.atime;
        }
    }
    for (int i = total_satellites_; i < padded_satellites_; ++i) {
        active[i] = 0.0; last_tsince[i] = -1e9;
    }
}

void SGP4Batch::Propagate(double tsince, std::vector<Eci>& results, MathMode mode) const {
    results.resize(total_satellites_, Eci(DateTime(), Vector()));
    
    // Near Space SoA
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

    // Deep Space SoA
    const double* p_ds_ssl = ds_ssl.get(); const double* p_ds_ssg = ds_ssg.get(); const double* p_ds_ssh = ds_ssh.get();
    const double* p_ds_sse = ds_sse.get(); const double* p_ds_ssi = ds_ssi.get(); const double* p_ds_zmos = ds_zmos.get();
    const double* p_ds_zmol = ds_zmol.get(); const double* p_ds_se2 = ds_se2.get(); const double* p_ds_se3 = ds_se3.get();
    const double* p_ds_si2 = ds_si2.get(); const double* p_ds_si3 = ds_si3.get(); const double* p_ds_sl2 = ds_sl2.get();
    const double* p_ds_sl3 = ds_sl3.get(); const double* p_ds_sl4 = ds_sl4.get(); const double* p_ds_sgh2 = ds_sgh2.get();
    const double* p_ds_sgh3 = ds_sgh3.get(); const double* p_ds_sgh4 = ds_sgh4.get(); const double* p_ds_sh2 = ds_sh2.get();
    const double* p_ds_sh3 = ds_sh3.get(); const double* p_ds_ee2 = ds_ee2.get(); const double* p_ds_e3 = ds_e3.get();
    const double* p_ds_xi2 = ds_xi2.get(); const double* p_ds_xi3 = ds_xi3.get(); const double* p_ds_xl2 = ds_xl2.get();
    const double* p_ds_xl3 = ds_xl3.get(); const double* p_ds_xl4 = ds_xl4.get(); const double* p_ds_xgh2 = ds_xgh2.get();
    const double* p_ds_xgh3 = ds_xgh3.get(); const double* p_ds_xgh4 = ds_xgh4.get(); const double* p_ds_xh2 = ds_xh2.get();
    const double* p_ds_xh3 = ds_xh3.get(); const double* p_ds_gsto = ds_gsto.get();
    const double* p_ds_del1 = ds_del1.get(); const double* p_ds_del2 = ds_del2.get(); const double* p_ds_del3 = ds_del3.get();
    const double* p_ds_d2201 = ds_d2201.get(); const double* p_ds_d2211 = ds_d2211.get(); const double* p_ds_d3210 = ds_d3210.get();
    const double* p_ds_d3222 = ds_d3222.get(); const double* p_ds_d4410 = ds_d4410.get(); const double* p_ds_d4422 = ds_d4422.get();
    const double* p_ds_d5220 = ds_d5220.get(); const double* p_ds_d5232 = ds_d5232.get(); const double* p_ds_d5421 = ds_d5421.get();
    const double* p_ds_d5433 = ds_d5433.get(); const double* p_ds_xfact = ds_xfact.get(); const double* p_ds_xlamo = ds_xlamo.get();
    const double* p_ds_shape = ds_shape.get();
    double* p_int_xli = int_xli.get(); double* p_int_xni = int_xni.get(); double* p_int_atime = int_atime.get();

    double* p_last_t = last_tsince.get(); double* p_memo_a = memo_a.get(); double* p_memo_e = memo_e.get();
    double* p_memo_om = memo_omega.get(); double* p_memo_xl = memo_xl.get(); double* p_memo_xnode = memo_xnode.get();

    Vec8 v_tsince(tsince); Vec8 v_tsq = v_tsince * v_tsince; Vec8 v_tcube = v_tsq * v_tsince; Vec8 v_tfour = v_tsince * v_tcube;
    Vec8 v_one(1.0); Vec8 v_zero(0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < padded_satellites_; i += 8) {
        Vec8 v_last_t = Vec8::load(&p_last_t[i]);
        __mmask8 m_memo = _mm512_cmp_pd_mask(v_tsince.v, v_last_t.v, _CMP_EQ_OQ);
        Vec8 a, e, omega, xl, xnode, xn, xinc;
        
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
            
            xn = Vec8::load(&p_no_kozai[i]);
            e = Vec8::load(&p_ecco[i]);
            xinc = Vec8::load(&p_inclo[i]);

            __mmask8 m_ds = _mm512_cmp_pd_mask(Vec8::load(&p_use_deep[i]).v, v_one.v, _CMP_EQ_OQ);
            
            if (m_ds) {
                // --- DEEP SPACE SECULAR ---
                xmdf.v = _mm512_mask_add_pd(xmdf.v, m_ds, xmdf.v, (Vec8::load(&p_ds_ssl[i]) * v_tsince).v);
                omgadf.v = _mm512_mask_add_pd(omgadf.v, m_ds, omgadf.v, (Vec8::load(&p_ds_ssg[i]) * v_tsince).v);
                xnoddf.v = _mm512_mask_add_pd(xnoddf.v, m_ds, xnoddf.v, (Vec8::load(&p_ds_ssh[i]) * v_tsince).v);
                e.v = _mm512_mask_add_pd(e.v, m_ds, e.v, (Vec8::load(&p_ds_sse[i]) * v_tsince).v);
                xinc.v = _mm512_mask_add_pd(xinc.v, m_ds, xinc.v, (Vec8::load(&p_ds_ssi[i]) * v_tsince).v);

                Vec8 v_shape = Vec8::load(&p_ds_shape[i]);
                __mmask8 m_integ = _mm512_mask_cmp_pd_mask(m_ds, v_shape.v, v_zero.v, _CMP_GT_OQ);
                if (m_integ) {
                    Vec8 atime = Vec8::load(&p_int_atime[i]);
                    Vec8 xni = Vec8::load(&p_int_xni[i]);
                    Vec8 xli = Vec8::load(&p_int_xli[i]);
                    Vec8 step(720.0); Vec8 step2(259200.0);
                    
                    __mmask8 m_reset = _mm512_cmp_pd_mask(_mm512_abs_pd(v_tsince.v), step.v, _CMP_LT_OQ);
                    m_reset |= _mm512_cmp_pd_mask(_mm512_mul_pd(v_tsince.v, atime.v), v_zero.v, _CMP_LE_OQ);
                    m_reset |= _mm512_cmp_pd_mask(_mm512_abs_pd(v_tsince.v), _mm512_abs_pd(atime.v), _CMP_LT_OQ);
                    m_reset &= m_integ;
                    
                    atime.v = _mm512_mask_blend_pd(m_reset, atime.v, v_zero.v);
                    xni.v = _mm512_mask_blend_pd(m_reset, xni.v, Vec8::load(&p_no_kozai[i]).v);
                    xli.v = _mm512_mask_blend_pd(m_reset, xli.v, Vec8::load(&p_ds_xlamo[i]).v);

                    __mmask8 m_running = m_integ;
                    while (m_running) {
                        Vec8 xndot, xnddt;
                        __mmask8 m_synch = _mm512_mask_cmp_pd_mask(m_running, v_shape.v, _mm512_set1_pd(1.0), _CMP_EQ_OQ);
                        if (m_synch) {
                            Vec8 s1, c1, s2, c2, s3, c3;
                            v_sincos_vector(xli - Vec8(0.13130908), s1, c1);
                            v_sincos_vector(2.0 * (xli - Vec8(2.8843198)), s2, c2);
                            v_sincos_vector(3.0 * (xli - Vec8(0.37448087)), s3, c3);
                            xndot.v = _mm512_mask_add_pd(xndot.v, m_synch, (Vec8::load(&p_ds_del1[i])*s1).v, (Vec8::load(&p_ds_del2[i])*s2 + Vec8::load(&p_ds_del3[i])*s3).v);
                            xnddt.v = _mm512_mask_add_pd(xnddt.v, m_synch, (Vec8::load(&p_ds_del1[i])*c1).v, (2.0*Vec8::load(&p_ds_del2[i])*c2 + 3.0*Vec8::load(&p_ds_del3[i])*c3).v);
                        }
                        __mmask8 m_reso = _mm512_mask_cmp_pd_mask(m_running, v_shape.v, _mm512_set1_pd(2.0), _CMP_EQ_OQ);
                        if (m_reso) {
                            Vec8 xomi = Vec8::load(&p_omegao[i]) + Vec8::load(&p_omgdot[i]) * atime;
                            Vec8 s1, c1, s2, c2, s3, c3, s4, c4, s5, c5;
                            v_sincos_vector(2.0*xomi + xli - Vec8(5.7686396), s1, c1);
                            v_sincos_vector(xli - Vec8(5.7686396), s2, c2);
                            v_sincos_vector(xomi + xli - Vec8(0.95240898), s3, c3);
                            v_sincos_vector(-1.0*xomi + xli - Vec8(0.95240898), s4, c4);
                            v_sincos_vector(2.0*xomi + 2.0*xli - Vec8(1.8014998), s5, c5);
                            xndot.v = _mm512_mask_add_pd(xndot.v, m_reso, xndot.v, (Vec8::load(&p_ds_d2201[i])*s1 + Vec8::load(&p_ds_d2211[i])*s2 + Vec8::load(&p_ds_d3210[i])*s3 + Vec8::load(&p_ds_d3222[i])*s4 + Vec8::load(&p_ds_d4410[i])*s5).v);
                            xnddt.v = _mm512_mask_add_pd(xnddt.v, m_reso, xnddt.v, (Vec8::load(&p_ds_d2201[i])*c1 + Vec8::load(&p_ds_d2211[i])*c2 + Vec8::load(&p_ds_d3210[i])*c3 + Vec8::load(&p_ds_d3222[i])*c4 + 2.0*Vec8::load(&p_ds_d4410[i])*c5).v);
                            v_sincos_vector(2.0*xli - Vec8(1.8014998), s1, c1);
                            v_sincos_vector(xomi + xli - Vec8(1.0508330), s2, c2);
                            v_sincos_vector(-1.0*xomi + xli - Vec8(1.0508330), s3, c3);
                            v_sincos_vector(xomi + 2.0*xli - Vec8(4.4108898), s4, c4);
                            v_sincos_vector(-1.0*xomi + 2.0*xli - Vec8(4.4108898), s5, c5);
                            xndot.v = _mm512_mask_add_pd(xndot.v, m_reso, xndot.v, (Vec8::load(&p_ds_d4422[i])*s1 + Vec8::load(&p_ds_d5220[i])*s2 + Vec8::load(&p_ds_d5232[i])*s3 + Vec8::load(&p_ds_d5421[i])*s4 + Vec8::load(&p_ds_d5433[i])*s5).v);
                            xnddt.v = _mm512_mask_add_pd(xnddt.v, m_reso, xnddt.v, (2.0*Vec8::load(&p_ds_d4422[i])*c1 + Vec8::load(&p_ds_d5220[i])*c2 + Vec8::load(&p_ds_d5232[i])*c3 + 2.0*(Vec8::load(&p_ds_d5421[i])*c4 + Vec8::load(&p_ds_d5433[i])*c5)).v);
                        }
                        Vec8 xldot = xni + Vec8::load(&p_ds_xfact[i]);
                        xnddt = xnddt * xldot;
                        Vec8 ft = v_tsince - atime;
                        __mmask8 m_step = _mm512_mask_cmp_pd_mask(m_running, _mm512_abs_pd(ft.v), step.v, _CMP_GE_OQ);
                        if (m_step) {
                            Vec8 delt = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(ft.v, v_zero.v, _CMP_GE_OQ), (0.0 - step).v, step.v);
                            xli.v = _mm512_mask_add_pd(xli.v, m_step, xli.v, (xldot * delt + xndot * step2).v);
                            xni.v = _mm512_mask_add_pd(xni.v, m_step, xni.v, (xndot * delt + xnddt * step2).v);
                            atime.v = _mm512_mask_add_pd(atime.v, m_step, atime.v, delt.v);
                        }
                        __mmask8 m_final = m_running & ~m_step;
                        if (m_final) {
                            xn.v = _mm512_mask_add_pd(xn.v, m_final, xni.v, (xndot * ft + 0.5 * xnddt * ft * ft).v);
                            Vec8 xl_temp = xli + xldot * ft + 0.5 * xndot * ft * ft;
                            Vec8 theta = v_wrap_twopi(Vec8::load(&p_ds_gsto[i]) + v_tsince * Vec8(0.00437526908801129966));
                            xmdf.v = _mm512_mask_blend_pd(m_synch & m_final, xmdf.v, (xl_temp + theta - xnode - omgadf).v);
                            xmdf.v = _mm512_mask_blend_pd(m_reso & m_final, xmdf.v, (xl_temp + 2.0 * (theta - xnode)).v);
                        }
                        m_running &= m_step;
                    }
                    atime.store(&p_int_atime[i]); xni.store(&p_int_xni[i]); xli.store(&p_int_xli[i]);
                }
            }

            Vec8 xmp = xmdf; omega = omgadf;
            Vec8 sw_xmdf, cw_xmdf;
            if (mode == MathMode::Minimax) v_sincos_minimax(xmdf, sw_xmdf, cw_xmdf); else v_sincos_vector(xmdf, sw_xmdf, cw_xmdf);
            
            __mmask8 m_ns = _mm512_cmp_pd_mask(Vec8::load(&p_use_simple[i]).v, v_zero.v, _CMP_EQ_OQ);
            m_ns &= ~m_ds; 

            if (m_ns) {
                Vec8 etacos = v_one + Vec8::load(&p_eta[i]) * cw_xmdf;
                Vec8 delm = Vec8::load(&p_xmcof[i]) * (etacos * etacos * etacos - Vec8::load(&p_delmo[i]));
                Vec8 delomg = Vec8::load(&p_omgcof[i]) * v_tsince;
                Vec8 temp_ns = delomg + delm;
                xmp.v = _mm512_mask_add_pd(xmp.v, m_ns, xmp.v, temp_ns.v);
                omega.v = _mm512_mask_sub_pd(omega.v, m_ns, omega.v, temp_ns.v);
                tempa.v = _mm512_mask_sub_pd(tempa.v, m_ns, tempa.v, (Vec8::load(&p_d2[i])*v_tsq + Vec8::load(&p_d3[i])*v_tcube + Vec8::load(&p_d4[i])*v_tfour).v);
                Vec8 sw_xmp_ns, cw_xmp_ns;
                if (mode == MathMode::Minimax) v_sincos_minimax(xmp, sw_xmp_ns, cw_xmp_ns); else v_sincos_vector(xmp, sw_xmp_ns, cw_xmp_ns);
                tempe.v = _mm512_mask_add_pd(tempe.v, m_ns, tempe.v, (v_bstar * Vec8::load(&p_c5[i]) * (sw_xmp_ns - Vec8::load(&p_sinmo[i]))).v);
                templ.v = _mm512_mask_add_pd(templ.v, m_ns, templ.v, (Vec8::load(&p_t3cof[i])*v_tcube + v_tfour * Vec8::load(&p_t5cof[i]).fmadd(v_tsince, Vec8::load(&p_t4cof[i]))).v);
            }

            a = Vec8::load(&p_aodp[i]) * tempa * tempa;
            e = e - tempe;
            e.v = _mm512_max_pd(e.v, _mm512_set1_pd(1e-6)); e.v = _mm512_min_pd(e.v, _mm512_set1_pd(1.0 - 1e-6));

            if (m_ds) {
                // --- DEEP SPACE PERIODICS ---
                Vec8 zm = Vec8::load(&p_ds_zmos[i]) + Vec8(1.19459E-5) * v_tsince;
                Vec8 sw_zm, cw_zm; v_sincos_vector(zm, sw_zm, cw_zm);
                Vec8 zf = zm + Vec8(2.0 * 0.01675) * sw_zm;
                Vec8 sw_zf, cw_zf; v_sincos_vector(zf, sw_zf, cw_zf);
                Vec8 f2 = Vec8(0.5) * sw_zf * sw_zf - Vec8(0.25);
                Vec8 f3 = Vec8(-0.5) * sw_zf * cw_zf;
                Vec8 pe = Vec8::load(&p_ds_se2[i]) * f2 + Vec8::load(&p_ds_se3[i]) * f3;
                Vec8 pinc = Vec8::load(&p_ds_si2[i]) * f2 + Vec8::load(&p_ds_si3[i]) * f3;
                Vec8 zm_l = Vec8::load(&p_ds_zmol[i]) + Vec8(1.5835218E-4) * v_tsince;
                Vec8 sw_zml, cw_zml; v_sincos_vector(zm_l, sw_zml, cw_zml);
                Vec8 zf_l = zm_l + Vec8(2.0 * 0.05490) * sw_zml;
                Vec8 sw_zfl, cw_zfl; v_sincos_vector(zf_l, sw_zfl, cw_zfl);
                Vec8 f2l = Vec8(0.5) * sw_zfl * sw_zfl - Vec8(0.25);
                Vec8 f3l = Vec8(-0.5) * sw_zfl * cw_zfl;
                pe = pe + Vec8::load(&p_ds_ee2[i]) * f2l + Vec8::load(&p_ds_e3[i]) * f3l;
                pinc = pinc + Vec8::load(&p_ds_xi2[i]) * f2l + Vec8::load(&p_ds_xi3[i]) * f3l;
                e.v = _mm512_mask_add_pd(e.v, m_ds, e.v, pe.v);
                xinc.v = _mm512_mask_add_pd(xinc.v, m_ds, xinc.v, pinc.v);
            }

            xl = xmp + omega + xnode + xn * templ;
            if (m_memo != 0) {
                a.v = _mm512_mask_blend_pd(m_memo, a.v, Vec8::load(&p_memo_a[i]).v); e.v = _mm512_mask_blend_pd(m_memo, e.v, Vec8::load(&p_memo_e[i]).v);
                omega.v = _mm512_mask_blend_pd(m_memo, omega.v, Vec8::load(&p_memo_om[i]).v); xl.v = _mm512_mask_blend_pd(m_memo, xl.v, Vec8::load(&p_memo_xl[i]).v);
                xnode.v = _mm512_mask_blend_pd(m_memo, xnode.v, Vec8::load(&p_memo_xnode[i]).v);
            }
            v_tsince.store(&p_last_t[i]); a.store(&p_memo_a[i]); e.store(&p_memo_e[i]); omega.store(&p_memo_om[i]); xl.store(&p_memo_xl[i]); xnode.store(&p_memo_xnode[i]);
        }
        
        // --- FINAL TRANSFORM ---
        Vec8 beta2 = v_one - e * e; Vec8 sw_om, cw_om;
        if (mode == MathMode::Minimax) v_sincos_minimax(omega, sw_om, cw_om); else v_sincos_vector(omega, sw_om, cw_om);
        Vec8 axn = e * cw_om; Vec8 temp11 = v_one / (a * beta2);
        Vec8 xlt = xl + temp11 * Vec8::load(&p_xlcof[i]) * axn;
        Vec8 ayn = e * sw_om + temp11 * Vec8::load(&p_aycof[i]);
        Vec8 capu = v_wrap_twopi(xlt - xnode); Vec8 epw = capu; Vec8 sinepw, cosepw, ecose, esine;
        for(int j=0; j<3; j++) {
            if (mode == MathMode::Minimax) v_sincos_minimax(epw, sinepw, cosepw); else v_sincos_vector(epw, sinepw, cosepw);
            ecose = axn * cosepw + ayn * sinepw; esine = axn * sinepw - ayn * cosepw;
            epw = epw + (capu - epw + esine) / (v_one - ecose + Vec8(0.5) * esine * ((capu - epw + esine) / (v_one - ecose)));
        }
        if (mode == MathMode::Minimax) v_sincos_minimax(epw, sinepw, cosepw); else v_sincos_vector(epw, sinepw, cosepw);
        ecose = axn * cosepw + ayn * sinepw; esine = axn * sinepw - ayn * cosepw;
        Vec8 r = a * (v_one - ecose); Vec8 temp31 = v_one / r;
        Vec8 rdot = Vec8(kXKE) * _mm512_sqrt_pd(a.v) * esine * temp31;
        Vec8 rfdot = Vec8(kXKE) * _mm512_sqrt_pd((a * (v_one - axn*axn - ayn*ayn)).v) * temp31;
        Vec8 cosu_un = a * temp31 * (cosepw - axn + ayn * esine * (v_one / (v_one + _mm512_sqrt_pd((v_one - axn*axn - ayn*ayn).v))));
        Vec8 sinu_un = a * temp31 * (sinepw - ayn - axn * esine * (v_one / (v_one + _mm512_sqrt_pd((v_one - axn*axn - ayn*ayn).v))));
        Vec8 u = v_atan2_vector(sinu_un, cosu_un);
        Vec8 sin2u = Vec8(2.0) * sinu_un * cosu_un; Vec8 cos2u = Vec8(2.0) * cosu_un * cosu_un - v_one;
        Vec8 t41 = v_one / (a * (v_one - axn*axn - ayn*ayn)); Vec8 t42 = Vec8(kCK2) * t41; Vec8 t43 = t42 * t41;
        Vec8 rk = r * (v_one - Vec8(1.5) * t43 * _mm512_sqrt_pd((v_one - axn*axn - ayn*ayn).v) * Vec8::load(&p_x3thm1[i])) + Vec8(0.5) * t42 * Vec8::load(&p_x1mth2[i]) * cos2u;
        Vec8 uk = u - Vec8(0.25) * t43 * Vec8::load(&p_x7thm1[i]) * sin2u;
        Vec8 xnodek = xnode + Vec8(1.5) * t43 * Vec8::load(&p_cosio[i]) * sin2u;
        Vec8 xinck = Vec8::load(&p_inclo[i]) + Vec8(1.5) * t43 * Vec8::load(&p_cosio[i]) * Vec8::load(&p_sinio[i]) * cos2u;
        Vec8 rdotk = rdot - Vec8(kXKE) / (a * _mm512_sqrt_pd(a.v)) * t42 * Vec8::load(&p_x1mth2[i]) * sin2u;
        Vec8 rfdotk = rfdot + Vec8(kXKE) / (a * _mm512_sqrt_pd(a.v)) * t42 * (Vec8::load(&p_x1mth2[i]) * cos2u + Vec8(1.5) * Vec8::load(&p_x3thm1[i]));
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
            if (p_active[idx] == 0.0) continue;
            results[idx] = Eci(epochs[idx].AddMinutes(tsince), Vector(a_rk[s] * a_ux[s] * kXKMPER, a_rk[s] * a_uy[s] * kXKMPER, a_rk[s] * a_uz[s] * kXKMPER),
                           Vector((a_rdotk[s] * a_ux[s] + a_rfdotk[s] * a_vx[s]) * kXKMPER / 60.0, (a_rdotk[s] * a_uy[s] + a_rfdotk[s] * a_vy[s]) * kXKMPER / 60.0, (a_rdotk[s] * a_uz[s] + a_rfdotk[s] * a_vz[s]) * kXKMPER / 60.0));
        }
    }
}

} // namespace libsgp4

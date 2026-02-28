#include "SGP4Batch.h"
#include "SGP4.h"
#include "Globals.h"
#include <cmath>
#include <immintrin.h>
#include <omp.h>

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
    Vec8 fnmadd(Vec8 b, Vec8 c) const { return _mm512_fnmadd_pd(v, b.v, c.v); }
};

// Accuracy-First Sincos (Serial calls but Parallel batches)
static inline void v_sincos_accurate(Vec8 x, Vec8& s, Vec8& c) {
    alignas(64) double a_x[8], a_s[8], a_c[8];
    x.store(a_x);
    for(int i=0; i<8; i++) __builtin_sincos(a_x[i], &a_s[i], &a_c[i]);
    s = Vec8::load(a_s);
    c = Vec8::load(a_c);
}

static inline Vec8 v_atan2_accurate(Vec8 y, Vec8 x) {
    alignas(64) double a_y[8], a_x[8], a_res[8];
    y.store(a_y); x.store(a_x);
    for(int i=0; i<8; i++) a_res[i] = std::atan2(a_y[i], a_x[i]);
    return Vec8::load(a_res);
}

static inline Vec8 v_fmod_accurate(Vec8 a, double b) {
    alignas(64) double a_a[8], a_res[8];
    a.store(a_a);
    for(int i=0; i<8; i++) {
        a_res[i] = fmod(a_a[i], b);
        if (a_res[i] < 0) a_res[i] += b;
    }
    return Vec8::load(a_res);
}

SGP4Batch::SGP4Batch(const std::vector<Tle>& tles)
    : total_satellites_(static_cast<int>(tles.size()))
{
    int num_batches = (total_satellites_ + BATCH_SIZE - 1) / BATCH_SIZE;
    batches_.resize(num_batches);
    for (int i = 0; i < total_satellites_; ++i) {
        int b = i / BATCH_SIZE;
        int s = i % BATCH_SIZE;
        SGP4 model(tles[i]);
        batches_[b].cosio[s] = model.common_consts_.cosio;
        batches_[b].sinio[s] = model.common_consts_.sinio;
        batches_[b].eta[s] = model.common_consts_.eta;
        batches_[b].t2cof[s] = model.common_consts_.t2cof;
        batches_[b].x1mth2[s] = model.common_consts_.x1mth2;
        batches_[b].x3thm1[s] = model.common_consts_.x3thm1;
        batches_[b].x7thm1[s] = model.common_consts_.x7thm1;
        batches_[b].aycof[s] = model.common_consts_.aycof;
        batches_[b].xlcof[s] = model.common_consts_.xlcof;
        batches_[b].xnodcf[s] = model.common_consts_.xnodcf;
        batches_[b].c1[s] = model.common_consts_.c1;
        batches_[b].c4[s] = model.common_consts_.c4;
        batches_[b].omgdot[s] = model.common_consts_.omgdot;
        batches_[b].xnodot[s] = model.common_consts_.xnodot;
        batches_[b].xmdot[s] = model.common_consts_.xmdot;
        batches_[b].c5[s] = model.nearspace_consts_.c5;
        batches_[b].omgcof[s] = model.nearspace_consts_.omgcof;
        batches_[b].xmcof[s] = model.nearspace_consts_.xmcof;
        batches_[b].delmo[s] = model.nearspace_consts_.delmo;
        batches_[b].sinmo[s] = model.nearspace_consts_.sinmo;
        batches_[b].d2[s] = model.nearspace_consts_.d2;
        batches_[b].d3[s] = model.nearspace_consts_.d3;
        batches_[b].d4[s] = model.nearspace_consts_.d4;
        batches_[b].t3cof[s] = model.nearspace_consts_.t3cof;
        batches_[b].t4cof[s] = model.nearspace_consts_.t4cof;
        batches_[b].t5cof[s] = model.nearspace_consts_.t5cof;
        batches_[b].xmo[s] = model.elements_.MeanAnomoly();
        batches_[b].nodeo[s] = model.elements_.AscendingNode();
        batches_[b].omegao[s] = model.elements_.ArgumentPerigee();
        batches_[b].ecco[s] = model.elements_.Eccentricity();
        batches_[b].inclo[s] = model.elements_.Inclination();
        batches_[b].bstar[s] = model.elements_.BStar();
        batches_[b].aodp[s] = model.elements_.RecoveredSemiMajorAxis();
        batches_[b].no_kozai[s] = model.elements_.RecoveredMeanMotion();
        batches_[b].use_simple_model[s] = model.use_simple_model_;
        batches_[b].use_deep_space[s] = model.use_deep_space_;
        batches_[b].active[s] = true;
        batches_[b].epoch[s] = model.elements_.Epoch();
    }
    for (int i = total_satellites_; i < (int)batches_.size() * BATCH_SIZE; ++i) {
        batches_[i/BATCH_SIZE].active[i%BATCH_SIZE] = false;
    }
}

void SGP4Batch::Propagate(double tsince, std::vector<Eci>& results) const
{
    results.resize(total_satellites_, Eci(DateTime(), Vector()));
    PropagatePool(tsince, results);
}

void SGP4Batch::PropagatePool(double tsince, std::vector<Eci>& pool) const
{
    // Assumes pool is already correctly sized to total_satellites_
    Vec8 v_tsince(tsince);
    Vec8 v_tsq = v_tsince * v_tsince;
    Vec8 v_tcube = v_tsq * v_tsince;
    Vec8 v_tfour = v_tsq * v_tsq;
    Vec8 v_one(1.0);
    Vec8 v_zero(0.0);
    Vec8 v_kXKMPER(kXKMPER);
    Vec8 v_kXKE(kXKE);
    Vec8 v_kCK2(kCK2);

    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batches_.size(); ++b) {
        const auto& c = batches_[b];
        
        // --- SECULAR UPDATES ---
        Vec8 xmdf = Vec8::load(c.xmdot).fmadd(v_tsince, Vec8::load(c.xmo));
        Vec8 omgadf = Vec8::load(c.omgdot).fmadd(v_tsince, Vec8::load(c.omegao));
        Vec8 xnoddf = Vec8::load(c.xnodot).fmadd(v_tsince, Vec8::load(c.nodeo));
        Vec8 xnode = Vec8::load(c.xnodcf).fmadd(v_tsq, xnoddf);
        Vec8 tempa = Vec8::load(c.c1).fnmadd(v_tsince, v_one);
        Vec8 tempe = (Vec8::load(c.bstar) * Vec8::load(c.c4)) * v_tsince;
        Vec8 templ = Vec8::load(c.t2cof).fmadd(v_tsq, v_zero);

        // --- NEAR-SPACE UPDATES ---
        Vec8 sw_xmdf, cw_xmdf;
        v_sincos_accurate(xmdf, sw_xmdf, cw_xmdf);
        
        Vec8 etacos = Vec8::load(c.eta).fmadd(cw_xmdf, v_one);
        Vec8 etacos3 = etacos * etacos * etacos;
        Vec8 delm = Vec8::load(c.xmcof) * (etacos3 - Vec8::load(c.delmo));
        Vec8 temp = Vec8::load(c.omgcof).fmadd(v_tsince, delm);
        
        Vec8 xmp = xmdf + temp;
        Vec8 omega = omgadf - temp;
        
        Vec8 tempa_ns = tempa - Vec8::load(c.d2).fmadd(v_tsq, Vec8::load(c.d3).fmadd(v_tcube, Vec8::load(c.d4) * v_tfour));
        Vec8 sw_xmp, cw_xmp;
        v_sincos_accurate(xmp, sw_xmp, cw_xmp);
        Vec8 tempe_ns = tempe + Vec8::load(c.bstar) * Vec8::load(c.c5) * (sw_xmp - Vec8::load(c.sinmo));
        Vec8 templ_ns = templ + Vec8::load(c.t3cof).fmadd(v_tcube, v_tfour * Vec8::load(c.t4cof).fmadd(v_tsince, Vec8::load(c.t5cof)));

        // Masking for simple_model (branchless)
        __mmask8 m_ns = 0;
        for(int s=0; s<8; s++) if(!c.use_simple_model[s]) m_ns |= (1 << s);
        
        tempa.v = _mm512_mask_blend_pd(m_ns, tempa.v, tempa_ns.v);
        tempe.v = _mm512_mask_blend_pd(m_ns, tempe.v, tempe_ns.v);
        templ.v = _mm512_mask_blend_pd(m_ns, templ.v, templ_ns.v);

        Vec8 a = Vec8::load(c.aodp) * tempa * tempa;
        Vec8 e = Vec8::load(c.ecco) - tempe;
        e.v = _mm512_max_pd(e.v, _mm512_set1_pd(1e-6));
        e.v = _mm512_min_pd(e.v, _mm512_set1_pd(1.0 - 1e-6));
        Vec8 xl = xmp + omega + xnode + Vec8::load(c.no_kozai) * templ;

        // --- FINAL POSITION & VELOCITY ---
        Vec8 beta2 = v_one - e * e;
        Vec8 xn = v_kXKE / (a * _mm512_sqrt_pd(a.v));
        Vec8 sw_om, cw_om;
        v_sincos_accurate(omega, sw_om, cw_om);
        
        Vec8 axn = e * cw_om;
        Vec8 inv_a_beta2 = v_one / (a * beta2);
        Vec8 ayn = e.fmadd(sw_om, inv_a_beta2 * Vec8::load(c.aycof));
        Vec8 xlt = xl + inv_a_beta2 * Vec8::load(c.xlcof) * axn;
        
        Vec8 capu = v_fmod_accurate(xlt - xnode, kTWOPI);
        Vec8 epw = capu;
        for(int i=0; i<10; i++) {
            Vec8 sw, cw; v_sincos_accurate(epw, sw, cw);
            Vec8 ec = axn.fmadd(cw, ayn * sw);
            Vec8 es = axn.fmadd(sw, v_zero - ayn * cw);
            epw = epw + (capu - epw + es) / (v_one - ec);
        }
        
        Vec8 sw_epw, cw_epw; v_sincos_accurate(epw, sw_epw, cw_epw);
        Vec8 ecose = axn.fmadd(cw_epw, ayn * sw_epw);
        Vec8 esine = axn.fmadd(sw_epw, v_zero - ayn * cw_epw);
        
        Vec8 temp21 = v_one - (axn * axn + ayn * ayn);
        Vec8 pl = a * temp21;
        Vec8 r = a * (v_one - ecose);
        Vec8 rdot = v_kXKE * _mm512_sqrt_pd(a.v) * esine / r;
        Vec8 rfdot = v_kXKE * _mm512_sqrt_pd(pl.v) / r;
        Vec8 betal = _mm512_sqrt_pd(temp21.v);
        Vec8 temp33 = v_one / (v_one + betal);
        Vec8 cosu_un = (a / r) * (cw_epw - axn + ayn * esine * temp33);
        Vec8 sinu_un = (a / r) * (sw_epw - ayn - axn * esine * temp33);
        
        Vec8 u = v_atan2_accurate(sinu_un, cosu_un);
        Vec8 sin2u = Vec8(2.0) * sinu_un * cosu_un;
        Vec8 cos2u = Vec8(2.0) * cosu_un * cosu_un - v_one;
        
        Vec8 t41 = v_one / pl;
        Vec8 t42 = v_kCK2 * t41;
        Vec8 t43 = t42 * t41;
        
        Vec8 rk = r * (v_one - Vec8(1.5) * t43 * betal * Vec8::load(c.x3thm1)) + Vec8(0.5) * t42 * Vec8::load(c.x1mth2) * cos2u;
        Vec8 uk = u - Vec8(0.25) * t43 * Vec8::load(c.x7thm1) * sin2u;
        Vec8 xnodek = xnode + Vec8(1.5) * t43 * Vec8::load(c.cosio) * sin2u;
        Vec8 xinck = Vec8::load(c.inclo) + Vec8(1.5) * t43 * Vec8::load(c.cosio) * Vec8::load(c.sinio) * cos2u;
        Vec8 rdotk = rdot - xn * t42 * Vec8::load(c.x1mth2) * sin2u;
        Vec8 rfdotk = rfdot + xn * t42 * (Vec8::load(c.x1mth2) * cos2u + Vec8(1.5) * Vec8::load(c.x3thm1));

        Vec8 sn_uk, cs_uk, sn_ik, cs_ik, sn_nk, cs_nk;
        v_sincos_accurate(uk, sn_uk, cs_uk);
        v_sincos_accurate(xinck, sn_ik, cs_ik);
        v_sincos_accurate(xnodek, sn_nk, cs_nk);
        
        Vec8 xmx = (v_zero - sn_nk) * cs_ik;
        Vec8 xmy = cs_nk * cs_ik;
        Vec8 ux = xmx * sn_uk + cs_nk * cs_uk;
        Vec8 uy = xmy * sn_uk + sn_nk * cs_uk;
        Vec8 uz = sn_ik * sn_uk;
        Vec8 vx = xmx * cs_uk - cs_nk * sn_uk;
        Vec8 vy = xmy * cs_uk - sn_nk * sn_uk;
        Vec8 vz = sn_ik * cs_uk;

        alignas(64) double px[8], py[8], pz[8], vxr[8], vyr[8], vzr[8];
        (rk * ux * v_kXKMPER).store(px); (rk * uy * v_kXKMPER).store(py); (rk * uz * v_kXKMPER).store(pz);
        ((rdotk * ux + rfdotk * vx) * v_kXKMPER / Vec8(60.0)).store(vxr);
        ((rdotk * uy + rfdotk * vy) * v_kXKMPER / Vec8(60.0)).store(vyr);
        ((rdotk * uz + rfdotk * vz) * v_kXKMPER / Vec8(60.0)).store(vzr);

        for (int s = 0; s < BATCH_SIZE; s++) {
            if (!c.active[s] || c.use_deep_space[s]) continue;
            pool[b * BATCH_SIZE + s] = Eci(c.epoch[s].AddMinutes(tsince), 
                Vector(px[s], py[s], pz[s]), Vector(vxr[s], vyr[s], vzr[s]));
        }
    }
}

} // namespace libsgp4

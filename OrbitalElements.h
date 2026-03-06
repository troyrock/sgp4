/*
 * Copyright 2013 Daniel Warner <contact@danrw.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#pragma once

#include "Util.h"
#include "DateTime.h"

namespace libsgp4
{

class Tle;

/**
 * @brief The extracted orbital elements used by the SGP4 propagator.
 */
class OrbitalElements
{
public:
    explicit OrbitalElements(const Tle& tle);

    /*
     * XMO
     */
    double MeanAnomoly() const
    {
        return mean_anomoly_;
    }

    /*
     * XNODEO
     */
    double AscendingNode() const
    {
        return ascending_node_;
    }

    /*
     * OMEGAO
     */
    double ArgumentPerigee() const
    {
        return argument_perigee_;
    }

    /*
     * EO
     */
    double Eccentricity() const
    {
        return eccentricity_;
    }

    /*
     * XINCL
     */
    double Inclination() const
    {
        return inclination_;
    }

    /*
     * XNO
     */
    double MeanMotion() const
    {
        return mean_motion_;
    }

    /*
     * BSTAR
     */
    double BStar() const
    {
        return bstar_;
    }

    /*
     * AODP
     */
    double RecoveredSemiMajorAxis() const
    {
        return recovered_semi_major_axis_;
    }

    /*
     * XNODP
     */
    double RecoveredMeanMotion() const
    {
        return recovered_mean_motion_;
    }

    /*
     * PERIGE
     */
    double Perigee() const
    {
        return perigee_;
    }

    /*
     * Period in minutes
     */
    double Period() const
    {
        return period_;
    }

    /*
     * EPOCH
     */
    DateTime Epoch() const
    {
        return epoch_;
    }

    void SetMeanMotion(double n) {
        mean_motion_ = n;
        // re-calculate recovered_mean_motion_ and recovered_semi_major_axis_
        const double a1 = pow(kXKE / mean_motion_, kTWOTHIRD);
        const double cosio = cos(inclination_);
        const double theta2 = cosio * cosio;
        const double x3thm1 = 3.0 * theta2 - 1.0;
        const double eosq = eccentricity_ * eccentricity_;
        const double betao2 = 1.0 - eosq;
        const double betao = sqrt(betao2);
        const double temp = (1.5 * kCK2) * x3thm1 / (betao * betao2);
        const double del1 = temp / (a1 * a1);
        const double a0 = a1 * (1.0 - del1 * (1.0 / 3.0 + del1 * (1.0 + del1 * 134.0 / 81.0)));
        const double del0 = temp / (a0 * a0);
        recovered_mean_motion_ = mean_motion_ / (1.0 + del0);
        recovered_semi_major_axis_ = a0 / (1.0 - del0);
        perigee_ = (recovered_semi_major_axis_ * (1.0 - eccentricity_) - kAE) * kXKMPER;
        period_ = kTWOPI / recovered_mean_motion_;
    }

    void SetBStar(double bstar) {
        bstar_ = bstar;
    }

private:
    double mean_anomoly_;
    double ascending_node_;
    double argument_perigee_;
    double eccentricity_;
    double inclination_;
    double mean_motion_;
    double bstar_;
    double recovered_semi_major_axis_;
    double recovered_mean_motion_;
    double perigee_;
    double period_;
    DateTime epoch_;
};

} // namespace libsgp4

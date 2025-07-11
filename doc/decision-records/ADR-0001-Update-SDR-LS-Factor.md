# ADR-0001: Update the InVEST SDR LS Factor

* Author: James
* Science Lead: Rafa

## Changelog

* 2025-07-09: Updating ADR to include a note about weighted mean of
  proportional flow, described in
  [ADR-0005](ADR-0005-SDR-Weighted-Mean-of-Proportional-Flow.md)
* 2023-08-01: Initial release of this ADR.

## Context

Since we released the updated InVEST SDR model in InVEST 3.1.0, we have seen a
common refrain of users and NatCap science staff noticing that the LS factor
output of SDR did not produce realistic results and that the LS factor produced
by SAGA was much more realistic.  We have over the years made a couple of notable
changes to the model and to the LS factor that have altered the output including:

1. The SDR model's underlying routing model was changed from d-infinity to MFD in 3.5.0
2. The $x$ parameter was changed in InVEST 3.8.1 from the true on-pixel aspect
   $|\sin \theta|+|\cos \theta|$ (described in Zevenbergen & Thorne 1987 and repeated
   in Desmet & Govers 1996) to the weighted mean of proportional flow from the
   current pixel to its neighbors.
3. A typo in a constant value in the LS factor was corrected in InVEST 3.9.1
4. An `l_max` parameter was exposed to the user in InVEST 3.9.1

Despite these changes to the LS factor, we still received occasional reports
describing unrealistic LS factor outputs from SDR and that SAGA's LS factor
was much more realistic.

After diving into the SAGA source code, it turns out that there are several
important differences between the two despite both using Desmet & Govers (1996)
for their LS factor equations:

1. The contributing area $A_{i,j-in}$ is not strictly defined in Desmet &
   Govers (1996), it is only referred to as "the contributing area at the inlet
   of a grid cell with coordinates (i, j) (m^2)".
   InVEST assumes that "contributing area" is $area_{pixel} \cdot n\\_upstream\\_pixels$.
   SAGA refers to this as "specific catchment area" and allows the user to choose their
   specific catchment area equation,  where the available options are
   "contour length simply as cell size", "contour length dependent on aspect", "square
   root of catchment area" and "effective flow length".
2. SAGA uses on-pixel aspect, $|\sin \theta|+|\cos \theta|$, and does not consider
   flow direction derived from a routing model when calculating the LS factor.
3. The length exponent $m$ differs between the implementations.  In SAGA,
   $m = \beta / (1 + \beta)$.  In InVEST, we have a discontinuous function where
   $m$ is dependent on the slope of the current pixel and described as "classical USLE"
   in the user's guide and discussed in Oliveira et al (2013).
4. SAGA's flow accumulation function [`Get_Flow()`](https://github.com/saga-gis/saga-gis/blob/master/saga-gis/src/tools/terrain_analysis/ta_hydrology/Erosion_LS_Fields.cpp#L394)
   only considers a pixel downstream if and only if its elevation is strictly less
   than the current pixel's elevation, which implies that flow accumulation will
   not navigate plateaus.  InVEST's flow accumulation handles plateaus well,
   which can lead to longer flow accumulation values on the same DEM.
5. SAGA's flow accumulation function `Get_Flow()` uses D8, InVEST's flow
   accumulation uses MFD.

It is important to note that when evaluating differences between the SAGA and InVEST
LS Factor implementations, it is _critical_ to use a hydrologically conditioned DEM such
as conditioned by Wang & Liu so that we control for differences in output due
to the presence of plateaus.

Once we finally understood these discrepancies, James implemented several of the
contributing area functions available in SAGA to see what might be most comparable
to the real world.  Source code and a docker container for these experiments are
available at
https://github.com/phargogh/invest-ls-factor-vs-saga/blob/main/src/natcap/invest/sdr/sdr.py#L901.
Some additional discussion and notes can be viewed in the related github issue:
https://github.com/natcap/invest/issues/915.

## Decision

After inspecting the results, Rafa decided that we should make these changes to
the LS Factor calculation:

1. We will revert to using the on-pixel aspect, $|\sin \theta|+|\cos \theta|$.
   This is in line with the published literature.
2. We will convert the "contributing area" portion of the LS Factor to be
   $\sqrt{ n\\_upstream\\_pixels \cdot area\_{pixel} }$. Rafa's opinion on this
   is that the LS factor equations were designed for a 1-dimensional situation,
   so our specific catchment area number should reflect this.

## Status

The above changes were implemented and released in InVEST 3.14.0 (2023-09-08).

### Update 2025-07-09

After implementing the changes described in this document, there remained some
question about the "weighted mean of proportional flow" described above. The
findings and decision related to this have been summarised in
[ADR-0005-SDR-Weighted-Mean-of-Proportional-Flow.md](ADR-0005-SDR-Weighted-Mean-of-Proportional-Flow.md).

## Consequences

Once implemented and released, the LS factor outputs of SDR will be
significantly different, but they should more closely match reality.

We hope that there will be fewer support requests about this once the change is
released.

## References

Zevenbergen & Thorne (1987): Zevenbergen, L.W. and Thorne, C.R. (1987), Quantitative analysis of land surface topography. Earth Surf. Process. Landforms, 12: 47-56. https://doi-org.stanford.idm.oclc.org/10.1002/esp.3290120107

Desmet & Govers (1996): 1. Desmet PJJ, Govers G. A GIS procedure for automatically calculating the USLE LS factor on topographically complex landscape units. J Soil Water Conserv. 1996;51(5):427. https://www.proquest.com/scholarly-journals/gis-procedure-automatically-calculating-usle-ls/docview/220970105/se-2.

Oliveira et al (2013): http://dx.doi.org/10.5772/54439

# ADR-0005: Findings around Weighted Mean of Proportional Flow in SDR

* Author: James
* Science Lead: Rafa

## Changelog

* 2025-07-09: Initial release of this ADR.

## Context

In InVEST 3.14.0, we released a revised LS Factor calculation in the InVEST SDR
in response to critiques that our LS Factor formulation was incorrect, and that
SAGA's LS Factor implementation more closely matched real-world results. The
resulting changes, described in [ADR-0001](ADR-0001-Update-SDR-LS-FActor.md),
were implemented after digging into the source publications and the SAGA source
code.

A part of these changes included removing a computational step that we had
introduced in InVEST 3.8.1, referred to as "Weighted Mean of Proportional
Flow" (hereafter WMPF for short), or "Average Aspect"
([see the version released in InVEST 3.13.0](https://github.com/natcap/invest/blob/ca3051d91f48cbf286f96b83c9e1f2110ba0b2a0/src/natcap/invest/sdr/sdr_core.pyx#L674)).
The intent of this computational step at the time was to represent the
aspect direction as a function of the proportional flow quantity coming into
the current pixel from the surrounding neighbors according to the Multiple Flow
Direction algorithm's computed flow direction, which would provide an
approximation of the aspect as a representation of the water moving across the
pixel. In a preliminary reading of Desmet & Govers (1996), it appeared that
they were using both a WMPF calculation and also dot-product of the slope
vector in order to calculate the LS Factor, which made James doubt that we had
adhered to the methodology described in the paper.

## What is represented in Desmet & Govers (1996)

Desmet & Govers (1996) does indeed use a WMPF claculation in the calculation of
the unit contributing area.  The formulation here is described in equations 2,
3 and 4:

Equation 2 is provided in the context of the Quinn et al. (1991) multiple flow
direction algorithm.  As stated by Desmet & Govers, "In this algorithm, the
contributing area of the central cell in a 3x3 submatrix increased with the
grid cell area is divided over all neighboring cells downslope of the central
cell."  Thus, Equation 2 represents "The fraction received by each downslope
cell", which is "proportional to the product of the distance-weighted drop and
a geometric weight factor, which depends on the direction:"

```math
 A_i = A \frac{\tan \beta_i \cdot W_i}{\sum_{j=1}^{k} \tan \beta_j \cdot W_j}
```
Where:

* $`A_i`$ = Fraction of the contributing area draining through neighbor $i$ (in $`m^2`$)
* $A$ = Upslope area available for distribution (in $`m^2`$)
* $`\tan(\beta_i)`$ = tangent of the slope angle towards neighbor $i$ ($m/m$)
* $`W_i`$ = Weight factor (`0.5` for a cardinal and `0.354` for a diagonal direction) towards neighbor $i$
* $k$ = number of lower neighbors

We also need to compute the effective contour length, which represents "the length of
the line through the grid cell center and perpendicular to the aspect direction."
This is represented by Equation 3:

```math
D_{i,j} = D \cdot (\sin \alpha_{i,j} + \cos \alpha_{i,j}) = D \cdot x_{i,j}
```

Where:

* $`D_{i,j}`$ = the effective contour length (m)
* $`D`$ = the grid cell size (m)
* $`x`$ = $`(\sin \alpha_{i,j} + \cos \alpha_{i,j})`$
* $`\alpha_{i,j}`$ = aspect direction for the grid cell with coordinates (i, j)

Using the above two equations, we can finally calculate the unit contributing area
at the inlet to a pixel in Equation 4:

```math
A_{s_{i,j-in}} = \frac{A_{i,j-in}}{D_{i,j}}
```

Where:

* $`A_{i,j-in}`$ = contributing area at the inlet of a grid cell with coordinates (i, j) ($`m^2`$)
* $`A_{s_{i,j-in}}`$ = unit contributing area at the inlet of grid cell with coordinates (i,j) ($`m^2/m`$)
* $`D_{i,j}`$ = the effective contour length (m)

This unit contributing area is then finally used by Desmet & Govers to calculate the LS Factor,
represented in their paper as Equation 9:

```math

L_{i,j} = \frac{(A_{i,j-in} + D^2)^{m+1} - A_{i,j-in}^{m+1}}{D^{m+2}\cdot x_{i,j}^m \cdot (22.13)^m}
```

And this is reproduced in the [SDR User's Guide chapter](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/sdr.html#equation-ls).


## What SAGA Does

SAGA offers the user a choice of methods with which to calculate the "unit contributing area".
In reading Desmet & Govers (1996) and the SAGA source code, we are interpreting the term
"unit contributing area" to mean "specific catchment area" of a given pixel. Specific
catchment area is a specific and technical term in Hydrology defined as the "area of land
upslope of a width of contour, divided by the contour width"
([source](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2011EO270019)). Unlike InVEST,
SAGA gives users a choice of what to use ([source](https://github.com/saga-gis/saga-gis/blob/master/saga-gis/src/tools/terrain_analysis/ta_hydrology/Erosion_LS_Fields.cpp#L347)):

1. The contour length represented as the number of pixels upstream
2. The contour length dependent on aspect (which mirrors what D&G (1996) uses)
3. The catchment length, calculated as the square root of the catchment area
   (this is what SDR uses today, after we revisited SDR's LS factor this
    last time in 3.14.0)
4. The effective flow length, which appears to be calculated as the sum of the
   log of the pixel lengths of all upstream pixels.


### Are we following Desmet & Govers?

Short answer: No.  But maybe that's OK.

With the changes described in [ADR-0001](ADR-0001-Update-SDR-LS-FActor.md),
we are _not_, in fact, using their formulation of the Unit Contributing Area.

However, we _are_ using an alternative formulation of the Unit Contributing Area,
as represented by $`\sqrt{n\_upstream\_pixels \cdot pixel\_area}`$.  This is
a reasonable estimate of the specific catchment area, as determined by some
rough testing, and is one of the options provided to users in the SAGA interface.

It's also worth noting that Desmet & Govers (1996) specifically call out their
use of the Quinn et al. (1991) Multiple Flow Direction algorithm in order to
calculate the unit contributing area. Since SAGA uses D8 for their LS Factor
calculations it does not make sense to directly adopt Desmet & Govers' formulation
in that context.  It does make sense to instead give users a choice of how they
would like to estimate the unit contributing area.

Our MFD formulation appears to be conceptually similar, if not identical, to
that of Quinn et al. (1991) although a more careful comparison is required.  As
a result, we very likely could include Desmet & Govers' unit contributing area
calculations into the MFD version of SDR if we wanted to.  Further work would
be needed to adapt Desmet & Govers' unit contributing area to work with D8 flow
direction and to document these changes in the InVEST User's Guide.

## Conclusions

If we are using Desmet & Govers (1996) as our point of reference for the LS Factor,
then no, we are not (as of InVEST 3.14.0) following Desmet & Govers in the WMPF
calculations, as we are instead using an alternative, good-enough formulation
of the specific catchment area instead of the unit contributing area, which
Rafa did sign off on.

However, it's also worth noting that the WMPF calculations we were using in InVEST
3.8.0 - 3.13.0 _also_ did not match what Desmet & Govers (1996) described.

If it later becomes desirable or necessary to include Desmet & Govers' WMPF
calculations, we should make sure that we, at least, do the following in
addition to implementing support for WMPF for D8 and MFD:

* Adapt the WMPF calculations (Equation 2) to work with D8
* Document and justify these changes in an ADR
* Describe these changes in the User's Guide

## Status

No changes to InVEST are required due to this ADR.

## References

Desmet & Govers (1996): 1. Desmet PJJ, Govers G. A GIS procedure for automatically calculating the USLE LS factor on topographically complex landscape units. J Soil Water Conserv. 1996;51(5):427. https://www.proquest.com/scholarly-journals/gis-procedure-automatically-calculating-usle-ls/docview/220970105/se-2.

Quinn et al. (1991): Quinn, P., Beven, K., Chevallier, P. and Planchon, O. (1991), The prediction of hillslope flow paths for distributed hydrological modelling using digital terrain models. Hydrol. Process., 5: 59-79. https://doi-org.stanford.idm.oclc.org/10.1002/hyp.3360050106

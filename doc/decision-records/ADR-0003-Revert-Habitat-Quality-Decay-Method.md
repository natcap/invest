# ADR-0003: Revert Habitat Quality Decay Method

Author: Doug

Science Lead(s): Lisa, Stacie, Jade

## Context
The Habitat Quality model has used convolutions as an implementation for
decaying threat rasters over distance since InVEST 3.3.0. This approach
strayed from the implementation described in the User's Guide and the User's
Guide was never updated to reflect it. The User's Guide described decay using
an euclidean distance implementation. When thinking about updating the User's
Guide to reflect the convolution implementation it was not clear that the
exponential decay via convolutions was providing the desired result. The
justification for the convolution method was to better reflect the real world
in how the density of a threat or surrounding threat pixels could have an even
greater, cumulative impact and degradation over space. However, the degradation
raster produced from the filtered threat rasters did not make intuitive sense.

Stacie noted that users via the Forum were reporting HQ outputs that were all
at the very high end of the 0-1 range and this didn't happen prior to the
convolution implementation. The degradation outputs were all very low too. I
believe the reason these values were not reflecting a 0-1 index response for
degradation was because the convolution approach ends up calculating the
impact of each threat ($i_{rxy}$ in degradation equation) to be a very small
number even if the distance is very small (meaning the pixel is close to the
threat).

We also investigated why the exponential decay equation was using a constant
of 2.99 as a scalar. With the constant of 2.99, the impact of the threat is
reduced by 95% (to 5%) at the specified max threat distance. So we suspect
it's based on the traditional 95% cutoff that used in statistics. We could
tweak this cutoff (e.g., 99% decay at max distance), if we wanted.

## Decision
After talking things over with the science team (Lisa, Stacie, Jade) we
decided to switch to a simpler euclidean distance implementation and to
update the User's Guide with why the 2.99 constant is being used.

## Status
Completed and released in InVEST 3.13.0 (2023-03-17)

## Consequences
The degradation and quality outputs will be quite different from previous
versions but should be more intuitive to interpret.

We should see less user forum questions regarding this topic.

There should be a noticeable runtime improvement from calculating euclidean
distances vs convolutions.

## References
GitHub:
  * [Pull Request](https://github.com/natcap/invest/pull/1159)
  * [Degradation range discussion](https://github.com/natcap/invest/issues/646)
  * [Decay function discrepency](https://github.com/natcap/invest/issues/1104)
  * [Users Guide PR](https://github.com/natcap/invest.users-guide/pull/109)

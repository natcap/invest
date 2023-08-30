# ADR-0002: Switch the InVEST License

Author: Doug

Science Lead: N/A

## Context

NatCap and Stanford OTL decided to trademark "InVEST" and register it with 
the USPTO. During those discussions Stanford OTL suggested that we revisit 
our 3-Clause BSD License and make sure it still met our needs. There was also
concern about whether InVEST was being fairly attributed in derivative works 
and whether the InVEST License should play a role in helping define attribution.
While reviewing various licenses the intention was to always stick with a
permissive open source license and not move to a copyleft or more restrictive 
license. Doug did a thorough audit of licenses, specifically looking at how 
permissive open source licenses and software projects handle attribution.
Doug collaborated with James, Lisa, and others throughout this process.

## Decision

After reviewing different possibilities Doug, with approval from Lisa and the
Software Team, decided we should switch to the Apache 2.0 License. The Apache
2.0 License provides the following benefits while remaining a permissive open 
source license:

1. Explicitly states the terms in clear language, including attribution and
   trademark guidelines, which helps remove ambiguity and room for
   interpretation.
2. Includes a clause requiring derivative works to clearly note any major changes
   to the original source. This felt like a nice addition given the scientific 
   nature of our software and quality that our software is known for.
3. Is a widely established and adopted license that is useable "as is".

I will note that after many discussions about whether we could or should
include more strict attribution requirements in the license the NatCap Stanford 
Leadership Team decided the license was not the place to address those issues.

Doug and James did have a discussion about whether we should reach out to prior
contributors to get their sign off on switching licenses. Doug had noticed this
was something other large open source projects had to contend with when 
switching licenses. However, because the license remains permissive and the 
fact that all major InVEST source code contributors mostly pushed changes as a
NatCap Stanford University employee, we did not feel it necessary.

## Status

Complete. Released in InVEST 3.14.0.

## Consequences

This should have limited impact given the Apache 2.0 License is also a 
permissive license.

We will now distribute a NOTICE file alongside the LICENSE file that contains
custom copyright language. Derivative works are required to also distribute 
this file alongside the Apache 2.0 License.

We hope that derivative works who make changes to InVEST source will note those
in a reasonable way.

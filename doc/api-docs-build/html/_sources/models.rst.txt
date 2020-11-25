
.. _models:

=========================
InVEST Model Entry Points
=========================

All InVEST models share a consistent python API:

    1) The model has a function called ``execute`` that takes a single python
       dict (``"args"``) as its argument.
    2) This arguments dict contains an entry, ``'workspace_dir'``, which
       points to the folder on disk where all files created by the model
       should be saved.

Calling a model requires importing the model's execute function and then
calling the model with the correct parameters.  For example, if you were
to call the Carbon Storage and Sequestration model, your script might
include

.. code-block:: python

    import natcap.invest.carbon.carbon_combined
    args = {
        'workspace_dir': 'path/to/workspace'
        # Other arguments, as needed for Carbon.
    }

    natcap.invest.carbon.carbon_combined.execute(args)

For examples of scripts that could be created around a model run,
or multiple successive model runs, see :ref:`CreatingSamplePythonScripts`.


.. contents:: Available Models and Tools:
    :local:

Annual Water Yield: Reservoir Hydropower Production
===================================================
.. autofunction:: natcap.invest.hydropower.hydropower_water_yield.execute

Coastal Blue Carbon Preprocessor
================================
.. autofunction:: natcap.invest.coastal_blue_carbon.preprocessor.execute

Crop Production Percentile Model
================================
.. autofunction:: natcap.invest.crop_production_percentile.execute

Crop Production Regression Model
================================
.. autofunction:: natcap.invest.crop_production_regression.execute

DelineateIt: Watershed Delineation
==================================
.. autofunction:: natcap.invest.delineateit.delineateit.execute

Finfish Aquaculture
===================
.. autofunction:: natcap.invest.finfish_aquaculture.finfish_aquaculture.execute

Fisheries
=========
.. autofunction:: natcap.invest.fisheries.fisheries.execute

Fisheries: Habitat Scenario Tool
================================
.. autofunction:: natcap.invest.fisheries.fisheries_hst.execute

Forest Carbon Edge Effect
=========================
.. autofunction:: natcap.invest.forest_carbon_edge_effect.execute

GLOBIO
======
.. autofunction:: natcap.invest.globio.execute

Habitat Quality
===============
.. autofunction:: natcap.invest.habitat_quality.execute

InVEST Carbon Model
===================
.. autofunction:: natcap.invest.carbon.execute

InVEST Coastal Vulnerability Model
==================================
.. autofunction:: natcap.invest.coastal_vulnerability.execute

InVEST Habitat Risk Assessment (HRA) Model
==========================================
.. autofunction:: natcap.invest.hra.execute

InVEST Pollination Model
========================
.. autofunction:: natcap.invest.pollination.execute

Model Coastal Blue Carbon over a time series
============================================
.. autofunction:: natcap.invest.coastal_blue_carbon.coastal_blue_carbon.execute

Nutrient Delivery Ratio
=======================
.. autofunction:: natcap.invest.ndr.ndr.execute

Recreation
==========
.. autofunction:: natcap.invest.recreation.recmodel_client.execute

RouteDEM: Hydrological routing
==============================
.. autofunction:: natcap.invest.routedem.execute

Run the Scenic Quality Model
============================
.. autofunction:: natcap.invest.scenic_quality.scenic_quality.execute

Scenario Generator: Proximity-Based
===================================
.. autofunction:: natcap.invest.scenario_gen_proximity.execute

Seasonal Water Yield
====================
.. autofunction:: natcap.invest.seasonal_water_yield.seasonal_water_yield.execute

Sediment Delivery Ratio
=======================
.. autofunction:: natcap.invest.sdr.sdr.execute

Urban Cooling Model
===================
.. autofunction:: natcap.invest.urban_cooling_model.execute

Urban Flood Risk Mitigation model
=================================
.. autofunction:: natcap.invest.urban_flood_risk_mitigation.execute

Wave Energy
===========
.. autofunction:: natcap.invest.wave_energy.execute

Wind Energy
===========
.. autofunction:: natcap.invest.wind_energy.execute


"""InVEST Timber model."""

import os
import math
import csv

from osgeo import ogr


def execute(args):
    """
    Invoke the timber model given uri inputs specified by the user guide.

    Args:
        args['workspace_dir'] (string): The file location where the outputs
            will be written (Required)
        args['results_suffix']  (string): a string to append to any output file
            name (optional)
        args['timber_shape_uri'] (string): The shapefile describing timber
            parcels with fields as described in the user guide (Required)
        args['attr_table_uri'] (string): The CSV attribute table
            location with fields that describe polygons in timber_shape_uri
            (Required)
        market_disc_rate (float): The market discount rate

    Returns:
        nothing
    """
    # append a _ to the suffix if it's not empty and doens't already have one
    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    timber_shape = ogr.Open(args['timber_shape_uri'], 1)

    # Add the Output directory onto the given workspace
    workspace_dir = os.path.join(args['workspace_dir'], 'output')
    if not os.path.isdir(workspace_dir):
        os.makedirs(workspace_dir)

    shape_source = os.path.join(workspace_dir, 'timber%s.shp' % file_suffix)

    # If there is already an existing shapefile with the same name
    # and path, delete it
    if os.path.isfile(shape_source):
        os.remove(shape_source)

    # Copy the input shapefile into the designated output folder
    driver = ogr.GetDriverByName('ESRI Shapefile')
    copy = driver.CopyDataSource(timber_shape, shape_source)

    # OGR closes datasources this way to make sure data gets flushed properly
    timber_shape.Destroy()
    copy.Destroy()

    timber_output_shape = ogr.Open(shape_source, 1)

    layer = timber_output_shape.GetLayerByName('timber%s' % file_suffix)
    # Set constant variables from arguments
    mdr = args['market_disc_rate']

    attr_table = open(args['attr_table_uri'], 'rU')
    reader = csv.DictReader(attr_table)

    # Making a shallow copy of the attribute 'fieldnames' explicitly to
    # edit to all the fields to lowercase because it is more readable
    # and easier than editing the attribute itself
    field_names = reader.fieldnames

    for index in xrange(len(field_names)):
        field_names[index] = field_names[index].lower()

    parcel_id_lookup = {}
    # Iterate through the CSV file and construct a lookup dictionary
    for row in reader:
        parcel_id_lookup[int(row['parcel_id'])] = row

    attr_table.close()

    # Set constant variables for calculations
    mdr_perc = 1 + (mdr / 100.00)
    sumtwo_lower_limit = 0

    # Create three new fields on the shapefile's polygon layer
    for fieldname in ('TNPV', 'TBiomass', 'TVolume'):
        field_def = ogr.FieldDefn(fieldname, ogr.OFTReal)
        layer.CreateField(field_def)

    def _npv_summation_one(
            lower, upper, harvest_value, mdr_perc, freq_harv, subtractor):
        """Calculate the first summation for the npv of a parcel.

        Parameters:
            lower (int) : lower limit for the summation
            upper (int) : upper limit for the summation
            harvest_value (float) : the harvested value for a parcel
            mdr_perc (float) : the discount rate factor as percent
            freq_harv (float) : the frequency of harvest periods, in years,
                for each parcel
            subtractor (float) : used to distinguish if immediate harvest
                occurs

        Returns:
            The first summation as a float
        """
        summation = 0.0
        upper = upper + 1
        for num in xrange(lower, upper):
            summation = summation + (
                harvest_value / (mdr_perc ** ((freq_harv * num) - subtractor)))

        return summation

    def _npv_summation_two(lower, upper, maint_cost, mdr_perc):
        """Calculate the second summation for the npv of a parcel.

        Parameters:
            lower (int) : lower limit for the summation
            upper (int) : upper limit for the summation
            maint_cost (float) : the cost to maintain the parcel
            mdr_perc (float) : the discount rate factor as percent

        Returns:
            The second summation as a float
        """
        summation = 0.0
        upper = upper + 1
        for num in xrange(lower, upper):
            summation = summation + (maint_cost / (mdr_perc ** num))

        return summation

    # Loop through each feature (polygon) in the shapefile layer
    for feat in layer:
        # Get the correct polygon attributes to be calculated by matching the
        # feature's polygon Parcl_ID with the attribute tables polygon
        # Parcel_ID
        parcl_index = feat.GetFieldIndex('Parcl_ID')
        parcl_id = feat.GetField(parcl_index)
        attr_row = parcel_id_lookup[parcl_id]
        # Set polygon attribute values from row
        freq_harv = float(attr_row['freq_harv'])
        num_years = float(attr_row['t'])
        harv_mass = float(attr_row['harv_mass'])
        harv_cost = float(attr_row['harv_cost'])
        price = float(attr_row['price'])
        maint_cost = float(attr_row['maint_cost'])
        bcef = float(attr_row['bcef'])
        parcl_area = float(attr_row['parcl_area'])
        perc_harv = float(attr_row['perc_harv'])
        immed_harv = attr_row['immed_harv']

        sumtwo_upper_limit = int(num_years - 1)
        # Variable used in npv summation one equation as a distinguisher
        # between two immed_harv possibilities
        subtractor = 0.0
        yr_per_freq = num_years / freq_harv

        # Calculate the harvest value for parcel x
        harvest_value = (
            perc_harv / 100.00) * ((price * harv_mass) - harv_cost)

        # Initiate the biomass variable. Depending on 'immed_Harv' biomass
        # calculation will differ
        biomass = None

        # Check to see if immediate harvest will occur and act accordingly
        if immed_harv.upper() == 'N' or immed_harv.upper() == 'NO':
            sumone_upper_limit = int(math.floor(yr_per_freq))
            sumone_lower_limit = 1
            subtractor = 1.0
            summation_one = _npv_summation_one(
                sumone_lower_limit, sumone_upper_limit, harvest_value,
                mdr_perc, freq_harv, subtractor)
            summation_two = _npv_summation_two(
                sumtwo_lower_limit, sumtwo_upper_limit, maint_cost, mdr_perc)
            # Calculate Biomass
            biomass = (parcl_area * (perc_harv / 100.00) * harv_mass *
                       math.floor(yr_per_freq))

        elif immed_harv.upper() == 'Y' or immed_harv.upper() == 'YES':
            sumone_upper_limit = int((math.ceil(yr_per_freq) - 1.0))
            sumone_lower_limit = 0
            summation_one = _npv_summation_one(
                sumone_lower_limit, sumone_upper_limit, harvest_value,
                mdr_perc, freq_harv, subtractor)
            summation_two = _npv_summation_two(
                sumtwo_lower_limit, sumtwo_upper_limit, maint_cost, mdr_perc)
            # Calculate Biomass
            biomass = (
                parcl_area * (perc_harv / 100.00) * harv_mass *
                math.ceil(yr_per_freq))

        # Calculate Volume
        volume = biomass * (1.0 / bcef)

        net_present_value = (summation_one - summation_two)
        total_npv = net_present_value * parcl_area

        # For each new field set the corresponding value to the specific
        # polygon
        for field, value in (
                ('TNPV', total_npv), ('TBiomass', biomass),
                ('TVolume', volume)):
            index = feat.GetFieldIndex(field)
            feat.SetField(index, value)

        # save the field modifications to the layer.
        layer.SetFeature(feat)
        feat.Destroy()

    # OGR closes datasources this way to make sure data gets flushed properly
    timber_output_shape.Destroy()

    # Wipe datasources
    copy = None
    timber_shape = None
    timber_output_shape = None

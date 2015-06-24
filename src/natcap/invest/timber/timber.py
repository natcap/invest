"""InVEST Timber model at the "uri" level.  No separation between
    biophysical and valuation since the model is so simple."""

import sys
import os
import math

from osgeo import ogr

from natcap.invest.dbfpy import dbf


def execute(args):
    """
    This function invokes the timber model given uri inputs specified by
    the user guide.

    Args:
        args['workspace_dir'] (string): The file location where the outputs will
            be written (Required)
        args['results_suffix']  (string): a string to append to any output file
            name (optional)
        args['timber_shape_uri'] (string): The shapefile describing timber
            parcels with fields as described in the user guide (Required)
        args['attr_table_uri'] (string): The DBF polygon attribute table
            location with fields that describe polygons in timber_shape_uri
            (Required)
        market_disc_rate (float): The market discount rate

    Returns:
        nothing

    """

    #append a _ to the suffix if it's not empty and doens't already have one
    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    filesystemencoding = sys.getfilesystemencoding()

    timber_shape = ogr.Open(
        args['timber_shape_uri'].encode(filesystemencoding), 1)

    #Add the Output directory onto the given workspace
    workspace_dir = args['workspace_dir'] + os.sep + 'output/'
    if not os.path.isdir(workspace_dir):
        os.makedirs(workspace_dir)

    #CopyDataSource expects a python string, yet some versions of json load a
    #'unicode' object from the dumped command line arguments.  The cast to a
    #python string here should ensure we are able to proceed.
    shape_source = str(workspace_dir + 'timber%s.shp' % file_suffix)

    #If there is already an existing shapefile with the same name
    #and path, delete it
    if os.path.isfile(shape_source):
        os.remove(shape_source)

    #Copy the input shapefile into the designated output folder
    driver = ogr.GetDriverByName('ESRI Shapefile')
    copy = driver.CopyDataSource(timber_shape, shape_source)

    #OGR closes datasources this way to make sure data gets flushed properly
    timber_shape.Destroy()
    copy.Destroy()

    timber_output_shape = ogr.Open(shape_source.encode(filesystemencoding), 1)

    layer = timber_output_shape.GetLayerByName('timber%s' % file_suffix)
    #Set constant variables from arguments
    mdr = args['market_disc_rate']
    attr_table = dbf.Dbf(args['attr_table_uri'], readOnly=True)
    #Set constant variables for calculations
    mdr_perc = 1 + (mdr / 100.00)
    sumtwo_lower_limit = 0

    #Create three new fields on the shapefile's polygon layer
    for fieldname in ('TNPV', 'TBiomass', 'TVolume'):
        field_def = ogr.FieldDefn(fieldname, ogr.OFTReal)
        layer.CreateField(field_def)

    #Build a lookup table mapping the Parcel_IDs and corresponding row index
    parcel_id_lookup = {}
    for i in range(attr_table.recordCount):
        parcel_id_lookup[attr_table[i]['Parcel_ID']] = attr_table[i]

    #Loop through each feature (polygon) in the shapefile layer
    for feat in layer:
        #Get the correct polygon attributes to be calculated by matching the
        #feature's polygon Parcl_ID with the attribute tables polygon Parcel_ID
        parcl_index = feat.GetFieldIndex('Parcl_ID')
        parcl_id = feat.GetField(parcl_index)
        attr_row = parcel_id_lookup[parcl_id]
        #Set polygon attribute values from row
        freq_harv = attr_row['Freq_harv']
        num_years = float(attr_row['T'])
        harv_mass = attr_row['Harv_mass']
        harv_cost = attr_row['Harv_cost']
        price = attr_row['Price']
        maint_cost = attr_row['Maint_cost']
        bcef = attr_row['BCEF']
        parcl_area = attr_row['Parcl_area']
        perc_harv = attr_row['Perc_harv']
        immed_harv = attr_row['Immed_harv']

        sumtwo_upper_limit = int(num_years - 1)
        #Variable used in npv summation one equation as a distinguisher
        #between two immed_harv possibilities
        subtractor = 0.0
        yr_per_freq = num_years / freq_harv

        #Calculate the harvest value for parcel x
        harvest_value = (perc_harv / 100.00) * ((price * harv_mass) - harv_cost)

        #Initiate the biomass variable. Depending on 'immed_Harv' biomass
        #calculation will differ
        biomass = None

        #Check to see if immediate harvest will occur and act accordingly
        if immed_harv.upper() == 'N' or immed_harv.upper() == 'NO':
            sumone_upper_limit = int(math.floor(yr_per_freq))
            sumone_lower_limit = 1
            subtractor = 1.0
            summation_one = npv_summation_one(
                sumone_lower_limit, sumone_upper_limit, harvest_value,
                mdr_perc, freq_harv, subtractor)
            summation_two = npv_summation_two(
                sumtwo_lower_limit, sumtwo_upper_limit, maint_cost, mdr_perc)
            #Calculate Biomass
            biomass = \
                    parcl_area * (perc_harv / 100.00) * harv_mass \
                    * math.floor(yr_per_freq)
        elif immed_harv.upper() == 'Y' or immed_harv.upper() == 'YES':
            sumone_upper_limit = int((math.ceil(yr_per_freq) - 1.0))
            sumone_lower_limit = 0
            summation_one = npv_summation_one(
                sumone_lower_limit, sumone_upper_limit, harvest_value,
                mdr_perc, freq_harv, subtractor)
            summation_two = npv_summation_two(
                sumtwo_lower_limit, sumtwo_upper_limit, maint_cost, mdr_perc)
            #Calculate Biomass
            biomass = (
                parcl_area * (perc_harv / 100.00) * harv_mass *
                math.ceil(yr_per_freq))

        #Calculate Volume
        volume = biomass * (1.0 / bcef)

        net_present_value = (summation_one - summation_two)
        total_npv = net_present_value * parcl_area

        #For each new field set the corresponding value to the specific polygon
        for field, value in (
                ('TNPV', total_npv), ('TBiomass', biomass),
                ('TVolume', volume)):
            index = feat.GetFieldIndex(field)
            feat.SetField(index, value)

        #save the field modifications to the layer.
        layer.SetFeature(feat)
        feat.Destroy()

    #OGR closes datasources this way to make sure data gets flushed properly
    timber_output_shape.Destroy()

    #Close the polygon attribute table DBF file and wipe datasources
    attr_table.close()
    copy = None
    timber_shape = None
    timber_output_shape = None


#Calculates the first summation for the net present value of a parcel
def npv_summation_one(
        lower, upper, harvest_value, mdr_perc, freq_harv, subtractor):
    summation = 0.0
    upper = upper + 1
    for num in range(lower, upper):
        summation = summation + (
            harvest_value / (mdr_perc ** ((freq_harv * num) - subtractor)))

    return summation

#Calculates the second summation for the net present value of a parcel
def npv_summation_two(lower, upper, maint_cost, mdr_perc):
    summation = 0.0
    upper = upper + 1
    for num in range(lower, upper):
        summation = summation + (maint_cost / (mdr_perc ** num))

    return summation

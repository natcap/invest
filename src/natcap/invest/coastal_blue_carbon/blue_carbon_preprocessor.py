import logging
import os
import math

import pygeoprocessing.geoprocessing
from osgeo import gdal

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.blue_carbon.preprocessor')


def get_transition_set_count_from_uri(dataset_uri_list, ignore_nodata=True):
    '''

    '''
    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(dataset_uri_list[0])
    lulc_nodata = int(pygeoprocessing.geoprocessing.get_nodata_from_uri(dataset_uri_list[0]))
    nodata = 0

    #reclass rasters to compact bit space
    lulc_codes = set()
    unique_raster_values_count = {}

    for dataset_uri in dataset_uri_list:
        unique_raster_values_count[dataset_uri] = pygeoprocessing.geoprocessing.unique_raster_values_count(dataset_uri)
        lulc_codes.update(unique_raster_values_count[dataset_uri].keys())

    lulc_codes = list(lulc_codes)
    lulc_codes.sort()

    if len(lulc_codes) < 2 ** 8:
        data_type = gdal.GDT_UInt16
        shift = 8
    elif len(lulc_codes) < 2 ** 16:
        data_type = gdal.GDT_UInt32
        shift = 16
    else:
        raise ValueError, "Too many LULC codes."

    #renumber and reclass rasters
    reclass_orig_dict = dict(zip(lulc_codes,range(1,len(lulc_codes)+1)))

    reclass_dest_dict = {}
    for key in reclass_orig_dict:
        reclass_dest_dict[key] = reclass_orig_dict[key] << shift

    def add_op(orig, dest):
        return orig + dest

    counts={}
    for i in range(len(dataset_uri_list)-1):
        orig_uri = pygeoprocessing.geoprocessing.temporary_filename()
        dest_uri = pygeoprocessing.geoprocessing.temporary_filename()
        multi_uri = pygeoprocessing.geoprocessing.temporary_filename()

        #reclass orig values
        pygeoprocessing.geoprocessing.reclassify_dataset_uri(dataset_uri_list[i],
                                            reclass_orig_dict,
                                            orig_uri,
                                            data_type,
                                            nodata,
                                            exception_flag="values_required")

        #reclass dest values
        pygeoprocessing.geoprocessing.reclassify_dataset_uri(dataset_uri_list[i+1],
                                            reclass_dest_dict,
                                            dest_uri,
                                            data_type,
                                            nodata,
                                            exception_flag="values_required")

        #multiplex orig with dest
        pygeoprocessing.geoprocessing.vectorize_datasets([orig_uri, dest_uri],
                                        add_op,
                                        multi_uri,
                                        data_type,
                                        nodata,
                                        cell_size,
                                        "union")

        #get unique counts
        counts[i]=pygeoprocessing.geoprocessing.unique_raster_values_count(multi_uri, False)

    restore_classes = {}
    for key in reclass_orig_dict:
        restore_classes[reclass_orig_dict[key]] = key
    restore_classes[nodata] = lulc_nodata

    LOGGER.debug("Decoding transition table.")
    transitions = {}
    for key in counts:
        transitions[key]={}
        for k in counts[key]:
            try:
                orig = restore_classes[k % (2**shift)]
            except KeyError:
                orig = lulc_nodata
            try:
                dest = restore_classes[k >> shift]
            except KeyError:
                dest = lulc_nodata

            try:
                transitions[key][orig][dest] = counts[key][k]
            except KeyError:
                transitions[key][orig] = {dest : counts[key][k]}

    if ignore_nodata:
        for r in unique_raster_values_count:
            if lulc_nodata in unique_raster_values_count[r]:
                unique_raster_values_count[r].pop(lulc_nodata)

        for r in transitions:
            if lulc_nodata in transitions[r]:
                transitions[r].pop(lulc_nodata)

    return unique_raster_values_count, transitions


def execute(args):
    '''

    '''
    transition_matrix_uri = os.path.join(
        args["workspace_dir"], "transition.csv")
    values_matrix_uri = args["preprocessor_key_uri"]
    values_matrix_id = "Id"

    report_uri = os.path.join(args["workspace_dir"], "preprocessor_report.htm")

    nodata = set([pygeoprocessing.geoprocessing.get_nodata_from_uri(
        uri) for uri in args["lulc"]])

    LOGGER.debug("Validating LULCs.")
    if len(nodata) > 1:
        msg = "The LULCs contain more than one no data value."
        LOGGER.error(msg)
        raise ValueError, msg

    nodata = list(nodata)[0]
    LOGGER.debug("No data value is %i.", nodata)

   # #It might be handy allow for a SetCategoryNames
   # dataset = gdal.Open(args["lulc"][0])
   # band = dataset.GetRasterBand(1)
   # names = band.GetCategoryNames()
   # band = None
   # dataset = None

   # if not names == None:
   #     LOGGER.debug("Found category names: %s.", names)
   # else:
   #     LOGGER.debug("No imbedded category names found.")

    LOGGER.info("Reading all transitions.")
    unique_raster_values_count, transition_counts = \
        get_transition_set_count_from_uri(args["lulc"])
    count_max = 0
    for key in unique_raster_values_count:
        for value in unique_raster_values_count[key]:
            if unique_raster_values_count[key][value] > count_max:
                count_max = unique_raster_values_count[key][value]
    count_width = int(math.log10(count_max))

    transitions = set()
    original_values = set()
    final_values = set()
    transition_width = 0
    no_data_msg = "No data values cannot change."
    for transition in transition_counts:
        for orig in transition_counts[transition]:
            for dest in transition_counts[transition][orig]:
                if (orig == nodata) != (dest == nodata):
                    LOGGER.error(no_data_msg)
                    raise ValueError, no_data_msg

                original_values.add(orig)
                final_values.add(dest)
                if (orig != nodata) and (dest != nodata):
                    transitions.add((orig, dest))

                width = int(math.log10(
                    transition_counts[transition][orig][dest]))
                if width > transition_width:
                    transition_width = width

    LOGGER.info("Creating transition matrix.")
    original_values = list(original_values)
    final_values = list(final_values)
    original_values.sort()
    final_values.sort()
    transition_matrix = open(transition_matrix_uri, 'w')
    transition_matrix.write("Id,Name,")
    transition_matrix.write(",".join([str(value) for value in final_values]))

    args["lulc_id"] = "Id"
    args["lulc_name"] = "Name"
    args["lulc_type"] = "Veg Type"

    labels_dict = {}

    #This will cause problems if the carbon table is missing more than one label.
    if args["labels"] != "":
        LOGGER.info("Reading category names from table.")
        labels_dict = pygeoprocessing.geoprocessing.get_lookup_from_csv(args["labels"], args["lulc_id"])

    values = pygeoprocessing.geoprocessing.get_lookup_from_csv(values_matrix_uri, values_matrix_id)
    for original in original_values:
        transition_matrix.write("\n%i" % original)
        if original in labels_dict:
            transition_matrix.write(",%s" % labels_dict[original][args["lulc_name"]])
        else:
            transition_matrix.write(",")

        for final in final_values:
            if (original, final) in transitions:
                transition_matrix.write(",%s" % values[labels_dict[original][args["lulc_type"]]][str(labels_dict[final][args["lulc_type"]])])
            else:
                transition_matrix.write(",%s" % "None")
    transition_matrix.write('\n\n,Replace all instances of "Disturbance"\n'
                            ',in the above matrix with either:\n,Low '
                            'Disturbance\n,Medium Disturbance\n,High Disturbance')
    transition_matrix.close()

    #open report
    report = open(report_uri, 'w')
    report.write("<HTML><TITLE>InVEST - Blue Carbon Preprocessor Report</TITLE><BODY>")

    #summary table
    all_values = list(set(original_values + final_values))
    all_values.sort()
    report.write("\n<P><P><B>LULC Summary</B>")
    column_name_list = ["Name"] + [str(val).ljust(count_width, "#").replace("#", "&ensp;") for val in all_values]
    report.write("\n<TABLE BORDER=1><TR><TD><B>%s</B></TD></TR>" % "</B></TD><TD><B>".join(column_name_list))
    for dataset_uri in args["lulc"]:
        report.write("\n<TR align=\"right\"><TD>%s</TD>" % os.path.basename(dataset_uri))
        for val in all_values:
            try:
                report.write("<TD>%i</TD>" % unique_raster_values_count[dataset_uri][val])
            except KeyError:
                report.write("<TD>%i</TD>" % 0)
        report.write("</TR>")
    report.write("\n</TABLE>")

    #transition count tables
    veg_types = set([labels_dict[k][args["lulc_type"]] for k in labels_dict])
    veg_counts = {}
    for transition in transition_counts:
        veg_counts[transition] = {}
        for orig in transition_counts[transition]:
            for dest in transition_counts[transition][orig]:
                orig_veg = labels_dict[orig][args["lulc_type"]]
                dest_veg = labels_dict[dest][args["lulc_type"]]

                if not (orig_veg in veg_counts[transition]):
                    veg_counts[transition][orig_veg] = {}

                if not (dest_veg in veg_counts[transition][orig_veg]):
                    veg_counts[transition][orig_veg][dest_veg] = 0

                veg_counts[transition][orig_veg][dest_veg] += \
                    transition_counts[transition][orig][dest]

    for transition in transition_counts:
        report.write("\n<P><P><B>LULC Transition %i</B>" % (transition + 1))
        column_name_list = [""] + [str(val).ljust(transition_width, "#").replace("#", "&ensp;") for val in final_values]
        report.write("\n<TABLE BORDER=1><TR><TD><B>%s</B></TD></TR>" % "</B></TD><TD><B>".join(column_name_list))
        for orig in original_values:
            report.write("\n<TR align=\"right\"><TD><B>%i<B></TD>" % orig)
            for dest in final_values:
                try:
                    report.write("<TD>%i</TD>" % transition_counts[transition][orig][dest])
                except KeyError:
                    report.write("<TD><font color=lightgray>%i</font></TD>" % 0)

        report.write("\n</TABLE>")

        report.write("\n<P><P><B>Vegetation Transition %i</B>" % (transition +1))
        column_name_list = [""] + [str(val).ljust(transition_width, "#").replace("#", "&ensp;") for val in veg_types]
        report.write("\n<TABLE BORDER=1><TR><TD><B>%s</B></TD></TR>" % "</B></TD><TD><B>".join(column_name_list))

        for orig in veg_types:
            report.write("\n<TR align=\"right\"><TD><B>%i<B></TD>" % orig)
            for dest in veg_types:
                try:
                    report.write("<TD>%i</TD>" % veg_counts[
                        transition][orig][dest])
                except KeyError:
                    report.write("<TD><font color=lightgray>%i</font></TD>" % 0)

        report.write("\n</TABLE>")

    #close report
    report.close()

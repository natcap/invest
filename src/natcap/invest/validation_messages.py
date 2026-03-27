from . import gettext

MISSING_KEY = gettext('Key is missing from the args dict')
MISSING_VALUE = gettext('Input is required but has no value')
MATCHED_NO_HEADERS = gettext(
    'Expected the {header} "{header_name}" but did not find it')
PATTERN_MATCHED_NONE = gettext(
    'Expected to find at least one {header} matching '
    'the pattern "{header_name}" but found none')
DUPLICATE_HEADER = gettext(
    'Expected the {header} "{header_name}" only once '
    'but found it {number} times')
NOT_A_NUMBER = gettext(
    'Value "{value}" could not be interpreted as a number')
WRONG_PROJECTION_UNIT = gettext(
    'Layer must be projected in this unit: '
    '"{unit_a}" but found this unit: "{unit_b}"')
UNEXPECTED_ERROR = gettext('An unexpected error occurred in validation')
DIR_NOT_FOUND = gettext('Directory not found')
NOT_A_DIR = gettext('Path must be a directory')
FILE_NOT_FOUND = gettext('File not found')
INVALID_PROJECTION = gettext('Dataset must have a valid projection.')
NOT_PROJECTED = gettext('Dataset must be projected in linear units.')
NOT_GDAL_RASTER = gettext('File could not be opened as a GDAL raster')
OVR_FILE = gettext('File found to be an overview ".ovr" file.')
NOT_GDAL_VECTOR = gettext('File could not be opened as a GDAL vector')
REGEXP_MISMATCH = gettext(
    "Value did not match expected pattern {regexp}")
INVALID_OPTION = gettext("Value must be one of: {option_list}")
INVALID_VALUE = gettext('Value does not meet condition {condition}')
NOT_WITHIN_RANGE = gettext('Value {value} is not in the range {range}')
NOT_AN_INTEGER = gettext('Value "{value}" does not represent an integer')
NOT_BOOLEAN = gettext("Value must be either True or False, not {value}")
NO_PROJECTION = gettext('Spatial file {filepath} has no projection')
BBOX_NOT_INTERSECT = gettext(
    'Not all of the spatial layers overlap each '
    'other. All bounding boxes must intersect: {bboxes}')
NEED_PERMISSION_DIRECTORY = gettext(
    'You must have {permission} access to this directory')
NEED_PERMISSION_FILE = gettext(
    'You must have {permission} access to this file')
WRONG_GEOM_TYPE = gettext('Geometry type must be one of {allowed}')

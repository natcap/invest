"""Script to generate shapefiles from InVEST logging database."""

import urllib.request
import datetime
import json


_ENDPOINTS_INDEX_URL = (
    'http://data.naturalcapitalproject.org/server_registry/'
    'invest_usage_logger_v2/index.html')


if __name__ == '__main__':
    USAGE_POLYGON_URL = json.loads(urllib.request.urlopen(
        _ENDPOINTS_INDEX_URL).read().strip())['STATS']


    OUT_FILENAME = 'invest_usage_%s.geojson' % (
        datetime.datetime.now().isoformat('_').replace(':', '_'))
    print('Writing usage to %s' % OUT_FILENAME)
    with open(OUT_FILENAME, 'w') as out_geojson:
        print('downloading run_summary vector')
        out_geojson.write(
            urllib.request.urlopen(urllib.request.Request(
                USAGE_POLYGON_URL)).read().decode('utf-8'))
    print('Done.')

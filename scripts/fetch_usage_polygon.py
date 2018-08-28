"""Script to generate shapefiles from InVEST logging database."""

import urllib
import urllib2
import datetime
import json


_ENDPOINTS_INDEX_URL = (
    'http://data.naturalcapitalproject.org/server_registry/'
    'invest_usage_logger_v2/index.html')


if __name__ == '__main__':
    USAGE_POLYGON_URL = json.loads(urllib.urlopen(
        _ENDPOINTS_INDEX_URL).read().strip())['STATS']


    OUT_FILENAME = 'invest_usage_%s.geojson' % (
        datetime.datetime.now().isoformat('_').replace(':', '_'))
    print 'Writing usage to %s' % OUT_FILENAME
    with open(OUT_FILENAME, 'w') as out_geojson:
        print 'downloading run_summary vector'
        out_geojson.write(
            urllib2.urlopen(urllib2.Request(
                USAGE_POLYGON_URL)).read())
    print 'Done.'

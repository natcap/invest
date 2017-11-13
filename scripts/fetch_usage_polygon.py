"""Script to generate shapefiles from InVEST logging database."""

import os
import urllib
import urllib2
import datetime


INVEST_USAGE_POLYGON_FUNC = (
    'https://us-central1-natcap-servers.cloudfunctions.net/'
    'function-invest-model-fetch-usage-polygon')


if __name__ == '__main__':
    OUT_FILENAME = 'invest_usage_%s.geojson' % (
        datetime.datetime.now().isoformat('_'))
    print 'Writing usage to %s' % OUT_FILENAME
    with open(OUT_FILENAME, 'w') as out_geojson:
        print 'downloading run_summary vector'
        out_geojson.write(
            urllib2.urlopen(urllib2.Request(
                INVEST_USAGE_POLYGON_FUNC)).read())
    print 'Done.'

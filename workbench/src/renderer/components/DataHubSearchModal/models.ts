/* Transformed search result metadata for use in React components. */
export interface DataHubSearchResult {
  id: string,
  title: string,
  description: string,
  pixelSize: number[],
  crsUnits: string,
  tags: string[],
  places: string[],
  collections: string[],
  license: string,
  author: string,
  lastUpdated: string,
  created: string,
  datasetUrl: string,
  webpageUrl: string,
}

/* Raw search result metadata returned by Data Hub Abstraction Layer. */
export interface DHALSearchResult {
  dataset_url: string,
  source_catalog_url: string,
  name: string,
  description: string,
  extent: number[],
  pixel_size: number[],
  crs_wkt: string,
  crs_units: string,
  tags: string[],
  places: string[],
  collection: string[],
  license: {
    id: string,
    title: string,
    url: string,
  },
  author: string,
  created: string,
  last_updated: string,
}

/**
 * Convert raw search result metadata (from DHAL)
 * into the format the React components expect.
 *
 * This translation layer affords the Workbench robustness to future DHAL API
 * changes, allowing us to make needed updates here rather than find & update
 * multiple references scattered across React components.
 */
export function transformDHALSearchResult(
  d: DHALSearchResult
): DataHubSearchResult {
  return {
    id: d.source_catalog_url,
    title: d.name,
    description: d.description,
    pixelSize: d.pixel_size,
    crsUnits: d.crs_units,
    tags: d.tags,
    places: d.places,
    collections: d.collection,
    license: d.license.title,
    author: d.author,
    lastUpdated: d.last_updated,
    created: d.created,
    datasetUrl: d.dataset_url,
    webpageUrl: d.source_catalog_url,
  }
}

// For now, LULC/Biophysical Table pairs are the only "sibling" pairs supported.
export enum DataHubSearchSiblingType {
  LULC,
  BIOPHYSICAL_TABLE,
  NONE,
}

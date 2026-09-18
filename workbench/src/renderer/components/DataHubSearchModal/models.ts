export interface DataHubSearchResult {
  id: string,
  title: string,
  description: string,
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

export interface DHALDataset {
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

export function transformDHALSearchResult(
  r: DHALDataset
): DataHubSearchResult {
  return {
    id: r.source_catalog_url,
    title: r.name,
    description: r.description,
    tags: r.tags,
    places: r.places,
    collections: r.collection,
    license: r.license.title,
    author: r.author,
    lastUpdated: r.last_updated,
    created: r.created,
    datasetUrl: r.dataset_url,
    webpageUrl: r.source_catalog_url,
  }
}

// For now, LULC/Biophysical Table pairs are the only "sibling" pairs supported.
export enum DataHubSearchSiblingType {
  LULC,
  BIOPHYSICAL_TABLE,
  NONE,
}

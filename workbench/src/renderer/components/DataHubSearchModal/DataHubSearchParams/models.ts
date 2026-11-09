export interface DataHubSearchQuery {
  tags: string[],
  index_tags: string[],
  datatype: string,
  extent: number[], // [xmin, ymin, xmax, ymax]
  collections: string[],
}

export class DHALSearchParams extends URLSearchParams {
  constructor(workbenchQuery: DataHubSearchQuery) {
    super();
    for (let tag of workbenchQuery.tags) {
      this.append('tags', tag);
    }
    for (let tag of workbenchQuery.indexTags) {
      this.append('index_tags', tag);
    }
    this.append('datatype', workbenchQuery.datatype === 'csv' ? 'table' : workbenchQuery.datatype);
    for (let coord of workbenchQuery.extent) {
      this.append('extent', coord.toString());
    }
    for (let collection of workbenchQuery.collections) {
      this.append('sibling', collection);
    }
  }
}

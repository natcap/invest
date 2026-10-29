export interface DataHubSearchQuery {
  tags: string[],
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
    this.append('datatype', workbenchQuery.datatype === 'csv' ? 'table' : workbenchQuery.datatype);
    for (let coord of workbenchQuery.extent) {
      this.append('extent', coord.toString());
    }
    for (let collection of workbenchQuery.collections) {
      this.append('sibling', collection);
    }
  }
}

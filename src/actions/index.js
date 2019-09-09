export const getCsvUrl = csvUrl => ({
  type: 'CSV_UPLOADED',
  payload: csvUrl
});

export const getVectorsOnMap = vectorsOnMap => ({
  type: 'VECTORS_UPDATED',
  payload: vectorsOnMap
});

export const getFileSuffix = fileSuffix => ({
  type: 'SUFFIX_OBTAINED',
  payload: fileSuffix
});
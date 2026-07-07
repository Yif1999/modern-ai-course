export function datasetSampleKey(sample, index) {
  return `${index}-${sample?.id ?? "na"}-${sample?.category ?? "unknown"}-${sample?.char_count ?? 0}`;
}

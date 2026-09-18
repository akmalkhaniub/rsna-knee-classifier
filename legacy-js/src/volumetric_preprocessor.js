/**
 * VolumetricPreprocessor - DICOM MRI Volume Normalization
 * Standardizes variable slice depths into uniform 32-slice tensors with percentile normalization.
 */

export class VolumetricPreprocessor {
  /**
   * Interpolate variable-length slice series into standardized target depth
   * @param {Array<number[][]>} slices 
   * @param {number} targetDepth (default 32 slices)
   * @returns {Array<number[][]>}
   */
  static standardizeDepth(slices, targetDepth = 32) {
    const currentDepth = slices.length;
    if (currentDepth === targetDepth) return slices;

    const resampled = [];
    const step = (currentDepth - 1) / (targetDepth - 1);

    for (let i = 0; i < targetDepth; i++) {
      const srcIdx = Math.round(i * step);
      const clampedIdx = Math.min(Math.max(srcIdx, 0), currentDepth - 1);
      resampled.push(slices[clampedIdx]);
    }

    return resampled;
  }

  /**
   * Apply intensity percentile clipping and z-score scaling to voxel arrays
   * @param {number[]} voxels 
   * @returns {number[]} normalized voxel array
   */
  static normalizeVoxelIntensity(voxels) {
    const sorted = [...voxels].sort((a, b) => a - b);
    const p1Idx = Math.floor(sorted.length * 0.01);
    const p99Idx = Math.floor(sorted.length * 0.99);
    const minVal = sorted[p1Idx];
    const maxVal = sorted[p99Idx];

    const range = (maxVal - minVal) || 1;
    return voxels.map(v => {
      const clipped = Math.min(Math.max(v, minVal), maxVal);
      return Number(((clipped - minVal) / range).toFixed(4));
    });
  }
}

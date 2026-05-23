/** Downsample captured frames to match lite model training (8x32x32). */
export function downsampleForModel(
  planes: number[][][],
  targetT = 8,
  targetH = 32,
  targetW = 32
): Float32Array {
  const tIn = planes.length;
  const hw = planes[0][0].length;
  const hIn = Math.round(Math.sqrt(hw));
  const wIn = hIn;

  const out = new Float32Array(3 * targetT * targetH * targetW);
  let idx = 0;
  for (let c = 0; c < 3; c++) {
    for (let ti = 0; ti < targetT; ti++) {
      const fi = Math.min(tIn - 1, Math.floor((ti / targetT) * tIn));
      const plane = planes[fi][c];
      for (let y = 0; y < targetH; y++) {
        for (let x = 0; x < targetW; x++) {
          const sy = Math.min(hIn - 1, Math.floor((y / targetH) * hIn));
          const sx = Math.min(wIn - 1, Math.floor((x / targetW) * wIn));
          out[idx++] = plane[sy * wIn + sx] ?? 0;
        }
      }
    }
  }
  return out;
}

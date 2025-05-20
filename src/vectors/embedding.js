import { getPipeline } from '../transformers.js';
const TASK = 'feature-extraction';

/**
 * Gets the vectorized text in form of an array of numbers.
 * @param {string} text - The text to vectorize
 * @returns {Promise<number[]>} - The vectorized text in form of an array of numbers
 */
export async function getTransformersVector(text) {
    const pipe = await getPipeline(TASK);
    const result = await pipe(text, { pooling: 'mean', normalize: true });
    const vector = Array.from(result.data);
    return vector;
}

/**
 * Gets the vectorized texts in form of an array of arrays of numbers.
 * @param {string[]} texts - The texts to vectorize
 * @returns {Promise<number[][]>} - The vectorized texts in form of an array of arrays of numbers
 */
export async function getTransformersBatchVector(texts) {
    const pipe = await getPipeline(TASK);
    const result = await pipe(texts, { pooling: 'mean', normalize: true });
    // The result.data is a Float32Array for the entire batch.
    // If the model outputs a vector of size N for each text, and we pass M texts,
    // then result.data will have M * N elements.
    // result.dims will be [M, N]
    const batchSize = result.dims[0];
    const vectorSize = result.dims[1];
    const output = [];
    for (let i = 0; i < batchSize; ++i) {
        const start = i * vectorSize;
        const end = start + vectorSize;
        output.push(Array.from(result.data.slice(start, end)));
    }
    return output;
}

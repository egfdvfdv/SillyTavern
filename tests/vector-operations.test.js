import { jest } from '@jest/globals';
import request from 'supertest';
import express from 'express';
import * as fs from 'fs/promises';
import * as path from 'path';
import { fileURLToPath } from 'url'; // Added for ESM __dirname equivalent
// Note: LocalIndex is imported here, it does not depend on the mocked module.
import { LocalIndex } from 'vectra'; 

// ESM equivalent for __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Mocking user directory for tests
const TEST_USER_DATA_DIR = path.join(__dirname, 'test-user-data');
const TEST_GENERAL_INDEXES_DIR = path.join(TEST_USER_DATA_DIR, 'indexes'); 
const TEST_VECTORS_DIR_IN_USER_DATA = path.join(TEST_USER_DATA_DIR, 'vectors');

// Test collection details
const TEST_COLLECTION_ID = 'test-transformers-collection';
const TEST_SOURCE = 'transformers'; 
const sampleDocuments = [
    { hash: 101, text: 'The quick brown fox jumps over the lazy dog.', index: 0 },
    { hash: 102, text: 'Transformers are powerful models for NLP.', index: 1 },
    { hash: 103, text: 'Jest is a delightful JavaScript Testing Framework.', index: 2 },
    { hash: 104, text: 'Batch processing can significantly improve efficiency.', index: 3 },
    { hash: 105, text: 'A test document about vector similarity searches.', index: 4 },
];

// --- JEST MOCKING SETUP ---

const createDeterministicEmbedding = (text, vectorSize = 384) => {
    const embedding = new Float32Array(vectorSize);
    let hash = 0;
    for (let k = 0; k < text.length; k++) {
        hash = (hash + text.charCodeAt(k) * (k+1)) % 2000; // Slightly more complex hash
    }
    for (let j = 0; j < vectorSize; j++) {
        embedding[j] = (((hash + j * 10) % 1000) / 500.0) - 1.0; // Values between -1.0 and 1.0
    }
    // Normalize the mock embedding
    let norm = 0;
    for(let val of embedding) norm += val * val;
    norm = Math.sqrt(norm);
    if (norm > 0) {
        for (let j = 0; j < vectorSize; j++) embedding[j] /= norm;
    } else { // Handle zero vector case, though unlikely with this hash
        embedding[0] = 1.0; // Prevent zero vector
    }
    return embedding;
};

// Mock '../src/transformers.js' BEFORE importing modules that depend on it.
jest.unstable_mockModule('../src/transformers.js', () => ({
    __esModule: true, // Indicate that this is an ES Module mock
    getPipeline: jest.fn().mockImplementation(async (task, modelName) => {
        if (task === 'feature-extraction') {
            return jest.fn().mockImplementation(async (textOrTexts, options) => {
                const isBatch = Array.isArray(textOrTexts);
                const vectorSize = 384;

                if (isBatch) {
                    const embeddings = new Float32Array(textOrTexts.length * vectorSize);
                    textOrTexts.forEach((text, i) => {
                        const singleEmbedding = createDeterministicEmbedding(text, vectorSize);
                        embeddings.set(singleEmbedding, i * vectorSize);
                    });
                    return { data: embeddings, dims: [textOrTexts.length, vectorSize] };
                } else {
                    const singleEmbedding = createDeterministicEmbedding(textOrTexts, vectorSize);
                    return { data: singleEmbedding, dims: [1, vectorSize] };
                }
            });
        }
        throw new Error(`Mocked getPipeline: Unhandled task ${task}`);
    }),
}));

// --- DYNAMIC IMPORTS ---
// Must wait for mocks to be set up before importing modules that use them.
let vectorRouter;

beforeAll(async () => {
    // Dynamically import the router after mocks are established
    const SUT = await import('../src/endpoints/vectors.js');
    vectorRouter = SUT.router;

    // Setup Express app after router is imported
    app.use(express.json());
    // Middleware to mock req.user and app.get
    app.use((req, res, next) => {
        req.user = {
            name: 'test-user',
            directories: {
                userData: TEST_USER_DATA_DIR,
                vectors: TEST_VECTORS_DIR_IN_USER_DATA, 
            },
        };
        req.app = {
            get: jest.fn((key) => {
                if (key === 'ensureUserIndexesDirectory') { 
                    return async () => {
                        await fs.mkdir(TEST_GENERAL_INDEXES_DIR, { recursive: true });
                        return TEST_GENERAL_INDEXES_DIR;
                    };
                }
                return undefined;
            }),
        };
        next();
    });
    app.use('/vectors', vectorRouter);

    // Create physical directories
    await fs.mkdir(TEST_VECTORS_DIR_IN_USER_DATA, { recursive: true });
    await fs.mkdir(TEST_GENERAL_INDEXES_DIR, { recursive: true });
});


const app = express(); // Define app here, configure in beforeAll

describe('Vector Operations API (transformers)', () => {
    // beforeAll is now primarily for async setup like imports and dir creation
    // app configuration also moved to beforeAll to use the dynamically imported router

    afterAll(async () => {
        await fs.rm(TEST_USER_DATA_DIR, { recursive: true, force: true });
    });

    const getVectraCollectionPath = (collectionId, source) => {
        const modelScope = ''; 
        return path.join(TEST_VECTORS_DIR_IN_USER_DATA, source, collectionId, modelScope);
    };

    test('Initial state: Listing a new collection should return empty', async () => {
        const specificCollectionId = `${TEST_COLLECTION_ID}-initial`;
        const response = await request(app)
            .post('/vectors/list')
            .send({ collectionId: specificCollectionId, source: TEST_SOURCE });

        expect(response.status).toBe(200);
        expect(response.body).toEqual([]); 

        const collectionPath = getVectraCollectionPath(specificCollectionId, TEST_SOURCE);
        const index = new LocalIndex(collectionPath);
        expect(await index.isIndexCreated()).toBe(true);
        const items = await index.listItems();
        expect(items).toHaveLength(0);
    });

    test('Indexing: should insert items and allow listing them', async () => {
        const insertResponse = await request(app)
            .post('/vectors/insert')
            .send({
                collectionId: TEST_COLLECTION_ID,
                source: TEST_SOURCE,
                items: sampleDocuments,
            });

        expect(insertResponse.status).toBe(200);

        const listResponse = await request(app)
            .post('/vectors/list')
            .send({ collectionId: TEST_COLLECTION_ID, source: TEST_SOURCE });

        expect(listResponse.status).toBe(200);
        expect(listResponse.body).toBeInstanceOf(Array);
        expect(listResponse.body).toHaveLength(sampleDocuments.length);
        
        const listedHashes = listResponse.body.map(Number).sort((a,b) => a - b);
        const expectedHashes = sampleDocuments.map(doc => doc.hash).sort((a,b) => a - b);
        expect(listedHashes).toEqual(expectedHashes);
    });

    test('Querying: should retrieve relevant documents', async () => {
        const queryText = sampleDocuments[1].text;
        const response = await request(app)
            .post('/vectors/query')
            .send({
                collectionId: TEST_COLLECTION_ID,
                source: TEST_SOURCE,
                searchText: queryText,
                topK: 3,
                threshold: 0.1, 
            });

        expect(response.status).toBe(200);
        expect(response.body.metadata).toBeDefined();
        expect(response.body.hashes).toBeDefined();
        expect(response.body.metadata.length).toBeGreaterThan(0);
        expect(response.body.metadata.length).toBeLessThanOrEqual(3);

        const topResultMetadata = response.body.metadata[0];
        expect(topResultMetadata.text).toEqual(queryText);
        expect(topResultMetadata.hash).toEqual(sampleDocuments[1].hash);
    }, 15000);

    test('Cleanup: should purge the collection', async () => {
        const purgeResponse = await request(app)
            .post('/vectors/purge')
            .send({ collectionId: TEST_COLLECTION_ID });

        expect(purgeResponse.status).toBe(200);

        const collectionPath = getVectraCollectionPath(TEST_COLLECTION_ID, TEST_SOURCE);
        try {
            await fs.access(collectionPath);
            const dirExists = await fs.stat(collectionPath).then(() => true).catch(() => false);
            expect(dirExists).toBe(false); 
        } catch (error) {
            expect(error.code).toBe('ENOENT');
        }
        
        const listResponse = await request(app)
            .post('/vectors/list')
            .send({ collectionId: TEST_COLLECTION_ID, source: TEST_SOURCE });
        expect(listResponse.status).toBe(200);
        expect(listResponse.body).toEqual([]);
    });
});

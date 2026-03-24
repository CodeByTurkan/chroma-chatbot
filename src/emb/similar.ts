import { createEmbeddings, loadJson } from "./index.ts";
import type { EmbeddingsData } from "./index.ts";

function dotProduct(a: number[], b: number[]) {
  return a.map((value, index) => value * b[index]).reduce((a, b) => a + b, 0);
}

//formula
function cosineSimilarity(a: number[], b: number[]) {
  const product = dotProduct(a, b);
  const aMagnitude = Math.sqrt(
    a.map((value, index) => value * b[index]).reduce((a, b) => a + b, 0),
  );
  const bMagnitude = Math.sqrt(
    b.map((value, index) => value * b[index]).reduce((a, b) => a + b, 0),
  );
  return product / (aMagnitude + bMagnitude);
}

async function createSimilarity() {
  const dataEmbedding = loadJson<EmbeddingsData[]>("dataEmbedding2.json");
  const input = "What is machine learning?";
  const inputEmbeddings = await createEmbeddings(input);

  const similarities: {
    input: string;
    similarity: number;
  }[] = [];

  for (const entry of dataEmbedding) {
    const similarity = dotProduct(
      entry.embedding,
      inputEmbeddings.data[0].embedding,
    );

    similarities.push({
      input: entry.input,
      similarity,
    });
  }
  console.log(`Similarity of ${input} with:`);
  const sortedSimilarities = similarities.sort(
    (a, b) => b.similarity - a.similarity,
  );
  sortedSimilarities.forEach((similarity) => {
    console.log(`${similarity.input} : ${similarity.similarity}`);
  });
}

createSimilarity();

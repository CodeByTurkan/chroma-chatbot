import { readFileSync, writeFileSync } from "fs";
import { OpenAI } from "openai";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const client = new OpenAI();
export type EmbeddingsData = {
  input: string;
  embedding: number[];
};
export async function createEmbeddings(input: string | string[]) {
  const response = await client.embeddings.create({
    input: input,
    model: "text-embedding-3-small",
  });
  // console.log(response.data[0].embedding);

  return response;
}

// createEmbeddings(["cat is an animal ", "dog is also an animal"]);

export function loadJson<T>(filename: string): T {
  const path = join(__dirname, filename);
  const rawData = readFileSync(path);
  return JSON.parse(rawData.toString());
}

function saveDataToJsonFile(data: any, filename: string) {
  const dataString = JSON.stringify(data, null, 4);
  const dataObject = Buffer.from(dataString);
  const path = join(__dirname, filename);

  writeFileSync(path, dataObject);
  console.log(`Save data to ${filename}`);
}

async function createJsonFile() {
  const data = loadJson<string[]>("index2.json");
  const embeddings = await createEmbeddings(data);
  const dataEmbedding: EmbeddingsData[] = [];
  for (let i = 0; i < data.length; i++) {
    dataEmbedding.push({
      input: data[i],
      embedding: embeddings.data[i].embedding,
    });
  }
  saveDataToJsonFile(dataEmbedding, "dataEmbedding2.json");
}

createJsonFile();

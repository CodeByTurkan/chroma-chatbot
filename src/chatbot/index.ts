import { OpenAIEmbeddingFunction } from "@chroma-core/openai";
import { ChromaClient } from "chromadb";
import OpenAI from "openai";

const chromaClient = new ChromaClient({
  host: "localhost",
  port: 8000,
});

const studentInfo = `Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,
is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking
in her free time in hopes of working at a tech company after graduating from the University of Washington.`;

const clubInfo = `The university chess club provides an outlet for students to come together and enjoy playing
the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning
the rules to experienced tournament players. The club typically meets a few times per week to play casual games,
participate in tournaments, analyze famous chess matches, and improve members' skills.`;

const universityInfo = `The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world.`;

const apiKey = process.env.OPENAI_API_KEY;
const embeddingFunction = new OpenAIEmbeddingFunction({
  modelName: "text-embedding-3-small",
  apiKey: apiKey,
});

const collectionName = "personal-infos";

async function getAndCreateCollection() {
  return chromaClient.getOrCreateCollection({
    name: collectionName,
    embeddingFunction,
  });
}

async function populateCollection() {
  const collection = await getAndCreateCollection();
  await collection.upsert({
    ids: ["id1", "id2", "id3"],
    documents: [studentInfo, clubInfo, universityInfo],
  });
}

async function askQuestion() {
  const question = "what is Alexandra fatehr";
  const collection = await getAndCreateCollection();
  const result = await collection.query({
    queryTexts: [question],
    nResults: 1,
  });
  const relevantInfo = result.documents[0]?.[0];
  if (!relevantInfo) {
    console.log("No matching context in Chroma.");
    return;
  }

  const client = new OpenAI({ apiKey });
  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    temperature: 0,
    messages: [
      {
        role: "system",
        content: `Answer using only this context. If the answer is not in the context, say you don't know.\n\nContext:\n${relevantInfo}`,
      },
      { role: "user", content: question },
    ],
  });
  const text = response.choices[0]?.message?.content;
  console.log(text ?? "(empty response)");
}

async function main() {
  await populateCollection();
  await askQuestion();
}
main();

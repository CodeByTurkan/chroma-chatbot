import { OpenAI } from "openai";
const client = new OpenAI();
async function createEmbeddings(input) {
    const response = await client.embeddings.create({
        input: input,
        model: "text-embedding-3-small",
    });
    console.log(response.data[0].embedding);
    return response;
}
createEmbeddings(["cat is an animal ", "dog is also an animal"]);

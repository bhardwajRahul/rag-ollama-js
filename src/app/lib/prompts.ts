import { PromptTemplate } from "@langchain/core/prompts";

export const standaloneTemplate = PromptTemplate.fromTemplate(
    `Given some conversation history and a question, convert the question to a standalone question. 
    conversation history: {history}
    question: {question} 
    standalone question:`
);

export const multiQueryTemplate = PromptTemplate.fromTemplate(
    `You are helping retrieve documents from a vector database. Generate 4 alternative phrasings of the user's question, each capturing the same intent from a different angle (different vocabulary, specificity, or perspective). Use the conversation history to resolve pronouns and references so every phrasing stands alone.
    Write one phrasing per line. No numbering, no bullets, no commentary — just the 4 lines.
    conversation history: {history}
    question: {question}
    alternative phrasings:`
);

export const answerTemplate = PromptTemplate.fromTemplate(`You are a helpful and enthusiastic support bot who answers questions based on the provided context.
The context is a list of numbered excerpts, each labeled "[Source N | Page P]" followed by its text.
Your goal is to find the most relevant information from the context to answer the question.

- Cite the excerpts you relied on inline, right after the relevant sentence, using their bracket label exactly as given, e.g. "[Source 2]".
- If you don't know the answer or cannot find it in the context, say, "I don't know," and do not fabricate an answer or a citation.
- Always respond in a friendly and conversational tone.

Context:
{context}

Question:
{question}

Answer:`);
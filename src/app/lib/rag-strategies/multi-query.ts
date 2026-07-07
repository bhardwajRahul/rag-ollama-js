import { RunnableSequence, RunnableLambda } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { answerTemplate, multiQueryTemplate } from "../prompts";
import { llm } from "../ollama";
import { retrieveManyAndBuildContext } from "./retrieval";
import type { RagStrategy } from "./types";

// Named so the chat route can pick this LLM's token-stream events out of `.streamEvents()`
// without also picking up the rephrasing generation below.
const answerLLM = llm.withConfig({ runName: "answerLLM" });

// The LLM is asked for one phrasing per line with no numbering, but small models add
// bullets/numbers anyway — strip them rather than let them pollute the embedding.
function parsePhrasings(raw: string): string[] {
    return raw
        .split("\n")
        .map((line) => line.replace(/^[-*\d.)\s]+/, "").trim())
        .filter(Boolean)
        .slice(0, 5);
}

const rephraseChain = RunnableSequence.from([
    multiQueryTemplate,
    llm,
    new StringOutputParser(),
    RunnableLambda.from(parsePhrasings),
]);

const answerChain = RunnableSequence.from([
    answerTemplate,
    answerLLM,
    new StringOutputParser()
]);

// Multi-query RAG: one phrasing of a question can miss the right chunk simply because its
// vocabulary doesn't match the document's. Generate several rephrasings, retrieve for each
// in parallel (original question included as a safety net), and answer from the deduped
// union — hedging the embedding lottery at the cost of one extra LLM call.
export const multiQueryChain: RagStrategy = (filter) => RunnableSequence.from([
    {
        phrasings: rephraseChain,
        question: ({ question }) => question,
        history: ({ history }) => history,
    },
    {
        retrieved: RunnableSequence.from([
            ({ question, phrasings }) => [question, ...phrasings],
            retrieveManyAndBuildContext(filter),
        ]),
        question: ({ question }) => question,
        history: ({ history }) => history,
    },
    {
        context: ({ retrieved }) => retrieved.context,
        question: ({ question }) => question,
        history: ({ history }) => history,
    },
    answerChain
]);

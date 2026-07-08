import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { answerTemplate } from "../prompts";
import { llm } from "../ollama";
import { hybridRetrieveAndBuildContext } from "./retrieval";
import type { RagStrategy } from "./types";

// Named so the chat route can pick this LLM's token-stream events out of `.streamEvents()`.
const answerLLM = llm.withConfig({ runName: "answerLLM" });

const answerChain = RunnableSequence.from([
    answerTemplate,
    answerLLM,
    new StringOutputParser()
]);

// Hybrid search: cosine similarity alone misses exact keyword/acronym/identifier matches
// ("RRF", "SKU-1042") that full-text search nails, and full-text alone misses paraphrases
// that embeddings nail. The hybrid_search RPC runs both and fuses the rankings with
// reciprocal rank fusion. The raw question is embedded AND keyword-matched unmodified —
// no rewriting — so exact user terms reach the keyword side intact, and comparisons
// against `naive` isolate the retrieval change alone.
export const hybridChain: RagStrategy = (filter) => RunnableSequence.from([
    {
        retrieved: RunnableSequence.from([
            ({ question }) => question,
            hybridRetrieveAndBuildContext(filter)
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

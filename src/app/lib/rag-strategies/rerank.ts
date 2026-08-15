import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { answerTemplate } from "../prompts";
import { llm } from "../ollama";
import { rerankRetrieveAndBuildContext } from "./retrieval";
import type { RagStrategy } from "./types";

// Named so the chat route can pick this LLM's token-stream events out of `.streamEvents()`
// without also picking up the per-chunk relevance-scoring calls made during retrieval.
const answerLLM = llm.withConfig({ runName: "answerLLM" });

const answerChain = RunnableSequence.from([
    answerTemplate,
    answerLLM,
    new StringOutputParser()
]);

// Re-ranking: retrieves on the unmodified question (isolating the change against `naive`),
// but through rerankRetrieveAndBuildContext, which over-fetches by vector similarity and
// has the LLM re-score/re-order the candidates before the top 4 reach the answer stage.
export const rerankChain: RagStrategy = (filter) => RunnableSequence.from([
    {
        retrieved: RunnableSequence.from([
            ({ question }) => question,
            rerankRetrieveAndBuildContext(filter)
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

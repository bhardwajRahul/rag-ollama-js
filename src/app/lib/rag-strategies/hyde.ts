import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { answerTemplate, hydeTemplate } from "../prompts";
import { llm } from "../ollama";
import { retrieveAndBuildContext } from "./retrieval";
import type { RagStrategy } from "./types";

// Named so the chat route can pick this LLM's token-stream events out of `.streamEvents()`
// without also picking up the hypothetical-passage draft below.
const answerLLM = llm.withConfig({ runName: "answerLLM" });

const hypotheticalChain = RunnableSequence.from([
    hydeTemplate,
    llm,
    new StringOutputParser()
]);

const retrieverChain = (filter: Record<string, unknown>) => RunnableSequence.from([
    (result: { hypotheticalAnswer: string }) => result.hypotheticalAnswer,
    retrieveAndBuildContext(filter)
]);

const answerChain = RunnableSequence.from([
    answerTemplate,
    answerLLM,
    new StringOutputParser()
]);

// HyDE (Hypothetical Document Embeddings): questions and answers live in different regions
// of embedding space — documents sound like answers, not questions. Draft a plausible (even
// factually wrong) answer passage and embed that instead of the question; wrong facts in the
// right vocabulary land closer to the real chunks than the question itself does. The real
// answer is still generated from retrieved context only — the draft never reaches the user.
export const hydeChain: RagStrategy = (filter) => RunnableSequence.from([
    {
        hypotheticalAnswer: hypotheticalChain,
        originalQuestion: new RunnablePassthrough()
    },
    {
        retrieved: retrieverChain(filter),
        question: ({ originalQuestion }) => originalQuestion.question,
        history: ({ originalQuestion }) => originalQuestion.history,
    },
    {
        context: ({ retrieved }) => retrieved.context,
        question: ({ question }) => question,
        history: ({ history }) => history,
    },
    answerChain
]);

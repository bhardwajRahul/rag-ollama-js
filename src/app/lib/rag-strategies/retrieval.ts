import { RunnableSequence, RunnableLambda } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import type { DocumentInterface } from "@langchain/core/documents";

import { retriever, hybridSearcher } from "../supabase";
import { buildContext } from "../../utils/helpers";
import { llm } from "../ollama";
import { rerankTemplate } from "../prompts";

// Retrieval piped into citation-context building — composed, not hand-orchestrated with
// async/await — as one named runnable step shared by every strategy, so the chat route can
// read its `sources` output via `.streamEvents()` as soon as it finishes, ahead of the answer
// tokens that follow later in the same chain.
export const retrieveAndBuildContext = (filter: Record<string, unknown>) => RunnableSequence.from([
    retriever(filter),
    RunnableLambda.from(buildContext),
]).withConfig({ runName: "retrieveAndBuildContext" });

// The same chunk often matches several phrasings of one question, so a multi-query union
// must dedupe before numbering sources — otherwise the LLM would see (and cite) the same
// text under two different [Source N] labels.
function dedupeDocuments(docGroups: DocumentInterface[][]): DocumentInterface[] {
    const seen = new Set<string>();
    return docGroups.flat().filter((doc) => {
        if (seen.has(doc.pageContent)) return false;
        seen.add(doc.pageContent);
        return true;
    });
}

// Multi-query variant: takes a list of query phrasings, retrieves for each concurrently
// (`.batch()` fans the retriever out over the list), unions and dedupes the chunks, then
// builds citation context. Carries the same runName as the single-query version so the
// chat route's `sources` extraction works identically for every strategy.
export const retrieveManyAndBuildContext = (filter: Record<string, unknown>) => RunnableSequence.from([
    RunnableLambda.from((queries: string[]) => retriever(filter).batch(queries)),
    RunnableLambda.from(dedupeDocuments),
    RunnableLambda.from(buildContext),
]).withConfig({ runName: "retrieveAndBuildContext" });

// Hybrid variant: keyword + vector search fused in-database by the hybrid_search RPC
// (see supabase.ts / supabaseScripts.txt STEP 9). Same runName convention as above.
export const hybridRetrieveAndBuildContext = (filter: Record<string, unknown>) => RunnableSequence.from([
    RunnableLambda.from(hybridSearcher(filter)),
    RunnableLambda.from(buildContext),
]).withConfig({ runName: "retrieveAndBuildContext" });

const RERANK_FETCH_COUNT = 20;
const RERANK_KEEP_COUNT = 4;

const rerankScoreChain = RunnableSequence.from([
    rerankTemplate,
    llm,
    new StringOutputParser(),
]);

function clamp01(value: number): number {
    if (!Number.isFinite(value)) return 0;
    return Math.min(1, Math.max(0, value));
}

// Ollama has no cross-encoder model, so "re-ranking" here means asking the chat LLM to
// pointwise-score each candidate's relevance (0-1) against the question, one call per chunk.
async function scoreRelevance(question: string, doc: DocumentInterface): Promise<number> {
    const raw = await rerankScoreChain.invoke({ question, passage: doc.pageContent });
    return clamp01(parseFloat(raw));
}

async function rerank(question: string, docs: DocumentInterface[]): Promise<DocumentInterface[]> {
    const scores = await Promise.all(docs.map((doc) => scoreRelevance(question, doc)));
    return docs
        .map((doc, index) => ({ doc, score: scores[index] }))
        .sort((a, b) => b.score - a.score)
        .slice(0, RERANK_KEEP_COUNT)
        .map(({ doc }) => doc);
}

// Re-ranking: vector similarity is a cheap proxy for relevance and sometimes wrong — it can
// rank a tangentially-related chunk above one that actually answers the question. Over-fetch
// a wider candidate pool (20) by cosine similarity, have the LLM score each one directly
// against the question, and keep only the top 4 by that score.
export const rerankRetrieveAndBuildContext = (filter: Record<string, unknown>) => RunnableSequence.from([
    RunnableLambda.from(async (question: string) => {
        const candidates = await retriever(filter, RERANK_FETCH_COUNT).invoke(question);
        return rerank(question, candidates);
    }),
    RunnableLambda.from(buildContext),
]).withConfig({ runName: "retrieveAndBuildContext" });

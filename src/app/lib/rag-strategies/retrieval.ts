import { RunnableSequence, RunnableLambda, RunnablePassthrough } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import type { DocumentInterface } from "@langchain/core/documents";

import { retriever, hybridSearcher } from "../supabase";
import { buildContext } from "../../utils/helpers";
import { llm } from "../ollama";
import { rerankTemplate, compressionTemplate } from "../prompts";

// Retrieval piped into citation-context building — composed, not hand-orchestrated with
// async/await — as one named runnable step shared by every strategy, so the chat route can
// read its `sources` output via `.streamEvents()` as soon as it finishes, ahead of the answer
// tokens that follow later in the same chain.
export const retrieveAndBuildContext = (filter: Record<string, unknown>) => RunnableSequence.from([
    retriever(filter).withConfig({ runName: "vectorRetrieve" }),
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
    RunnableLambda.from((queries: string[]) => retriever(filter).batch(queries)).withConfig({ runName: "vectorRetrieveMany" }),
    RunnableLambda.from(dedupeDocuments).withConfig({ runName: "dedupeCandidates" }),
    RunnableLambda.from(buildContext),
]).withConfig({ runName: "retrieveAndBuildContext" });

// Hybrid variant: keyword + vector search fused in-database by the hybrid_search RPC
// (see supabase.ts / supabaseScripts.txt STEP 9). Same runName convention as above.
export const hybridRetrieveAndBuildContext = (filter: Record<string, unknown>) => RunnableSequence.from([
    RunnableLambda.from(hybridSearcher(filter)).withConfig({ runName: "hybridSearch" }),
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
// These 20 calls run outside the outer chain's traced execution (a bare .invoke() with no
// config/callbacks threaded through), so they never reach `chain.streamEvents()` in route.ts
// no matter what runName they're given — the prompt/completion has to be captured here
// directly and carried out as data, not surfaced via the streaming instrumentation.
async function scoreRelevance(question: string, doc: DocumentInterface): Promise<{ score: number; prompt: string; completion: string }> {
    const prompt = await rerankTemplate.format({ question, passage: doc.pageContent });
    const completion = await rerankScoreChain.invoke({ question, passage: doc.pageContent });
    return { score: clamp01(parseFloat(completion)), prompt, completion };
}

export interface ScoredCandidate {
    snippet: string;
    pageNumber: number;
    score: number;
    kept: boolean;
    prompt: string;
    completion: string;
}

// Scores every candidate and returns the full ranked table (so the pipeline visualizer can show
// which candidates were dropped, not just the ones that survived, plus each one's raw LLM
// prompt/response) alongside the kept documents.
async function scoreCandidates(question: string, docs: DocumentInterface[]): Promise<{ table: ScoredCandidate[]; kept: DocumentInterface[] }> {
    const scored = await Promise.all(docs.map((doc) => scoreRelevance(question, doc)));
    const ranked = docs
        .map((doc, index) => ({ doc, ...scored[index] }))
        .sort((a, b) => b.score - a.score);
    const kept = ranked.slice(0, RERANK_KEEP_COUNT).map(({ doc }) => doc);
    const keptSet = new Set(kept);
    const table = ranked.map(({ doc, score, prompt, completion }) => ({
        snippet: doc.pageContent.slice(0, 160),
        pageNumber: doc.metadata?.pageNumber ?? 0,
        score,
        kept: keptSet.has(doc),
        prompt,
        completion,
    }));
    return { table, kept };
}

// Re-ranking: vector similarity is a cheap proxy for relevance and sometimes wrong — it can
// rank a tangentially-related chunk above one that actually answers the question. Over-fetch
// a wider candidate pool (20) by cosine similarity, have the LLM score each one directly
// against the question, and keep only the top 4 by that score. Split into two named steps
// (rather than one hand-rolled async function) so the chat route can surface the over-fetch
// count and the full scored table — including the dropped candidates — via streamEvents().
export const rerankRetrieveAndBuildContext = (filter: Record<string, unknown>) => RunnableSequence.from([
    RunnableLambda.from(async (question: string) => {
        const candidates = await retriever(filter, RERANK_FETCH_COUNT).invoke(question);
        return { question, candidates };
    }).withConfig({ runName: "overFetchCandidates" }),
    RunnableLambda.from(({ question, candidates }: { question: string; candidates: DocumentInterface[] }) =>
        scoreCandidates(question, candidates)
    ).withConfig({ runName: "scoreCandidates" }),
    ({ kept }: { table: ScoredCandidate[]; kept: DocumentInterface[] }) => kept,
    RunnableLambda.from(buildContext),
]).withConfig({ runName: "retrieveAndBuildContext" });

const compressionChain = RunnableSequence.from([
    compressionTemplate,
    llm,
    new StringOutputParser(),
]);

interface CompressibleDoc {
    pageContent: string;
    metadata?: { pageNumber?: number };
}

export interface CompressedChunk {
    pageNumber: number;
    original: string;
    compressed: string;
    reduced: boolean;
    prompt: string;
    completion: string;
}

// Same limitation as scoreRelevance above: a bare .invoke() outside the parent chain's traced
// execution, so the prompt/completion is captured here directly rather than via streamEvents().
async function compressChunk(question: string, doc: DocumentInterface): Promise<{ doc: CompressibleDoc; row: CompressedChunk }> {
    const prompt = await compressionTemplate.format({ question, passage: doc.pageContent });
    const completion = await compressionChain.invoke({ question, passage: doc.pageContent });
    const trimmed = completion.trim();
    // Never let a chunk collapse to nothing — if the LLM strips every sentence, fall back to
    // the original rather than handing the answer step an empty, uncited source.
    const keptContent = trimmed || doc.pageContent;
    return {
        doc: { pageContent: keptContent, metadata: doc.metadata },
        row: {
            pageNumber: doc.metadata?.pageNumber ?? 0,
            original: doc.pageContent.slice(0, 200),
            compressed: keptContent.slice(0, 200),
            reduced: trimmed.length > 0 && trimmed.length < doc.pageContent.length,
            prompt,
            completion,
        },
    };
}

async function compressCandidates(question: string, docs: DocumentInterface[]): Promise<{ table: CompressedChunk[]; compressed: CompressibleDoc[] }> {
    const results = await Promise.all(docs.map((doc) => compressChunk(question, doc)));
    return {
        table: results.map((result) => result.row),
        compressed: results.map((result) => result.doc),
    };
}

// Contextual compression: a retrieved chunk is often a mix of relevant and irrelevant
// sentences — the irrelevant ones are just noise the answer LLM has to read past (and can
// distract it into citing text that doesn't actually support the answer). Strip each chunk
// down to only the sentences relevant to the question before it reaches the answer prompt.
// The first step is a parallel branch (retrieve the docs, passthrough the question) rather than
// a custom lambda like overFetchCandidates' — that keeps its input/output shape identical to
// the plain retriever step other modes name "vectorRetrieve", so it can share that runName
// (and the pipeline visualizer's shared "Vector Retrieval" stage copy) instead of needing its own.
export const compressRetrieveAndBuildContext = (filter: Record<string, unknown>) => RunnableSequence.from([
    {
        docs: retriever(filter).withConfig({ runName: "vectorRetrieve" }),
        question: new RunnablePassthrough<string>(),
    },
    RunnableLambda.from(({ question, docs }: { question: string; docs: DocumentInterface[] }) =>
        compressCandidates(question, docs)
    ).withConfig({ runName: "compressChunks" }),
    ({ compressed }: { table: CompressedChunk[]; compressed: CompressibleDoc[] }) => compressed,
    RunnableLambda.from(buildContext),
]).withConfig({ runName: "retrieveAndBuildContext" });

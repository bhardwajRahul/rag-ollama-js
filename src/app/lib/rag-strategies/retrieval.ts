import { RunnableSequence, RunnableLambda } from "@langchain/core/runnables";
import type { DocumentInterface } from "@langchain/core/documents";

import { retriever } from "../supabase";
import { buildContext } from "../../utils/helpers";

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

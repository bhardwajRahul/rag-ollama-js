import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// Children are indexed at a finer granularity than the ~1000-char parent chunks in the
// `documents` table (see api/document/route.ts) so their embeddings are more precise matches
// for narrow queries, while the parent chunk still supplies the answer LLM full surrounding
// context at answer time.
export const CHILD_CHUNK_SIZE = 200;
export const CHILD_CHUNK_OVERLAP = 20;

const childSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHILD_CHUNK_SIZE,
    chunkOverlap: CHILD_CHUNK_OVERLAP,
    separators: ["\n\n", "\n", " ", ""],
});

// One or more child Documents per parent chunk — pageContent is the small child chunk (what
// gets embedded), metadata carries the full parent chunk text. Precomputing the parent text
// here (rather than a parent-lookup query at retrieval time) keeps the retrieval-side
// composition a plain synchronous map over already-fetched docs (see expandToParent in
// rag-strategies/retrieval.ts), not a second DB round-trip — same pattern as sentence-window's
// precomputed windowText.
export async function buildChildDocuments(parentChunks: Document[]): Promise<Document[]> {
    const childGroups = await Promise.all(
        parentChunks.map((parent) =>
            childSplitter.createDocuments(
                [parent.pageContent],
                [{ ...parent.metadata, parentText: parent.pageContent }]
            )
        )
    );
    return childGroups.flat();
}

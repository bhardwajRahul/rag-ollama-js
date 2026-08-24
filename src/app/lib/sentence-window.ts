import { Document } from "@langchain/core/documents";

// Sentences before/after a hit to include in its expanded window (5 sentences total:
// hit + 2 before + 2 after).
export const SENTENCE_WINDOW_SIZE = 2;

// Naive regex sentence splitter: splits on whitespace following a sentence-ending
// punctuation mark. Mis-splits on abbreviations ("Dr. Smith"), decimals ("3.14"), etc. —
// a known simplification accepted to avoid pulling in an NLP sentence-segmentation
// dependency for what's a teaching-focused RAG strategy comparison, not a production
// text pipeline.
export function splitIntoSentences(text: string): string[] {
    return text
        .split(/(?<=[.!?])\s+/)
        .map((sentence) => sentence.trim())
        .filter((sentence) => sentence.length > 0);
}

interface SentencePageHeader {
    documentName: string;
    pageNumber?: number;
    userId: string;
}

// One Document per sentence — pageContent is the bare sentence (what gets embedded),
// metadata carries the precomputed window text for that sentence. Precomputing the window
// here (rather than a neighbor-lookup query at retrieval time) keeps the retrieval-side
// composition a plain synchronous map over already-fetched docs (see expandToWindow in
// rag-strategies/retrieval.ts), not a second DB round-trip per hit.
export function buildSentenceWindowDocuments(pageContent: string, pageHeader: SentencePageHeader): Document[] {
    const sentences = splitIntoSentences(pageContent);
    return sentences.map((sentence, index) => {
        const windowText = sentences
            .slice(Math.max(0, index - SENTENCE_WINDOW_SIZE), index + SENTENCE_WINDOW_SIZE + 1)
            .join(" ");
        return new Document({
            pageContent: sentence,
            metadata: { ...pageHeader, sentenceIndex: index, windowText },
        });
    });
}

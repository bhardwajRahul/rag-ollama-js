import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";

import { embeddings } from "./ollama";
import { env } from "../utils/env";

const { supabase: { url, apiKey } } = env;

export const supabaseClient = createClient(
    url,
    apiKey,
);

export const vectorStore = (filter?: Record<string, unknown>) => new SupabaseVectorStore(embeddings, {
    client: supabaseClient,
    tableName: "documents",
    queryName: "match_documents",
    filter: filter || {}
});

export const retriever = (filter: Record<string, unknown>, k?: number) => vectorStore(filter).asRetriever(k);

export const sentenceVectorStore = (filter?: Record<string, unknown>) => new SupabaseVectorStore(embeddings, {
    client: supabaseClient,
    tableName: "sentence_documents",
    queryName: "match_sentence_documents",
    filter: filter || {}
});

export const sentenceRetriever = (filter: Record<string, unknown>, k?: number) => sentenceVectorStore(filter).asRetriever(k);

interface HybridSearchRow {
    id: number;
    content: string;
    metadata: { pageNumber?: number };
}

// Hybrid retrieval: the hybrid_search RPC (supabaseScripts.txt STEP 9) runs full-text and
// vector search side by side and fuses the two rankings with reciprocal rank fusion. The
// raw query text goes to keyword search while its embedding goes to vector search — this
// is why it can't go through SupabaseVectorStore, which only ever sends the embedding.
export const hybridSearcher = (filter: Record<string, unknown>) => async (query: string) => {
    const queryEmbedding = await embeddings.embedQuery(query);
    const { data, error } = await supabaseClient.rpc("hybrid_search", {
        query_text: query,
        query_embedding: queryEmbedding,
        match_count: 4,
        filter: filter || {},
    });
    if (error) throw new Error(`hybrid_search RPC failed: ${error.message}`);
    return (data as HybridSearchRow[]).map((row) => ({
        pageContent: row.content,
        metadata: row.metadata,
    }));
};
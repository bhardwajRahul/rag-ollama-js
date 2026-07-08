// Client-safe: no LangChain/LLM imports here, so this can be imported from UI components.
export type RagMode = "naive" | "condense" | "multi-query" | "hyde" | "hybrid";

export const DEFAULT_RAG_MODE: RagMode = "condense";

export const RAG_MODES: { value: RagMode; label: string }[] = [
    { value: "condense", label: "Query Condensing" },
    { value: "naive", label: "Naive RAG" },
    { value: "multi-query", label: "Multi-Query" },
    { value: "hyde", label: "HyDE" },
    { value: "hybrid", label: "Hybrid Search" },
];

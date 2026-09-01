import { getChunksByUser, CHUNK_INDEXES, type ChunkIndex } from '@/app/lib/supabase';

export async function GET(req: Request) {
    const userId = req.headers.get('User-Id');
    if (!userId) return new Response('User ID is required', { status: 400 });

    const { searchParams } = new URL(req.url);
    const index = searchParams.get('index');
    if (!index || !CHUNK_INDEXES.includes(index as ChunkIndex)) {
        return new Response('Invalid or missing index', { status: 400 });
    }

    const { data, error } = await getChunksByUser(index as ChunkIndex, userId);
    if (error) return new Response(error.message || 'Chunk fetch failed', { status: 400 });

    return Response.json({ index, count: data.length, chunks: data });
}

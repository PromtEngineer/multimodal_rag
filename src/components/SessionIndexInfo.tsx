import { useEffect, useState } from 'react';
import { chatAPI, ChatSession } from '@/lib/api';

interface Props {
  sessionId: string;
  onClose: () => void;
}

export default function SessionIndexInfo({ sessionId, onClose }: Props) {
  const [files, setFiles] = useState<string[]>([]);
  const [indexMeta, setIndexMeta] = useState<any | null>(null);
  const [session, setSession] = useState<ChatSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const data = await chatAPI.getSessionIndexes(sessionId);
        const first = data.indexes[0];
        if(first){
          setSession(first.session??{...first, title:first.name, model_used:first.model_used||''});
          setFiles(first.documents?.map((d:any)=>d.filename) || []);
          setIndexMeta(first.metadata || {});
        } else {
          setError('No indexes linked to this chat');
        }
      } catch (e:any){ setError(e.message||'Failed to load'); }
      finally{ setLoading(false);}
    })();
  }, [sessionId]);

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-50 p-4">
      <div className="relative bg-white/5 backdrop-blur rounded-xl p-8 w-full max-w-lg text-white space-y-6 overflow-y-auto max-h-full">
        <h2 className="text-lg font-semibold">Index details</h2>

        {loading && <p className="text-sm text-gray-300">Loading…</p>}
        {error && <p className="text-sm text-red-400">{error}</p>}

        {(!loading && !error) && (
          <>
            <div>
              <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Name</span>
              <p className="text-sm">{session?.title}</p>
            </div>
            <div>
              <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Model</span>
              <p className="text-sm">{session?.model_used}</p>
            </div>
            {indexMeta && (
              <div className="space-y-3">
                {indexMeta.embedding_model && (
                  <div>
                    <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Embedding model</span>
                    <p className="text-sm">{indexMeta.embedding_model}</p>
                  </div>
                )}
                {indexMeta.retrieval_mode && (
                  <div>
                    <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Retrieval mode</span>
                    <p className="text-sm capitalize">{indexMeta.retrieval_mode}</p>
                  </div>
                )}
                {typeof indexMeta.chunk_size==='number' && (
                  <div>
                    <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Chunk size</span>
                    <p className="text-sm">{indexMeta.chunk_size} tokens</p>
                  </div>
                )}
                {typeof indexMeta.window_size==='number' && (
                  <div>
                    <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Context window</span>
                    <p className="text-sm">{indexMeta.window_size}</p>
                  </div>
                )}
              </div>
            )}
            <div>
              <span className="block text-xs uppercase tracking-wide text-gray-300 mb-1">Files ({files.length})</span>
              <ul className="list-disc list-inside space-y-1 text-sm">
                {files.map((f) => (
                  <li key={f}>{f}</li>
                ))}
              </ul>
            </div>
            {/* Future: add chunk size etc. */}
          </>
        )}

        <div className="flex justify-end pt-4 border-t border-white/10">
          <button onClick={onClose} className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 text-sm">Close</button>
        </div>
      </div>
    </div>
  );
} 
import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  deleteDocument,
  getDocument,
  listDocuments,
  uploadDocument,
  type Document,
} from "../services/api";

export default function UploadPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  // ── Polling ──────────────────────────────────────────────────────────────
  const pollingRef = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map());

  const stopPolling = (id: string) => {
    const timer = pollingRef.current.get(id);
    if (timer) {
      clearInterval(timer);
      pollingRef.current.delete(id);
    }
  };

  const startPolling = useCallback((id: string) => {
    if (pollingRef.current.has(id)) return;
    const timer = setInterval(async () => {
      try {
        const doc = await getDocument(id);
        setDocuments((prev) => prev.map((d) => (d.id === id ? doc : d)));
        if (doc.status !== "processing") stopPolling(id);
      } catch {
        stopPolling(id);
      }
    }, 2000);
    pollingRef.current.set(id, timer);
  }, []);

  useEffect(() => {
    listDocuments().then((docs) => {
      setDocuments(docs);
      docs.filter((d) => d.status === "processing").forEach((d) => startPolling(d.id));
    });
    return () => {
      pollingRef.current.forEach((_, id) => stopPolling(id));
    };
  }, [startPolling]);

  // ── Upload ────────────────────────────────────────────────────────────────
  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setError(null);
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        const doc = await uploadDocument(file);
        setDocuments((prev) => [doc, ...prev]);
        startPolling(doc.id);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  const handleDelete = async (id: string) => {
    stopPolling(id);
    await deleteDocument(id);
    setDocuments((prev) => prev.filter((d) => d.id !== id));
  };

  const readyDocs = documents.filter((d) => d.status === "ready");

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-2xl">
        <h1 className="text-3xl font-bold text-gray-900 mb-1">RAG Note App</h1>
        <p className="text-gray-500 mb-8">Upload PDFs, then ask questions about them.</p>

        {/* Drop zone */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => fileRef.current?.click()}
          className="border-2 border-dashed border-blue-300 rounded-2xl p-10 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-colors"
        >
          <input
            ref={fileRef}
            type="file"
            accept=".pdf"
            multiple
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
          {uploading ? (
            <p className="text-blue-600 font-medium animate-pulse">Uploading…</p>
          ) : (
            <>
              <p className="text-gray-600 font-medium">Drop PDF files here</p>
              <p className="text-gray-400 text-sm mt-1">or click to browse</p>
            </>
          )}
        </div>

        {error && (
          <div className="mt-3 bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-2 text-sm">
            {error}
          </div>
        )}

        {/* Document list */}
        {documents.length > 0 && (
          <div className="mt-8">
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
              Your Documents
            </h2>
            <ul className="space-y-2">
              {documents.map((doc) => (
                <li
                  key={doc.id}
                  className="bg-white rounded-xl border border-gray-200 px-4 py-3 flex items-center justify-between shadow-sm"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span className="text-2xl">
                      {doc.status === "ready"
                        ? "📄"
                        : doc.status === "processing"
                        ? "⏳"
                        : "❌"}
                    </span>
                    <div className="min-w-0">
                      <p className="font-medium text-gray-800 truncate">{doc.filename}</p>
                      <p className="text-xs text-gray-400">
                        {doc.status === "ready"
                          ? `${doc.page_count} pages · ${doc.chunk_count} chunks`
                          : doc.status === "processing"
                          ? "Processing…"
                          : doc.error_msg ?? "Failed"}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 ml-3 shrink-0">
                    {doc.status === "ready" && (
                      <button
                        onClick={() => navigate(`/chat?doc=${doc.id}`)}
                        className="text-xs bg-blue-600 text-white px-3 py-1.5 rounded-lg hover:bg-blue-700 transition-colors"
                      >
                        Chat
                      </button>
                    )}
                    <button
                      onClick={() => handleDelete(doc.id)}
                      className="text-xs text-gray-400 hover:text-red-500 px-2 py-1.5 rounded-lg hover:bg-red-50 transition-colors"
                    >
                      Delete
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* CTA when docs are ready */}
        {readyDocs.length > 0 && (
          <div className="mt-6 text-center">
            <button
              onClick={() => {
                const ids = readyDocs.map((d) => d.id).join(",");
                navigate(`/chat?docs=${ids}`);
              }}
              className="bg-gray-900 text-white px-6 py-2.5 rounded-xl font-medium hover:bg-gray-700 transition-colors"
            >
              Chat with all {readyDocs.length} document{readyDocs.length > 1 ? "s" : ""}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
